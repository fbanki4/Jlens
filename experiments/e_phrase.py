"""A non-trained way to get the bag of concepts.

The J-lens names concepts that are single tokens. Here half the cities are two
words long, so half the latent intermediates in the two-hop task have no J-lens
vector at all — the same situation as ``blackmail`` or ``photosynthesis`` in a
real model, reproduced where we know the ground truth.

Five ways to get a direction for a phrase, compared head to head:

    first-token       J-lens vector of the phrase's first token   (baseline)
    token-average     mean of the constituent tokens' vectors     (baseline)
    template          whitened mean activation over contexts where the phrase
                      is the natural continuation                 (the paper's
                      method: needs authored contexts per phrase)
    phrase-Jacobian   E[ d log p(phrase) / d h ] over the corpus  (UNTRAINED)
    debiased-sum      sum_i ( v_{w_i} - E_{p_i}[v] )              (UNTRAINED)

Two evaluations, matching the paper's readout/swap pair:

    DECODE  rank the true latent among all candidate cities by cosine against
            the query-position activation, split by phrase token length.
    SWAP    replace the latent's vector with a distractor's in the workspace
            band and check whether the answer follows.

The interesting comparison is not "does the untrained method beat everything",
it is "does dropping the trained artifact cost anything".
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from jlens.cka import build_probe
from jlens.jspace import swap_coordinates
from jlens.lens import JacobianLens
from jlens.phrases import (
    build_debiased_vectors,
    fit_phrase_vectors,
    first_token_vector,
    token_average_vector,
)
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import (
    COMPOSED_RELATIONS,
    build_corpus,
    build_world,
    composed_prompt,
    filler_sentence,
    make_split,
)
from experiments.fit_toy_lens import fitting_corpus


# ----------------------------------------------------------------------------
# The paper's template lens, as the reference the untrained methods must match
# ----------------------------------------------------------------------------


def build_context_index(world, split, n_sentences: int = 40_000, seed: int = 5):
    """Mine the corpus for prefixes ending right where each phrase is next.

    The paper's recipe asks an LLM for passages whose natural continuation is
    *w*, never using *w* itself, then averages the residual at the final
    position. Mining the model's own training distribution is the same thing
    without the authoring step: scan the corpus, and every time *w* occurs,
    keep the text before it.

    Doing it this way keeps the comparison honest in both directions. The
    template lens gets perfectly on-distribution contexts in unlimited supply
    — the best case it could ask for — and it still needs this corpus pass per
    phrase, which is the cost the untrained methods are trying to avoid.
    """
    corpus = [text for text, _ in build_corpus(world, split, n_sentences=n_sentences,
                                               seed=seed)]
    index: dict[str, list[str]] = defaultdict(list)
    for text in corpus:
        words = text.split()
        for i, wd in enumerate(words):
            if i == 0:
                continue
            index[wd].append(" ".join(words[:i]))
            if i + 1 < len(words):
                index[f"{wd} {words[i + 1]}"].append(" ".join(words[:i]))
    return index


def template_contexts(index, phrase: str, n: int) -> list[str]:
    ctx = index.get(phrase, [])
    return ctx[:n]


@torch.no_grad()
def template_vectors(
    model: ToyLensModel,
    index,
    phrases: list[str],
    layers: list[int],
    *,
    n_contexts: int = 40,
    ridge: float = 1e-2,
) -> dict[str, dict[int, torch.Tensor]]:
    """Whitened template vectors: ``(Sigma + lambda I)^-1 (mu_w - mu)``.

    The whitening is what makes this work at all — it turns a class mean into a
    linear discriminant. Stein's lemma is the reason it approximates a
    gradient, and the reason the untrained phrase-Jacobian (which *is* a
    gradient) should land in the same place.
    """
    from jlens.hooks import ActivationRecorder

    per_phrase_acts: dict[str, dict[int, list[torch.Tensor]]] = {
        p: {l: [] for l in layers} for p in phrases
    }
    pooled: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    usable = []
    for phrase in phrases:
        contexts = template_contexts(index, phrase, n_contexts)
        if not contexts:
            continue
        usable.append(phrase)
        for text in contexts:
            ids = model.encode(text)
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(ids)
                for l in layers:
                    h = rec.activations[l][0, -1].detach().float()
                    per_phrase_acts[phrase][l].append(h)
                    pooled[l].append(h)

    out: dict[str, dict[int, torch.Tensor]] = {p: {} for p in usable}
    for l in layers:
        X = torch.stack(pooled[l])
        mu = X.mean(dim=0)
        Xc = X - mu
        Sigma = (Xc.T @ Xc) / max(len(X) - 1, 1)
        A = torch.linalg.inv(Sigma + ridge * torch.eye(Sigma.shape[0]))
        for phrase in usable:
            mu_w = torch.stack(per_phrase_acts[phrase][l]).mean(dim=0)
            out[phrase][l] = A @ (mu_w - mu)
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


@torch.no_grad()
def query_activations(model: ToyLensModel, prompt: str, layers: list[int]):
    from jlens.hooks import ActivationRecorder

    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
        return {l: rec.activations[l][0, -1].detach().float() for l in layers}


def decode_rank(
    vectors: dict[str, torch.Tensor], target: str, h: torch.Tensor
) -> int:
    names = list(vectors)
    M = torch.stack([vectors[n].float() for n in names])
    M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scores = M @ (h / h.norm().clamp_min(1e-12))
    return [names[int(i)] for i in scores.argsort(descending=True)].index(target) + 1


@torch.no_grad()
def swap_with_vectors(
    model: ToyLensModel,
    prompt: str,
    v_src: dict[int, torch.Tensor],
    v_tgt: dict[int, torch.Tensor],
    layers: list[int],
    alpha: float = 1.0,
) -> str:
    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            patched = t.clone()
            patched[0, -1] = swap_coordinates(
                t[0, -1], v_src[layer], v_tgt[layer], alpha=alpha
            )
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        return hook

    try:
        for l in layers:
            handles.append(model.layers[l].register_forward_hook(make_hook(l)))
        logits = model.unembed(model.forward(model.encode(prompt)))
        return model.tokenizer.decode([int(logits[0, -1].argmax())])
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/toy/final.pt")
    ap.add_argument("--lens", default="runs/lens/final_lens.pt")
    ap.add_argument("--n-corpus", type=int, default=40)
    ap.add_argument("--out", default="runs/phrase")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    net, tok = load_toy(args.model)
    model = ToyLensModel(net, tok)
    lens = JacobianLens.load(args.lens)
    world, split = build_world(), make_split(build_world())
    n_layers = model.n_layers
    band = [l for l in lens.source_layers if 0.3 <= l / (n_layers - 1) <= 0.8]
    print(f"workspace band for interventions: {band}")

    cities = list(world.entities["city"])
    # Rank against every entity in the world, not just cities. With a 24-way
    # choice every method scores ~95% and the comparison says nothing; the
    # paper ranks against the whole vocabulary for the same reason. The pool
    # includes the answer country and the subject, which are the competitors
    # that actually matter.
    candidates = sorted({e for group in world.entities.values() for e in group})
    corpus = fitting_corpus(args.n_corpus)

    # ---- build the five vector sets --------------------------------------
    P_full, _ = build_probe(model, token_ids=torch.arange(len(tok)))
    V_by_layer = {l: (P_full @ lens.jacobians[l]).float() for l in lens.source_layers}

    methods: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    methods["first-token"] = {
        c: {l: first_token_vector(V_by_layer[l], tok.encode(c)) for l in lens.source_layers}
        for c in candidates
    }
    methods["token-average"] = {
        c: {l: token_average_vector(V_by_layer[l], tok.encode(c)) for l in lens.source_layers}
        for c in candidates
    }
    print("building template vectors (the trained-data reference) ...")
    context_index = build_context_index(world, split)
    methods["template"] = template_vectors(
        model, context_index, candidates, lens.source_layers, n_contexts=30
    )
    print(f"    template vectors available for "
          f"{len(methods['template'])}/{len(candidates)} candidates")
    print("building phrase-Jacobian vectors (untrained) ...")
    methods["phrase-Jacobian"] = fit_phrase_vectors(
        model, candidates, corpus, lens.source_layers, max_seq_len=16, skip_first=1
    )
    print("building debiased-sum vectors (untrained) ...")
    methods["debiased-sum"] = build_debiased_vectors(
        model, lens, candidates, corpus[:12], lens.source_layers, max_seq_len=16
    )

    # ---- items the model actually solves ---------------------------------
    composed = world.composed_map("hails_from")
    latents = world.latent_map("hails_from")
    items = []
    for subject, answer in composed.items():
        prompt = composed_prompt("hails_from", subject)
        with torch.no_grad():
            pred = int(model.unembed(model.forward(model.encode(prompt)))[0, -1].argmax())
        if pred == tok.token_id(answer):
            items.append(
                {"prompt": prompt, "latent": latents[subject], "answer": answer,
                 "subject": subject}
            )
    n_multi = sum(len(i["latent"].split()) > 1 for i in items)
    print(f"\nanalysing {len(items)} subjects the model answers correctly "
          f"({n_multi} with a two-token latent city)")

    # ---- DECODE ----------------------------------------------------------
    per_method = {m: defaultdict(list) for m in methods}
    for item in items:
        acts = query_activations(model, item["prompt"], lens.source_layers)
        length = len(item["latent"].split())
        for m, vecs in methods.items():
            best = min(
                decode_rank({c: vecs[c][l] for c in candidates if c in vecs},
                            item["latent"], acts[l])
                for l in band
            )
            per_method[m][length].append(best)

    print()
    print("=" * 78)
    print(f"DECODE: rank of the true latent among all {len(candidates)} world entities")
    print("        (best over the workspace band; 1 = correct concept on top)")
    print("=" * 78)
    print(f"  {'method':17s} | {'1-token latent':>26s} | {'2-token latent':>26s}")
    print(f"  {'':17s} | {'top-1':>8s} {'top-3':>8s} {'med':>7s} "
          f"| {'top-1':>8s} {'top-3':>8s} {'med':>7s}")
    decode_summary = {}
    for m in ("first-token", "token-average", "template", "phrase-Jacobian", "debiased-sum"):
        cells, row = [], {}
        for length in (1, 2):
            r = np.array(per_method[m][length])
            if len(r) == 0:
                cells.append(f"{'-':>8s} {'-':>8s} {'-':>7s}")
                continue
            t1, t3, med = (r == 1).mean(), (r <= 3).mean(), np.median(r)
            row[length] = {"n": int(len(r)), "top1": float(t1), "top3": float(t3),
                           "median_rank": float(med)}
            cells.append(f"{t1:8.0%} {t3:8.0%} {med:7.0f}")
        decode_summary[m] = row
        print(f"  {m:17s} | {cells[0]:>26s} | {cells[1]:>26s}")

    # ---- SWAP ------------------------------------------------------------
    print()
    print("=" * 78)
    print("SWAP: replace the latent's vector with a distractor's in the band")
    print("=" * 78)
    city_country = world.atomic["city_in"]
    rng = random.Random(0)
    targets = {
        item["subject"]: rng.choice(
            [c for c in cities
             if c != item["latent"] and city_country[c] != item["answer"]]
        )
        for item in items
    }
    # Sweep where and how hard to intervene. This model finishes most of its
    # computation early, so a band chosen by relative depth alone can land
    # after the argument has already been consumed.
    bands = {
        "early L1-4": [l for l in lens.source_layers if 1 <= l <= 4],
        "mid L4-8": band,
        "all L1-10": [l for l in lens.source_layers if l >= 1],
    }
    swap_summary: dict[str, dict] = {}
    for band_name, band_layers in bands.items():
        print(f"\n  band {band_name} = {band_layers}")
        for m, vecs in methods.items():
            row = {}
            for alpha in (1.0, 2.0, 4.0):
                per_len = defaultdict(lambda: [0, 0])
                follow = total = 0
                for item in items:
                    target_city = targets[item["subject"]]
                    got = swap_with_vectors(
                        model, item["prompt"],
                        {l: vecs[item["latent"]][l] for l in band_layers},
                        {l: vecs[target_city][l] for l in band_layers},
                        band_layers, alpha=alpha,
                    )
                    total += 1
                    hit = got == city_country[target_city]
                    follow += hit
                    key = len(item["latent"].split())
                    per_len[key][1] += 1
                    per_len[key][0] += hit
                row[f"alpha{alpha}"] = {
                    "n": total, "follow": follow, "rate": follow / max(total, 1),
                    "by_length": {str(k): v for k, v in per_len.items()},
                }
            swap_summary[f"{m}|{band_name}"] = row
            cells = "   ".join(
                f"a={a.replace('alpha', '')}: {row[a]['rate']:5.1%} "
                f"({'/'.join(f'{h / max(t, 1):.0%}' for _, (h, t) in sorted(row[a]['by_length'].items()))})"
                for a in row
            )
            print(f"    {m:17s} {cells}")

    with open(os.path.join(args.out, "phrase.json"), "w") as f:
        json.dump(
            {"decode": decode_summary, "swap": swap_summary,
             "band": band, "n_items": len(items)},
            f, indent=2,
        )
    print(f"\nwrote {args.out}/phrase.json")


if __name__ == "__main__":
    main()
