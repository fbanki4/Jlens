"""Does the J-lens surface an unverbalized intermediate? And does it steer?

This is the paper's central read/write pair, at a scale where ground truth is
known. The model has memorised two atomic relations — ``person lives in city``
and ``city sits in country`` — and has been trained on the composed relation
``person hails from country`` for three quarters of the people. The remaining
quarter is held out entirely.

On a held-out subject, the model can only answer by composing, so the city must
be computed somewhere inside. And the city appears **nowhere in the prompt**:
``alice hails from`` is three tokens. That is exactly the structure of the
paper's headline example — "the color of the planet fourth from the sun is",
where ``Mars`` is neither asked for nor answered.

Two measurements:

**Read.** How highly does the lens rank the latent across layers, against three
controls: the same reading with the transport off (the plain logit lens), other
cities (which are equally absent from the prompt), and the answer itself.

**Write.** Swap the latent's J-lens vector for another city's at the workspace
layers, leaving everything orthogonal to those two directions untouched, and
see whether the answer follows to *that* city's country. Reading shows the
concept is present; only the swap shows it is the argument being consumed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from jlens.cka import build_probe, workspace_band
from jlens.jspace import swap_coordinates
from jlens.lens import JacobianLens
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import (
    COMPOSED_RELATIONS,
    build_world,
    composed_prompt,
    make_split,
)


def build_items(world, split, relation: str) -> list[dict]:
    composed = world.composed_map(relation)
    latents = world.latent_map(relation)
    held = set(split.heldout[relation])
    return [
        {
            "relation": relation,
            "subject": s,
            "prompt": composed_prompt(relation, s),
            "latent": latents[s],
            "answer": composed[s],
            "heldout": s in held,
        }
        for s in composed
    ]


@torch.no_grad()
def lens_ranks(
    model: ToyLensModel,
    lens: JacobianLens,
    prompt: str,
    targets: list[str],
    *,
    use_jacobian: bool = True,
) -> dict[str, list[int]]:
    """Rank of each target's first token at the last position, per layer."""
    lens_logits, _, _ = lens.apply(
        model, prompt, positions=[-1], use_jacobian=use_jacobian
    )
    out: dict[str, list[int]] = {t: [] for t in targets}
    for layer in lens.source_layers:
        order = lens_logits[layer][0].argsort(descending=True).tolist()
        rank_of = {tid: i + 1 for i, tid in enumerate(order)}
        for t in targets:
            out[t].append(rank_of[model.tokenizer.token_id(t.split()[0])])
    return out


@torch.no_grad()
def answer_after_swap(
    model: ToyLensModel,
    lens: JacobianLens,
    prompt: str,
    V: dict[int, torch.Tensor],
    source: str,
    target: str,
    layers: list[int],
    *,
    alpha: float = 1.0,
) -> str:
    src = model.tokenizer.token_id(source.split()[0])
    tgt = model.tokenizer.token_id(target.split()[0])
    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            patched = t.clone()
            patched[0, -1] = swap_coordinates(
                t[0, -1], V[layer][src], V[layer][tgt], alpha=alpha
            )
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        return hook

    try:
        for layer in layers:
            handles.append(model.layers[layer].register_forward_hook(make_hook(layer)))
        logits = model.unembed(model.forward(model.encode(prompt)))
        return model.tokenizer.decode([int(logits[0, -1].argmax())])
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/toy/final.pt")
    ap.add_argument("--lens", default="runs/lens/final_lens.pt")
    ap.add_argument("--relation", default="hails_from")
    ap.add_argument("--out", default="runs/multihop")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    net, tok = load_toy(args.model)
    model = ToyLensModel(net, tok)
    lens = JacobianLens.load(args.lens)
    world, split = build_world(), make_split(build_world())
    n_layers, layers = model.n_layers, lens.source_layers
    band = workspace_band(lens.source_layers, n_layers)

    P_full, _ = build_probe(model, token_ids=torch.arange(len(tok)))
    V = {l: (P_full @ lens.jacobians[l]).float() for l in layers}

    items = build_items(world, split, args.relation)
    correct = []
    for it in items:
        with torch.no_grad():
            pred = int(
                model.unembed(model.forward(model.encode(it["prompt"])))[0, -1].argmax()
            )
        it["correct"] = pred == tok.token_id(it["answer"])
        if it["correct"]:
            correct.append(it)
    n_held = sum(i["heldout"] for i in items)
    n_held_ok = sum(i["heldout"] for i in correct)
    print(f"relation {args.relation!r}: model correct on "
          f"{len(correct)}/{len(items)} subjects "
          f"({n_held_ok}/{n_held} held out; chance "
          f"{1 / len(world.entities['country']):.3f})")
    if len(correct) < 10:
        raise SystemExit("model does not solve the task; train longer")

    latent_type = COMPOSED_RELATIONS[args.relation][2]
    pool = world.entities[
        {"lives_in": "city", "city_in": "country", "made_of": "food"}[latent_type]
    ]

    # ---------------- read ----------------
    rows = defaultdict(list)
    for it in correct:
        distractors = random.Random(hash(it["subject"]) % 997).sample(
            [c for c in pool if c != it["latent"]], 5
        )
        targets = [it["latent"], it["answer"], *distractors]
        rj = lens_ranks(model, lens, it["prompt"], targets)
        rl = lens_ranks(model, lens, it["prompt"], targets, use_jacobian=False)
        rows["latent_j"].append(rj[it["latent"]])
        rows["answer_j"].append(rj[it["answer"]])
        rows["latent_logit"].append(rl[it["latent"]])
        rows["distractor_j"].append(
            np.mean([rj[d] for d in distractors], axis=0).tolist()
        )
        rows["distractor_logit"].append(
            np.mean([rl[d] for d in distractors], axis=0).tolist()
        )
        rows["heldout"].append(it["heldout"])
        rows["ntok"].append(len(it["latent"].split()))
    arr = {k: np.array(v) for k, v in rows.items() if k not in ("heldout", "ntok")}
    heldout = np.array(rows["heldout"], dtype=bool)
    ntok = np.array(rows["ntok"])

    print()
    print("=" * 76)
    print("READ: median rank of each token at the query position (1 = top)")
    print("=" * 76)
    print(f"  {'layer':>5s} {'rho':>5s} | {'latent':>7s} {'distract':>8s} "
          f"{'answer':>7s} | {'latent':>7s}  <- logit lens")
    for i, layer in enumerate(layers):
        print(f"  {layer:5d} {layer / (n_layers - 1):5.2f} | "
              f"{np.median(arr['latent_j'][:, i]):7.0f} "
              f"{np.median(arr['distractor_j'][:, i]):8.0f} "
              f"{np.median(arr['answer_j'][:, i]):7.0f} | "
              f"{np.median(arr['latent_logit'][:, i]):7.0f}")

    band_idx = [layers.index(l) for l in band]
    best_j = arr["latent_j"][:, band_idx].min(axis=1)
    best_l = arr["latent_logit"][:, band_idx].min(axis=1)
    best_d = arr["distractor_j"][:, band_idx].min(axis=1)
    print()
    print(f"  best rank over the workspace band, median: "
          f"latent {np.median(best_j):.0f}, distractors {np.median(best_d):.0f}, "
          f"logit-lens latent {np.median(best_l):.0f}")
    print(f"  latent in the lens top-3 somewhere in the band: "
          f"{(best_j <= 3).mean():.0%}  (logit lens {(best_l <= 3).mean():.0%})")
    best_dl = arr["distractor_logit"][:, band_idx].min(axis=1)
    print(f"  latent outranks the mean distractor: J-lens "
          f"{(best_j < best_d).mean():.0%}, logit lens {(best_l < best_dl).mean():.0%}")
    print(f"  specificity (median distractor rank / median latent rank) in band: "
          f"J-lens {np.median(best_d) / max(np.median(best_j), 1):.1f}x, "
          f"logit lens {np.median(best_dl) / max(np.median(best_l), 1):.1f}x")
    for name, sel in (("held-out subjects", heldout), ("trained subjects", ~heldout)):
        if sel.any():
            print(f"    {name:18s} n={sel.sum():3d}  latent top-3 "
                  f"{(best_j[sel] <= 3).mean():.0%}")
    for length in (1, 2):
        sel = ntok == length
        if sel.any():
            print(f"    {length}-token latent    n={sel.sum():3d}  latent top-3 "
                  f"{(best_j[sel] <= 3).mean():.0%}")

    # ---------------- write ----------------
    print()
    print("=" * 76)
    print("WRITE: swap the latent's vector for another entity's, in the band")
    print("=" * 76)
    outer = COMPOSED_RELATIONS[args.relation][1]
    outer_map = world.atomic[outer]
    rng = random.Random(0)
    results, control = defaultdict(int), defaultdict(int)
    per_len = defaultdict(lambda: [0, 0])
    for it in correct:
        alternatives = [
            c for c in pool
            if c != it["latent"] and outer_map[c] != it["answer"]
        ]
        target = rng.choice(alternatives)
        got = answer_after_swap(
            model, lens, it["prompt"], V, it["latent"], target, band
        )
        results["n"] += 1
        key = len(it["latent"].split())
        per_len[key][1] += 1
        if got == outer_map[target]:
            results["follow"] += 1
            per_len[key][0] += 1
        elif got == it["answer"]:
            results["unchanged"] += 1
        else:
            results["other"] += 1
        # Control: swap two tokens irrelevant to the task, same layers.
        got_c = answer_after_swap(
            model, lens, it["prompt"], V,
            rng.choice(["morning", "river", "market", "letter"]),
            rng.choice(["walked", "waited", "smiled", "paused"]),
            band,
        )
        control["n"] += 1
        if got_c == outer_map[target]:
            control["follow"] += 1
        elif got_c == it["answer"]:
            control["unchanged"] += 1
        else:
            control["other"] += 1

    for name, r in (("latent -> other entity", results), ("irrelevant tokens", control)):
        n = max(r["n"], 1)
        print(f"  {name:23s} n={r['n']:3d}  follows the swap {r['follow'] / n:6.1%}  "
              f"unchanged {r['unchanged'] / n:6.1%}  other {r['other'] / n:6.1%}")
    for length, (hit, tot) in sorted(per_len.items()):
        if tot:
            print(f"    {length}-token latent   n={tot:3d}  follows "
                  f"{hit / tot:6.1%}")

    summary = {
        "relation": args.relation,
        "n_items": len(correct),
        "n_heldout_correct": int(n_held_ok),
        "n_heldout": int(n_held),
        "layers": layers,
        "band": band,
        "median_rank_latent_jlens": np.median(arr["latent_j"], axis=0).tolist(),
        "median_rank_latent_logitlens": np.median(arr["latent_logit"], axis=0).tolist(),
        "median_rank_distractor_jlens": np.median(arr["distractor_j"], axis=0).tolist(),
        "median_rank_distractor_logitlens": np.median(arr["distractor_logit"], axis=0).tolist(),
        "median_rank_answer_jlens": np.median(arr["answer_j"], axis=0).tolist(),
        "latent_top3_in_band": float((best_j <= 3).mean()),
        "latent_top3_logitlens": float((best_l <= 3).mean()),
        "swap": dict(results),
        "swap_control": dict(control),
        "swap_by_latent_length": {str(k): v for k, v in per_len.items()},
    }
    with open(os.path.join(args.out, "multihop.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}/multihop.json")


if __name__ == "__main__":
    main()
