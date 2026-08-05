"""Does the J-lens surface an unverbalized intermediate? And does it steer?

This is the paper's central read/write pair, at a scale where ground truth is
known. Each benchmark item states some ``person lives in city`` and
``city sits in country`` facts in shuffled order, then asks
``<subject> hails from``. Answering needs subject -> city -> country. The city
is the *latent*: at the query position it is neither the current token nor the
answer.

Two measurements:

**Read.** At the query position, how highly does the lens rank the latent city
across layers — compared against (a) the same reading with the transport turned
off, which is the plain logit lens, and (b) the distractor cities, which appear
in the prompt exactly as often as the latent does. The distractor control is
the one that matters: without it, "the lens shows a city" could just mean "the
lens echoes tokens present in the context".

**Write.** Swap the latent city's J-lens vector for a distractor's at the
workspace layers, leaving everything orthogonal to those two directions
untouched, and see whether the model's answer follows to the distractor's
country. Reading is evidence that the concept is present; only the swap is
evidence that it is the argument being consumed downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from jlens.cka import build_probe
from jlens.hooks import ActivationRecorder
from jlens.jspace import swap_coordinates
from jlens.lens import JacobianLens
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import build_world, chain_benchmark


@torch.no_grad()
def lens_ranks_at_query(
    model: ToyLensModel,
    lens: JacobianLens,
    prompt: str,
    targets: list[str],
    *,
    use_jacobian: bool = True,
) -> dict[str, list[int]]:
    """Rank of each target token at the last position, per layer (1 = top)."""
    lens_logits, _, _ = lens.apply(
        model, prompt, positions=[-1], use_jacobian=use_jacobian
    )
    out: dict[str, list[int]] = {t: [] for t in targets}
    for layer in lens.source_layers:
        scores = lens_logits[layer][0]
        order = scores.argsort(descending=True).tolist()
        rank_of = {tid: i + 1 for i, tid in enumerate(order)}
        for t in targets:
            out[t].append(rank_of[model.tokenizer.token_id(t)])
    return out


@torch.no_grad()
def answer_after_swap(
    model: ToyLensModel,
    lens: JacobianLens,
    prompt: str,
    source_token: str,
    target_token: str,
    layers: list[int],
    *,
    alpha: float = 1.0,
) -> str:
    """Run the model with a J-lens coordinate swap applied at ``layers``.

    Hooks each named block and rewrites its output at the final position only,
    so the swap acts on the query's workspace and nothing else.
    """
    V = {}
    P_full, _ = build_probe(model, token_ids=torch.arange(len(model.tokenizer)))
    src_id = model.tokenizer.token_id(source_token)
    tgt_id = model.tokenizer.token_id(target_token)
    for layer in layers:
        Vl = (P_full @ lens.jacobians[layer]).float()
        V[layer] = (Vl[src_id], Vl[tgt_id])

    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            v_s, v_t = V[layer]
            patched = tensor.clone()
            patched[0, -1] = swap_coordinates(
                tensor[0, -1], v_s, v_t, alpha=alpha
            )
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        return hook

    try:
        for layer in layers:
            handles.append(model.layers[layer].register_forward_hook(make_hook(layer)))
        ids = model.encode(prompt)
        logits = model.unembed(model.forward(ids))
        return model.tokenizer.decode([int(logits[0, -1].argmax())])
    finally:
        for h in handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", default="runs/lens/final_lens.pt")
    ap.add_argument("--model", default="runs/toy/final.pt")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="runs/multihop")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    net, tok = load_toy(args.model)
    model = ToyLensModel(net, tok)
    lens = JacobianLens.load(args.lens)
    n_layers = model.n_layers
    layers = lens.source_layers

    chains = [
        c for c in chain_benchmark(build_world(), n=args.n * 2, k=3)
    ]
    # Keep only items the model actually gets right: on items it fails, there
    # is no reason to expect a correct intermediate to be present.
    correct = []
    for c in chains:
        ids = model.encode(c.prompt)
        with torch.no_grad():
            pred = int(model.unembed(model.forward(ids))[0, -1].argmax())
        if pred == tok.token_id(c.answer):
            correct.append(c)
        if len(correct) >= args.n:
            break
    print(f"model answers {len(correct)}/{len(chains)} probed items correctly; "
          f"analysing {len(correct)}")
    if not correct:
        raise SystemExit("model does not solve the task; train longer")

    # ---------------- read ----------------
    latent_j, latent_logit, distractor_j, answer_j = [], [], [], []
    for c in correct:
        targets = [c.latent, c.answer] + c.distractor_cities
        rj = lens_ranks_at_query(model, lens, c.prompt, targets)
        rl = lens_ranks_at_query(model, lens, c.prompt, targets, use_jacobian=False)
        latent_j.append(rj[c.latent])
        answer_j.append(rj[c.answer])
        latent_logit.append(rl[c.latent])
        distractor_j.append(
            np.mean([rj[d] for d in c.distractor_cities], axis=0).tolist()
        )

    latent_j = np.array(latent_j)
    answer_j = np.array(answer_j)
    latent_logit = np.array(latent_logit)
    distractor_j = np.array(distractor_j)

    print()
    print("=" * 74)
    print("READ: median rank at the query position (1 = top of the readout)")
    print("=" * 74)
    print(f"  {'layer':>5s} {'rho':>5s} | {'latent':>7s} {'distract':>8s} "
          f"{'answer':>7s} | {'latent':>7s}   <- logit lens")
    for i, layer in enumerate(layers):
        print(f"  {layer:5d} {layer / (n_layers - 1):5.2f} | "
              f"{np.median(latent_j[:, i]):7.0f} "
              f"{np.median(distractor_j[:, i]):8.0f} "
              f"{np.median(answer_j[:, i]):7.0f} | "
              f"{np.median(latent_logit[:, i]):7.0f}")

    best_layer_idx = int(np.argmin(np.median(latent_j, axis=0)))
    best_layer = layers[best_layer_idx]
    top1 = float((latent_j.min(axis=1) == 1).mean())
    top3 = float((latent_j.min(axis=1) <= 3).mean())
    sep = float(
        (latent_j[:, best_layer_idx] < distractor_j[:, best_layer_idx]).mean()
    )
    print()
    print(f"  latent city reaches rank 1 somewhere in depth on "
          f"{top1:.0%} of items, top-3 on {top3:.0%}")
    print(f"  best layer L{best_layer} (rho={best_layer / (n_layers - 1):.2f}); "
          f"there the latent outranks the mean distractor on {sep:.0%} of items")
    print(f"  logit-lens best median rank {np.median(latent_logit, axis=0).min():.0f} "
          f"vs J-lens best median rank {np.median(latent_j, axis=0).min():.0f}")

    # ---------------- write ----------------
    print()
    print("=" * 74)
    print("WRITE: swap the latent city for a distractor at the workspace layers")
    print("=" * 74)
    band = [l for l in layers if 0.3 <= l / (n_layers - 1) <= 0.75]
    results = {"follow": 0, "unchanged": 0, "other": 0, "n": 0}
    control = {"follow": 0, "unchanged": 0, "other": 0, "n": 0}
    rng = random.Random(0)
    for c in correct:
        if not c.distractor_cities:
            continue
        target_city = c.distractor_cities[0]
        # The country the distractor city sits in, read off this item's premises.
        target_country = None
        for chunk in c.prompt.split(" . "):
            parts = chunk.split()
            if len(parts) >= 4 and parts[0] == target_city and parts[1] == "sits":
                target_country = parts[3]
        if target_country is None:
            continue
        got = answer_after_swap(
            model, lens, c.prompt, c.latent, target_city, band
        )
        results["n"] += 1
        if got == target_country:
            results["follow"] += 1
        elif got == c.answer:
            results["unchanged"] += 1
        else:
            results["other"] += 1

        # Control: swap two tokens irrelevant to this item, at the same layers
        # and the same magnitude. Catches "any large perturbation flips it".
        filler = rng.choice(["morning", "river", "market", "letter"])
        filler2 = rng.choice(["walked", "waited", "smiled", "paused"])
        got_c = answer_after_swap(model, lens, c.prompt, filler, filler2, band)
        control["n"] += 1
        if got_c == target_country:
            control["follow"] += 1
        elif got_c == c.answer:
            control["unchanged"] += 1
        else:
            control["other"] += 1

    for name, r in (("latent -> distractor swap", results), ("irrelevant-token swap", control)):
        if r["n"]:
            print(f"  {name:26s} n={r['n']:3d}  "
                  f"answer follows the swap {r['follow'] / r['n']:5.1%}  "
                  f"unchanged {r['unchanged'] / r['n']:5.1%}  "
                  f"other {r['other'] / r['n']:5.1%}")

    summary = {
        "n_items": len(correct),
        "layers": layers,
        "median_rank_latent_jlens": np.median(latent_j, axis=0).tolist(),
        "median_rank_latent_logitlens": np.median(latent_logit, axis=0).tolist(),
        "median_rank_distractor_jlens": np.median(distractor_j, axis=0).tolist(),
        "median_rank_answer_jlens": np.median(answer_j, axis=0).tolist(),
        "latent_top1_anywhere": top1,
        "latent_top3_anywhere": top3,
        "best_layer": best_layer,
        "swap": results,
        "swap_control": control,
        "swap_band": band,
    }
    with open(os.path.join(args.out, "multihop.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}/multihop.json")


if __name__ == "__main__":
    main()
