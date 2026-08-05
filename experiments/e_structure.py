"""Is the J-space really just a bag of concepts?

The paper's own caveat: the lens reads the workspace as a flat set of
independently active concept vectors. A readout containing *spider*, *legs*
and *eight* says those concepts are present, not how they are bound together —
and if the model imposes extra structure (roles, relations, bindings), a
bag-of-features readout cannot see it.

The two-hop task makes this testable, because it has a binding in it. At the
query position the model must know that *this* city is the subject's, and the
other cities in context are not. So:

**The paired design.** For each item A, `swapped_chain` produces item B with
the *same token inventory* — every word identical — but the subject's `lives in`
fact repointed at a different city. Whatever changes between the two query-
position activations is the binding and nothing else:

    d = h_A - h_B

**The question.** If the workspace is a pure bag of concepts, `d` should be
essentially `v_c1 - v_c2`: the two cities' J-lens vectors, swapped, nothing
more. Anything left over is structure the bag cannot represent. We measure

    1. how much of `d` the J-lens dictionary explains at all (vs random control)
    2. how much of `d` the two cities' own vectors explain (the pure-bag model)
    3. what the leftover is made of — and whether it is *causally* live, by
       checking whether it alone can move the model's answer

and separately replicate the capacity measurements: occupancy and
fraction-of-variance-explained by layer.
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
from jlens.jspace import (
    fraction_variance_explained,
    gradient_pursuit,
    occupancy,
    random_control_dictionary,
)
from jlens.lens import JacobianLens
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import build_world, chain_benchmark, swapped_chain


@torch.no_grad()
def query_acts(model: ToyLensModel, prompt: str, layers: list[int]):
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
        return {l: rec.activations[l][0, -1].detach().float() for l in layers}


@torch.no_grad()
def answer_with_added(
    model: ToyLensModel, prompt: str, delta: dict[int, torch.Tensor], alpha: float
) -> str:
    """Run the model with ``alpha * delta[l]`` added at the query position."""
    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            patched = t.clone()
            patched[0, -1] = t[0, -1] + alpha * delta[layer].to(t.device)
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        return hook

    try:
        for l in delta:
            handles.append(model.layers[l].register_forward_hook(make_hook(l)))
        logits = model.unembed(model.forward(model.encode(prompt)))
        return model.tokenizer.decode([int(logits[0, -1].argmax())])
    finally:
        for h in handles:
            h.remove()


def explained(x: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of ``x``'s energy inside the span of ``basis`` rows."""
    Q, _ = torch.linalg.qr(basis.T.float())
    return float((x.float() @ Q).pow(2).sum() / x.float().pow(2).sum().clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/toy/final.pt")
    ap.add_argument("--lens", default="runs/lens/final_lens.pt")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--out", default="runs/structure")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    net, tok = load_toy(args.model)
    model = ToyLensModel(net, tok)
    lens = JacobianLens.load(args.lens)
    world = build_world()
    n_layers = model.n_layers
    layers = lens.source_layers
    band = [l for l in layers if 0.3 <= l / (n_layers - 1) <= 0.75]

    P_full, _ = build_probe(model, token_ids=torch.arange(len(tok)))
    V = {l: (P_full @ lens.jacobians[l]).float() for l in layers}

    # ---- build A/B pairs the model gets right on BOTH sides ---------------
    rng = random.Random(11)
    pairs = []
    for c in chain_benchmark(world, n=args.n * 6, k=3):
        s = swapped_chain(rng, c, world)
        if s is None or s.answer == c.answer:
            continue
        ok = True
        for item in (c, s):
            with torch.no_grad():
                pred = int(
                    model.unembed(model.forward(model.encode(item.prompt)))[0, -1].argmax()
                )
            ok &= pred == tok.token_id(item.answer)
        if ok:
            pairs.append((c, s))
        if len(pairs) >= args.n:
            break
    print(f"{len(pairs)} minimal pairs where the model answers both sides correctly")
    if len(pairs) < 8:
        raise SystemExit("not enough usable pairs; train the model longer")

    # ---- 1-3: what is in the binding difference? --------------------------
    stats = {l: {"jspace": [], "pairvec": [], "random": [], "cos": []} for l in band}
    residual_delta: list[dict[int, torch.Tensor]] = []
    full_delta: list[dict[int, torch.Tensor]] = []
    for a, b in pairs:
        ha, hb = query_acts(model, a.prompt, band), query_acts(model, b.prompt, band)
        res_d, full_d = {}, {}
        for l in band:
            d = ha[l] - hb[l]
            full_d[l] = d
            # (1) the whole J-lens dictionary, sparse non-negative, k atoms
            dec = gradient_pursuit(d, V[l], args.k)
            fve_j = 1.0 - float(dec.residual.pow(2).sum() / d.pow(2).sum().clamp_min(1e-12))
            ctrl = gradient_pursuit(d, random_control_dictionary(V[l]), args.k)
            fve_r = 1.0 - float(ctrl.residual.pow(2).sum() / d.pow(2).sum().clamp_min(1e-12))
            # (2) the pure-bag model: just the two cities' own vectors
            ids_a = tok.encode(a.latent)
            ids_b = tok.encode(b.latent)
            basis = torch.stack(
                [V[l][ids_a].sum(0), V[l][ids_b].sum(0)]
            )
            fve_p = explained(d, basis)
            pair_dir = basis[0] - basis[1]
            cos = float(
                (d @ pair_dir)
                / (d.norm() * pair_dir.norm()).clamp_min(1e-12)
            )
            stats[l]["jspace"].append(fve_j)
            stats[l]["random"].append(fve_r)
            stats[l]["pairvec"].append(fve_p)
            stats[l]["cos"].append(cos)
            # leftover after removing the pure-bag part
            Q, _ = torch.linalg.qr(basis.T.float())
            res_d[l] = d - Q @ (Q.T @ d)
        residual_delta.append(res_d)
        full_delta.append(full_d)

    print()
    print("=" * 78)
    print("WHAT IS IN THE BINDING DIFFERENCE  d = h(city A bound) - h(city B bound)")
    print("=" * 78)
    print(f"  {'layer':>5s} {'rho':>5s} | {'k=' + str(args.k) + ' J-lens':>12s} "
          f"{'random ctrl':>12s} | {'2 city vecs':>12s} {'cos(d, v_A-v_B)':>16s}")
    for l in band:
        s = stats[l]
        print(f"  {l:5d} {l / (n_layers - 1):5.2f} | "
              f"{np.mean(s['jspace']):12.3f} {np.mean(s['random']):12.3f} | "
              f"{np.mean(s['pairvec']):12.3f} {np.mean(s['cos']):16.3f}")

    mean_pair = float(np.mean([np.mean(stats[l]["pairvec"]) for l in band]))
    mean_j = float(np.mean([np.mean(stats[l]["jspace"]) for l in band]))
    mean_r = float(np.mean([np.mean(stats[l]["random"]) for l in band]))
    print()
    print(f"  averaged over the band: the two cities' own vectors explain "
          f"{mean_pair:.1%} of the binding difference;")
    print(f"  a {args.k}-atom J-lens decomposition explains {mean_j:.1%} "
          f"(random control {mean_r:.1%}).")
    print(f"  => {1 - mean_pair:.0%} of the difference is NOT the two swapped "
          f"concept vectors.")

    # ---- is the leftover causally live? ----------------------------------
    print()
    print("=" * 78)
    print("IS THE LEFTOVER CAUSALLY LIVE?")
    print("=" * 78)
    print("  Add the difference vector to item B's query activation and see whether")
    print("  B's answer moves to A's. Full d is the ceiling; the leftover is d with")
    print("  the two cities' own directions projected out — a pure bag-of-concepts")
    print("  account predicts the leftover does nothing.")
    for name, deltas in (("full d", full_delta), ("leftover only", residual_delta)):
        for alpha in (1.0, 2.0):
            moved = same = 0
            for (a, b), delta in zip(pairs, deltas, strict=True):
                got = answer_with_added(model, b.prompt, delta, alpha)
                moved += got == a.answer
                same += got == b.answer
            n = len(pairs)
            print(f"  {name:14s} alpha={alpha:<4}  -> A's answer {moved / n:6.1%}   "
                  f"stays B's {same / n:6.1%}   other {1 - moved / n - same / n:6.1%}")

    # ---- capacity ---------------------------------------------------------
    print()
    print("=" * 78)
    print("CAPACITY: occupancy and variance share by layer")
    print("=" * 78)
    acts_by_layer = {l: [] for l in layers}
    for a, _ in pairs[:24]:
        h = query_acts(model, a.prompt, layers)
        for l in layers:
            acts_by_layer[l].append(h[l])
    print(f"  {'layer':>5s} {'rho':>5s} {'occupancy':>10s} {'FVE':>7s} "
          f"{'FVE rand':>9s} {'excess':>7s}")
    capacity = []
    gen = torch.Generator().manual_seed(0)
    for l in layers:
        occ = [
            occupancy(h, V[l], k_max=40, n_controls=1, generator=gen)[0]
            for h in acts_by_layer[l][:10]
        ]
        k = max(1, int(np.median(occ)))
        fve, fve_r = fraction_variance_explained(
            torch.stack(acts_by_layer[l]), V[l], k
        )
        capacity.append(
            {"layer": l, "rho": l / (n_layers - 1), "occupancy": float(np.median(occ)),
             "fve": fve, "fve_random": fve_r}
        )
        print(f"  {l:5d} {l / (n_layers - 1):5.2f} {np.median(occ):10.1f} "
              f"{fve:7.3f} {fve_r:9.3f} {fve - fve_r:7.3f}")

    with open(os.path.join(args.out, "structure.json"), "w") as f:
        json.dump(
            {
                "n_pairs": len(pairs),
                "band": band,
                "binding": {
                    str(l): {k: float(np.mean(v)) for k, v in stats[l].items()}
                    for l in band
                },
                "capacity": capacity,
            },
            f, indent=2,
        )
    print(f"\nwrote {args.out}/structure.json")


if __name__ == "__main__":
    main()
