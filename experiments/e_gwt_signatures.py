"""The two GWT signatures the rest of this repo was missing.

Global workspace theory makes four testable structural claims. Three were
already covered here — dimensionality (`jlens.jspace.participation_ratio`),
causal readout fidelity (the swap experiments), and the layer-band structure
(`jlens.cka`). Two were not:

**Broadcast breadth.** Not "is this direction aligned with downstream weights"
— that is correlational and a direction can be aligned with a component it
never actually moves. The causal version perturbs along the direction and
watches how far the disturbance travels, against a random direction of
identical norm injected at the same place. Ratio 1.0 means "reaches exactly as
far as noise of the same size".

**The bottleneck.** Not "how many concepts can I read at once" — every
measurement has a floor and will look limited. The claim worth testing is that
contents *compete*: hold ``n`` concepts, then ``n+1``, and see whether the
earliest one gets pushed out. A positive eviction slope with positive
selectivity is a bottleneck; a positive slope with zero selectivity is just
the whole readout dimming as the prompt gets longer.

Both are written against the `LensModel` protocol, so the same code runs on
the toy models here and on any HuggingFace decoder.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from jlens.broadcast import broadcast_profile
from jlens.capacity import eviction_curve
from jlens.cka import build_probe, workspace_band
from jlens.lens import JacobianLens
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import build_world
from experiments.fit_toy_lens import fitting_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/toy/final.pt")
    ap.add_argument("--lens", default="runs/lens/final_lens.pt")
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--n-concepts", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--out", default="runs/gwt")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    net, tok = load_toy(args.model)
    model = ToyLensModel(net, tok)
    lens = JacobianLens.load(args.lens)
    world = build_world()
    prompts = fitting_corpus(args.n_prompts)
    band = workspace_band(lens.source_layers, model.n_layers)

    P, _ = build_probe(model, token_ids=torch.arange(len(tok)))
    cities = [c for c in world.entities["city"] if len(tok.encode(c)) == 1]
    concept_ids = [tok.token_id(c) for c in cities[: args.n_concepts]]

    report = {"band": band}

    # ---- 1. broadcast breadth -------------------------------------------
    print("=" * 76)
    print("BROADCAST BREADTH  (1.0 = reaches exactly as far as matched noise)")
    print("=" * 76)
    print(f"  {'layer':>5s} {'rho':>5s} {'J-lens vecs':>12s} {'random ctrl':>12s} "
          f"{'reach@last':>11s}")
    gen = torch.Generator().manual_seed(0)
    rows = []
    for layer in lens.source_layers[:-1]:
        V = (P @ lens.jacobians[layer]).float()
        profiles = [
            broadcast_profile(model, V[t], layer, prompts, n_controls=2,
                              max_seq_len=16, generator=gen)
            for t in concept_ids
        ]
        breadth = float(np.nanmean([p.breadth for p in profiles]))
        reach = float(np.nanmean([p.reach for p in profiles]))
        # Control: random directions scored the same way. By construction this
        # is ~1.0; printing it is the sanity check that the ratio is calibrated.
        ctrl = []
        for _ in range(3):
            r = torch.randn(model.d_model, generator=gen)
            ctrl.append(
                broadcast_profile(model, r, layer, prompts, n_controls=2,
                                  max_seq_len=16, generator=gen).breadth
            )
        ctrl_mean = float(np.nanmean(ctrl))
        rows.append({"layer": layer, "rho": layer / (model.n_layers - 1),
                     "breadth": breadth, "control": ctrl_mean, "reach": reach})
        print(f"  {layer:5d} {layer/(model.n_layers-1):5.2f} {breadth:12.2f} "
              f"{ctrl_mean:12.2f} {reach:11.2f}")
    report["broadcast"] = rows

    if rows:
        in_band = [r for r in rows if r["layer"] in band]
        out_band = [r for r in rows if r["layer"] not in band]
        print()
        if in_band and out_band:
            bi = np.nanmean([r["breadth"] for r in in_band])
            bo = np.nanmean([r["breadth"] for r in out_band])
            print(f"  mean breadth inside the band {bi:.2f} vs outside {bo:.2f}")
            print("  GWT predicts a peak inside the band, not a monotone rise "
                  "toward the output.")
        peak = max(rows, key=lambda r: r["breadth"])
        print(f"  peak breadth {peak['breadth']:.2f} at layer {peak['layer']} "
              f"(rho={peak['rho']:.2f})")

    # ---- 2. the bottleneck ----------------------------------------------
    print()
    print("=" * 76)
    print("BOTTLENECK  (does an N+1th item evict the first?)")
    print("=" * 76)
    pool = [c for c in world.entities["city"] if len(tok.encode(c)) == 1]
    pool += [c for c in world.entities["country"] if len(tok.encode(c)) == 1]
    lengths = [n for n in (1, 2, 3, 4, 6, 8) if n <= len(pool)]
    curve = eviction_curve(
        model, lens, pool, lengths=lengths, n_trials=args.n_trials,
        template="{items} .", separator=" ", band=band, max_seq_len=16,
    )
    print(f"  {'list len':>8s} {'first item':>11s} {'all items':>10s} "
          f"{'never-shown':>12s}")
    for i, n in enumerate(curve.lengths):
        print(f"  {n:8d} {curve.first_item_rank[i]:11.0f} "
              f"{curve.all_item_rank[i]:10.0f} {curve.control_rank[i]:12.0f}")
    print()
    print(f"  eviction slope {curve.eviction_slope:+.3f} "
          f"(log-rank of the first item per added item; >0 means eviction)")
    print(f"  selectivity    {curve.selectivity:+.3f} "
          f"(first item vs average; ~0 means the whole readout just dims)")
    if not np.isfinite(curve.eviction_slope):
        print("  -> not measurable on this model")
    elif curve.eviction_slope > 0.05 and curve.selectivity > 0.02:
        print("  -> BOTTLENECK: earlier items are selectively pushed out")
    elif curve.eviction_slope > 0.05:
        print("  -> decay, but uniform: the readout dims rather than evicting")
    else:
        print("  -> no bottleneck detected at these lengths")
    report["capacity"] = curve.summary()

    print()
    print("  NOTE: the toy model was never trained on list prompts, so its")
    print("  capacity numbers are a smoke test of the code, not a result. On a")
    print("  real model use a natural list template and an unrelated word pool.")

    with open(os.path.join(args.out, "gwt_signatures.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}/gwt_signatures.json")


if __name__ == "__main__":
    main()
