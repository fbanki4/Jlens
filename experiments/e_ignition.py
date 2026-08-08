"""Ignition across depth and across training — the lens-free axis.

Pilots the protocol that :mod:`experiments.pythia_sweep` runs at scale. Two
sweeps, both using :mod:`jlens.ignition` and therefore never touching ``J_l``:

* **depth** — the ladder from ``e_scaling.py`` (2/4/8/16 layers, same task,
  same data). Tests whether the onset sits at a fixed *fraction* of depth, and
  whether an all-or-none switch appears at all as depth grows.
* **training** — the checkpoint trajectory from ``runs/toy``. Tests whether
  ignition sharpens over training, on an axis where a lens-based measure
  cannot be trusted (early ``J_l`` is degenerate, so a lens-based "onset"
  could be the instrument warming up rather than the model changing).

The headline number is **sharpness**: max slope of the mixture-tracking curve,
normalised so proportional tracking is exactly 1.0. With 21 sweep points a
perfect all-or-none switch reads about 20. Anything near 1 means the model is
averaging the two interpretations rather than committing to one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import numpy as np
import torch

from jlens.ignition import ignition_sweep
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import build_world, composed_prompt


def build_trials(tok, world, n: int = 16, seed: int = 0) -> list[tuple[str, str]]:
    """Pairs of subjects with *different* answers and single-token names.

    Blending two subjects who share an answer is not an ambiguous input — the
    model can commit without choosing.
    """
    composed = world.composed_map("hails_from")
    rng = random.Random(seed)
    people = [p for p in composed if len(tok.encode(p)) == 1]
    trials, guard = [], 0
    while len(trials) < n and guard < n * 200:
        guard += 1
        a, b = rng.sample(people, 2)
        if composed[a] != composed[b]:
            trials.append(
                (composed_prompt("hails_from", a), composed_prompt("hails_from", b))
            )
    return trials


def measure(path: str, trials_fn, n_alphas: int, measure_position: int) -> dict:
    net, tok = load_toy(path)
    model = ToyLensModel(net, tok)
    trials = trials_fn(tok)
    r = ignition_sweep(
        model, trials, concept_position=0,
        measure_position=measure_position, n_alphas=n_alphas,
    )
    out = r.summary()
    out["n_layers"] = model.n_layers
    out["params"] = sum(p.numel() for p in net.parameters())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder-dir", default="runs/scaling")
    ap.add_argument("--traj-dir", default="runs/toy")
    ap.add_argument("--out", default="runs/ignition")
    ap.add_argument("--n-alphas", type=int, default=21)
    ap.add_argument("--n-trials", type=int, default=16)
    ap.add_argument("--measure-position", type=int, default=-1)
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    world = build_world()
    trials_fn = lambda tok: build_trials(tok, world, n=args.n_trials)  # noqa: E731
    step_ref = f"a perfect all-or-none switch reads ~{args.n_alphas - 1}"

    report = {"n_alphas": args.n_alphas, "depth": [], "training": []}

    # ---- depth ----------------------------------------------------------
    rungs = sorted(
        glob.glob(os.path.join(args.ladder_dir, "d*_w*", "final.pt")),
        key=lambda p: int(re.search(r"d(\d+)_", p).group(1)),
    )
    if rungs:
        print("=" * 78)
        print(f"IGNITION vs DEPTH   (proportional tracking = 1.0; {step_ref})")
        print("=" * 78)
        print(f"  {'depth':>5s} {'params':>9s} {'max sharp':>10s} "
              f"{'bimodality':>11s} {'onset rho':>10s}")
        for p in rungs:
            row = measure(p, trials_fn, args.n_alphas, args.measure_position)
            row["path"] = p
            report["depth"].append(row)
            print(f"  {row['n_layers']:5d} {row['params']:9,d} "
                  f"{row['max_sharpness']:10.2f} "
                  f"{np.nanmax(row['bimodality']):11.2f} {row['onset_rho']:10.2f}")

    # ---- training -------------------------------------------------------
    ckpts = sorted(
        glob.glob(os.path.join(args.traj_dir, "ckpt_step*.pt")),
        key=lambda p: int(re.search(r"step(\d+)", p).group(1)),
    )
    if ckpts:
        print()
        print("=" * 78)
        print("IGNITION vs TRAINING STEP   (lens-free, so early steps are readable)")
        print("=" * 78)
        print(f"  {'step':>6s} {'max sharp':>10s} {'bimodality':>11s} {'onset rho':>10s}")
        for p in ckpts:
            step = int(re.search(r"step(\d+)", p).group(1))
            row = measure(p, trials_fn, args.n_alphas, args.measure_position)
            row["step"] = step
            report["training"].append(row)
            print(f"  {step:6d} {row['max_sharpness']:10.2f} "
                  f"{np.nanmax(row['bimodality']):11.2f} {row['onset_rho']:10.2f}")

    # ---- verdict --------------------------------------------------------
    print()
    print("=" * 78)
    allrows = report["depth"] + report["training"]
    if allrows:
        peak = max(r["max_sharpness"] for r in allrows)
        print(f"  peak sharpness anywhere in the sweep: {peak:.2f}")
        if peak < 3.0:
            print(f"  -> NO IGNITION. {step_ref}; this is barely above the 1.0 of")
            print("     proportional tracking, so the model averages the two")
            print("     interpretations at every depth and every checkpoint rather")
            print("     than committing to one. Lens-free, so not an instrument")
            print("     artifact.")
        else:
            print("  -> a switch is present; check whether onset_rho is stable")
            print("     across depth (structural) or moves (compute-driven).")
        onsets = [r["onset_rho"] for r in report["depth"] if np.isfinite(r["onset_rho"])]
        if len(onsets) > 1:
            print(f"  onset_rho across the depth ladder: "
                  f"{min(onsets):.2f}-{max(onsets):.2f} (spread {max(onsets)-min(onsets):.2f})")

    with open(os.path.join(args.out, "ignition.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}/ignition.json")


if __name__ == "__main__":
    main()
