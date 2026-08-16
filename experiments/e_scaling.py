"""Where does the workspace *function* emerge? A scaling harness.

The point of this file is the protocol, not the toy numbers it produces here.
Four design rules, each of which exists because getting it wrong produces a
confident, wrong answer:

**1. Measure function, never geometry.** The CKA block structure is present in
a randomly initialised transformer (`e_architecture.py`), so a "scaling law"
fitted to blockiness is largely fitting architecture. Only the functional
criteria distinguish a workspace from a residual stream.

**2. Split the criteria — they dissociate.** The paper's workspace is a
conjunction: the intermediate is *present*, it is *specific* (beats matched
distractors), and it is *causal* (swapping it moves the answer). At 12 layers
this repo finds presence and specificity at full strength while causality sits
near 10%. A single scalar "workspace score" would average that dissociation
away, and the dissociation is the finding.

**3. Condition on capability.** A model that cannot do the task has no
intermediate to hold. Measure the workspace only on items the model gets
right, and report capability separately — otherwise the "emergence of the
workspace" you plot is the emergence of the *task*, which is a different and
much less interesting claim.

**4. Use continuous metrics.** Thresholded scores (top-1 accuracy, "is it in
the top 3") manufacture sharp transitions out of smooth underlying curves.
Report median rank and specificity ratios, which are continuous, and check any
apparent phase change survives on those.

Run: ``python experiments/e_scaling.py --depths 2,4,8,16``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from experiments.e_architecture import function, geometry
from experiments.fit_toy_lens import fitting_corpus
from jlens.fitting import fit
from jlens.lens import JacobianLens
from jlens.toy.model import ToyConfig, ToyLensModel, load_toy
from jlens.toy.train import TrainConfig, train
from jlens.toy.world import build_world, make_split


def run_one(depth: int, width: int, steps: int, out_dir: str, n_prompts: int) -> dict:
    world, split = build_world(), make_split(build_world())
    tag = f"d{depth}_w{width}"
    run_dir = os.path.join(out_dir, tag)
    model_path = os.path.join(run_dir, "final.pt")

    t0 = time.perf_counter()
    if not os.path.exists(model_path):
        print(f"\n=== training {tag} ({depth} layers, width {width}) ===", flush=True)
        train(
            world, split,
            model_cfg=ToyConfig(d_model=width, n_layers=depth, n_heads=max(2, width // 24)),
            train_cfg=TrainConfig(steps=steps, n_checkpoints=3, seq_len=16,
                                  batch_size=128, log_every=0),
            out_dir=run_dir,
        )
    net, tok = load_toy(model_path)
    model = ToyLensModel(net, tok)

    lens_path = os.path.join(run_dir, "lens.pt")
    if os.path.exists(lens_path):
        lens = JacobianLens.load(lens_path)
    else:
        lens = fit(model, fitting_corpus(n_prompts), dim_batch=width,
                   max_seq_len=16, skip_first=1, checkpoint_path=None, log_every=0)
        lens.save(lens_path)

    row = {
        "depth": depth,
        "width": width,
        "params": sum(p.numel() for p in net.parameters()),
        "seconds": round(time.perf_counter() - t0, 1),
        "geometry": geometry(model, lens),
        "function": function(model, lens, world, split),
    }
    print(f"  {tag}: acc={row['function']['heldout_accuracy']:.2f} "
          f"spec={row['function']['specificity']:.1f}x "
          f"swap={row['function']['swap_follow_rate']:.3f}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", default="2,4,8,16")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-prompts", type=int, default=150)
    ap.add_argument("--out", default="runs/scaling")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    depths = [int(d) for d in args.depths.split(",")]

    rows = []
    for d in depths:
        rows.append(run_one(d, args.width, args.steps, args.out, args.n_prompts))
        with open(os.path.join(args.out, "scaling.json"), "w") as f:
            json.dump(rows, f, indent=2)

    print()
    print("=" * 92)
    print("WORKSPACE CRITERIA vs DEPTH  (width fixed; capability reported separately)")
    print("=" * 92)
    print(f"  {'depth':>5s} {'params':>9s} | {'2-hop acc':>9s} | "
          f"{'latent rk':>9s} {'distr rk':>9s} {'specificity':>11s} "
          f"{'beats dist':>10s} | {'swap':>6s} | {'blockiness':>10s}")
    for r in rows:
        f_, g_ = r["function"], r["geometry"]
        measurable = f_["heldout_accuracy"] > 0.2
        swap = f"{f_['swap_follow_rate']:.3f}" if f_["swap_n"] else "  n/a"
        print(f"  {r['depth']:5d} {r['params']:9,d} | {f_['heldout_accuracy']:9.2f} | "
              f"{f_['median_latent_rank']:9.0f} {f_['median_distractor_rank']:9.0f} "
              f"{f_['specificity']:10.1f}x {f_['latent_beats_distractors_frac']:10.2f} | "
              f"{swap:>6s} | {g_['blockiness']:10.3f}"
              + ("" if measurable else "   <- task not solved; workspace unmeasurable"))

    solved = [r for r in rows if r["function"]["heldout_accuracy"] > 0.2]
    print()
    if len(solved) >= 2:
        print("  Among models that solve the task:")
        print(f"    specificity  {solved[0]['function']['specificity']:.1f}x "
              f"(d={solved[0]['depth']})  ->  "
              f"{solved[-1]['function']['specificity']:.1f}x (d={solved[-1]['depth']})")
        print(f"    swap rate    {solved[0]['function']['swap_follow_rate']:.3f} "
              f"(d={solved[0]['depth']})  ->  "
              f"{solved[-1]['function']['swap_follow_rate']:.3f} (d={solved[-1]['depth']})")
    print(f"\nwrote {args.out}/scaling.json")


if __name__ == "__main__":
    main()
