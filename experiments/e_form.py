"""When and how does the J-space form?

The paper compares a base model against its post-trained version and finds the
workspace already present in the base, so it is not a product of RLHF. What
happens *before* that is untouched: whether the structure appears gradually or
abruptly, and where in training it locks in.

Open-weights checkpoint series (OLMo, SmolLM3) let you watch the geometry, but
not the capability — those runs do not ship per-checkpoint task evals matched
to the thing the workspace is supposed to support. Here we have both, because
we trained the model: at every checkpoint we can fit a lens *and* measure
whether the model can yet do the two-hop task. That makes the sharper question
askable:

    does the workspace geometry settle before, with, or after the ability
    that is supposed to route through it?

Metrics per checkpoint (the same ones used on the real open-model
trajectories, so the shapes are comparable):

    blockiness      mean within-block CKA minus mean between-block CKA
    layer_coupling  mean off-diagonal CKA (segmentation-free)
    boundaries      relative depth of the block cuts
    lock_in         matched-depth CKA against the FINAL checkpoint
    drift           matched-depth CKA against the PREVIOUS checkpoint
    pr_peak         peak participation ratio of V_l over depth (capacity proxy)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from jlens.cka import (
    build_probe,
    cka_matrix,
    lens_vectors,
    linear_cka,
    matched_depth_pairs,
    relative_depth,
    segment_blocks,
)
from jlens.fitting import fit
from jlens.jspace import participation_ratio
from jlens.lens import JacobianLens
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.train import chain_accuracy
from jlens.toy.world import build_world, chain_benchmark
from experiments.fit_toy_lens import fitting_corpus


def checkpoint_paths(run_dir: str) -> list[tuple[int, str]]:
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "ckpt_step*.pt"))):
        m = re.search(r"step(\d+)", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def fit_or_load(path: str, model: ToyLensModel, prompts, cache: str, **kw) -> JacobianLens:
    if os.path.exists(cache):
        return JacobianLens.load(cache)
    lens = fit(model, prompts, checkpoint_path=None, log_every=0, **kw)
    lens.save(cache)
    return lens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/toy")
    ap.add_argument("--out", default="runs/formation")
    ap.add_argument("--n-prompts", type=int, default=60)
    ap.add_argument("--dim-batch", type=int, default=96)
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "lenses"), exist_ok=True)
    prompts = fitting_corpus(args.n_prompts)
    chains = chain_benchmark(build_world(), n=100)

    ckpts = checkpoint_paths(args.run_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.run_dir}")
    print(f"{len(ckpts)} checkpoints, fitting {args.n_prompts} prompts each\n")

    entries = []
    for step, path in ckpts:
        net, tok = load_toy(path)
        model = ToyLensModel(net, tok)
        cache = os.path.join(args.out, "lenses", f"lens_step{step:06d}.pt")
        lens = fit_or_load(
            path, model, prompts, cache,
            dim_batch=args.dim_batch, max_seq_len=48, skip_first=4,
        )
        P, _ = build_probe(model)
        C = cka_matrix(lens, P)
        seg = segment_blocks(C, lens.source_layers, model.n_layers)
        Vs = {l: lens_vectors(P, lens.jacobians[l]) for l in lens.source_layers}
        prs = [participation_ratio(V) for V in Vs.values()]
        entries.append(
            {
                "step": step,
                "path": path,
                "chain_acc": chain_accuracy(net, tok, chains),
                "n_blocks": seg.n_blocks,
                "blockiness": seg.blockiness,
                "layer_coupling": seg.layer_coupling,
                "boundaries_rho": seg.boundaries_rho,
                "pr_peak": float(np.max(prs)),
                "pr_by_layer": [float(x) for x in prs],
                "C": C,
                "V": Vs,
                "rho": relative_depth(lens, model.n_layers),
                "n_layers": model.n_layers,
            }
        )
        print(f"  step {step:6d}  chain {entries[-1]['chain_acc']:.2f}  "
              f"k={seg.n_blocks}  blockiness {seg.blockiness:.3f}  "
              f"coupling {seg.layer_coupling:.3f}  "
              f"cuts {[round(b, 2) for b in seg.boundaries_rho]}")

    # ---- lock-in and drift ------------------------------------------------
    final = entries[-1]
    for i, e in enumerate(entries):
        pairs = matched_depth_pairs(e["rho"], final["rho"])
        e["lock_in"] = float(
            np.mean([
                linear_cka(e["V"][sorted(e["V"])[a]], final["V"][sorted(final["V"])[b]])
                for a, b in pairs
            ])
        )
        if i == 0:
            e["drift"] = float("nan")
        else:
            prev = entries[i - 1]
            pairs = matched_depth_pairs(e["rho"], prev["rho"])
            e["drift"] = float(
                np.mean([
                    linear_cka(e["V"][sorted(e["V"])[a]], prev["V"][sorted(prev["V"])[b]])
                    for a, b in pairs
                ])
            )

    print()
    print("=" * 78)
    print("FORMATION OF THE J-SPACE")
    print("=" * 78)
    print(f"  {'step':>6s} {'chain':>6s} {'k':>2s} {'blockiness':>10s} "
          f"{'coupling':>8s} {'lock-in':>8s} {'drift':>7s} {'PR peak':>8s} {'cuts':>12s}")
    for e in entries:
        print(f"  {e['step']:6d} {e['chain_acc']:6.2f} {e['n_blocks']:2d} "
              f"{e['blockiness']:10.3f} {e['layer_coupling']:8.3f} "
              f"{e['lock_in']:8.3f} {e['drift']:7.3f} {e['pr_peak']:8.2f} "
              f"{str([round(b, 2) for b in e['boundaries_rho']]):>12s}")

    # ---- where does each thing reach 90% of its final value? --------------
    steps = np.array([e["step"] for e in entries], dtype=float)

    def crossing(values, frac=0.9, rising=True) -> float:
        v = np.array(values, dtype=float)
        finalv = v[-1]
        if not np.isfinite(finalv) or finalv == 0:
            return float("nan")
        hit = np.where(v >= frac * finalv) if rising else np.where(v <= finalv / frac)
        return float(steps[hit[0][0]]) if len(hit[0]) else float("nan")

    acc = [e["chain_acc"] for e in entries]
    print()
    print("  first checkpoint reaching 90% of the final value:")
    print(f"    two-hop capability     step {crossing(acc):>7.0f}")
    print(f"    lock-in (CKA to final) step {crossing([e['lock_in'] for e in entries]):>7.0f}")
    print(f"    blockiness             step {crossing([e['blockiness'] for e in entries]):>7.0f}")
    print(f"    PR peak                step {crossing([e['pr_peak'] for e in entries]):>7.0f}")

    lock = np.array([e["lock_in"] for e in entries])
    accv = np.array(acc)
    if np.std(lock) > 0 and np.std(accv) > 0:
        print(f"\n  corr(lock-in, capability) across checkpoints: "
              f"{np.corrcoef(lock, accv)[0, 1]:+.3f}")

    payload = [
        {k: v for k, v in e.items() if k not in ("C", "V", "rho")}
        | {"rho": e["rho"].tolist(), "C": e["C"].tolist()}
        for e in entries
    ]
    with open(os.path.join(args.out, "formation.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}/formation.json")


if __name__ == "__main__":
    main()
