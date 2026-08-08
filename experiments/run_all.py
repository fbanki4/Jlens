"""Run the whole pipeline end to end on CPU: train -> fit -> read -> intervene.

    python -m experiments.run_all

Stages are skipped if their outputs already exist, so this is safe to re-run.
Roughly half an hour on four CPU cores, dominated by training the toy model
and by fitting one lens per training checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch


def sh(*args: str) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument(
        "--data-root",
        default="/workspace/eliebak/open-jlens-data",
        help="clone of eliebak/open-jlens-data for the shapes analysis",
    )
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    # 1. Train the toy model, checkpointing along the way.
    if not args.skip_training and not os.path.exists("runs/toy/final.pt"):
        from jlens.toy.model import ToyConfig
        from jlens.toy.train import TrainConfig, train
        from jlens.toy.world import build_world, make_split

        world = build_world()
        train(
            world,
            make_split(world),
            model_cfg=ToyConfig(d_model=96, n_layers=12, n_heads=6),
            train_cfg=TrainConfig(steps=args.steps, n_checkpoints=16, seq_len=16,
                                  batch_size=128),
            out_dir="runs/toy",
        )

    # 2. Fit the lens on the final model.
    if not os.path.exists("runs/lens/final_lens.pt"):
        sh("experiments/fit_toy_lens.py", "--n-prompts", str(args.n_prompts),
           "--dim-batch", "96")

    # 3. Read and write: does the lens surface the latent, and does it steer?
    sh("experiments/e_multihop.py")

    # 4. Open problem: untrained multi-token concept vectors.
    sh("experiments/e_phrase.py")

    # 5. Open problem: is it more than a bag of concepts?
    sh("experiments/e_structure.py")

    # 6. Open problem: when does the J-space form?
    sh("experiments/e_form.py")

    # 7. Is the J-space architectural or learned? (untrained vs trained)
    sh("experiments/e_architecture.py")

    # 8. Open problem: what do the shapes indicate? (real open-model data)
    if os.path.isdir(args.data_root):
        sh("experiments/e_shapes.py", "--data-root", args.data_root)
    else:
        print(f"\nskipping shapes analysis: {args.data_root} not found "
              f"(clone github.com/eliebak/open-jlens-data)")

    print("\nall stages complete; results under runs/")


if __name__ == "__main__":
    main()
