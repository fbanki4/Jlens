"""Fit a Jacobian lens on a toy checkpoint.

The corpus matters more than its size. What ``J_l`` is supposed to capture is
the *average* map from a layer-``l`` direction to the output, marginalised over
contexts — so the corpus should look like the model's general input
distribution, not like the task you are about to study. Fitting on nothing but
two-hop chains would bake the chain template into the transport and make the
readouts look better than they are.
"""

from __future__ import annotations

import argparse
import logging
import os
import random

import torch

from jlens.fitting import fit
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import build_corpus, build_world, make_split


def fitting_corpus(n: int, seed: int = 7, min_tokens: int = 5) -> list[str]:
    """Generic sentences from the world, in the corpus's natural proportions.

    One sentence per prompt, deliberately: the toy model is trained on
    one-sentence rows, so concatenating them would put the lens on activations
    the model never sees. The whole point of ``J_l`` is to be the average
    transport *on the model's own input distribution*, and a corpus that
    drifts off it quietly poisons the estimate.
    """
    world, split = build_world(), make_split(build_world())
    sentences = [
        text for text, _ in build_corpus(world, split, n_sentences=n * 4, seed=seed)
    ]
    rng = random.Random(seed)
    rng.shuffle(sentences)
    return [s for s in sentences if len(s.split()) >= min_tokens][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/toy/final.pt")
    ap.add_argument("--out", default="runs/lens/final_lens.pt")
    ap.add_argument("--n-prompts", type=int, default=150)
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=16)
    ap.add_argument("--skip-first", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    torch.set_num_threads(4)
    net, tok = load_toy(args.model)
    model = ToyLensModel(net, tok)
    prompts = fitting_corpus(args.n_prompts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lens = fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=args.out + ".ckpt",
        checkpoint_every=50,
        log_every=50,
    )
    lens.save(args.out)
    print(f"{lens}  ->  {args.out}")


if __name__ == "__main__":
    main()
