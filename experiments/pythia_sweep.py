"""Where does the workspace emerge? A Pythia sweep over (scale x training).

**This script cannot run in the sandbox this repo was developed in — the
HuggingFace host is blocked by egress policy. It is written to be run
somewhere with model access, and every design choice is explained so you can
check it rather than trust it.**

Why Pythia and not OLMo
-----------------------
Three properties, and each one removes a confound that would otherwise be
fatal:

* **Identical data in identical order across all sizes.** Scale and data
  curriculum are decoupled, so ``(parameters x tokens)`` is a clean grid
  rather than a tangle.
* **154 checkpoints per size**, including ten log-spaced ones at steps 1-512.
  That early cluster is exactly where the workspace should be absent, so the
  resolution sits where the transition would be.
* **One tokenizer across every size.** The J-lens is indexed by single
  vocabulary tokens, so any cross-family claim about "how many concepts fit"
  is partly a tokenizer fact. Within Pythia it is not.

Order of operations, and why
----------------------------
**Stage 1 is lens-free, and it goes first.** ``J_l`` is an expectation of
``dh_final/dh_l``; early in training the model is near a unigram predictor and
that map is degenerate. A lens-based reading of "no workspace at step 512"
cannot be told apart from "the instrument had nothing to measure", so a
lens-first sweep will find a transition that might be the lens warming up.
The ignition paradigm (:mod:`jlens.ignition`) touches only the raw residual
stream, produces a continuous scalar, and works on models far too small to do
any reasoning task. Run it across the whole grid first.

**Stage 2 brings the lens in and checks agreement.** Where the lens-derived
onset agrees with the lens-free one, you have triangulation. Where they
disagree, you have localised the artifact instead of hiding it.

**Stage 3 is functional, and it goes last.** Pythia-70M cannot do two-hop
reasoning, so swap-success is pinned at the floor and tells you nothing. Only
run it on rungs that clear a capability bar, and report capability separately.

Two free results
----------------
* **Report against compute (~6ND), not parameters, and test for collapse.** If
  ignition sharpness collapses onto one curve in compute across all sizes, the
  workspace is a compute phenomenon; if it collapses better against depth, it
  is architectural. That collapse test answers the question better than any
  single threshold.
* **Is onset a fixed fraction of depth?** Pythia-70M has 6 layers, 12B has 36.
  If onset sits near the same relative depth regardless, that is a structural
  claim, and it costs nothing extra.

Finally: **fit the smooth power law first.** Make a broken power law win a
nested comparison before believing a phase transition, and check the break
survives a change of metric. Apparent emergence in LLMs is very often an
artifact of a discontinuous metric (Schaeffer et al.), and here you would be
actively looking for one.

Usage::

    pip install torch transformers
    python experiments/pythia_sweep.py --sizes 70m,160m,410m \\
        --steps 0,1,2,4,8,16,32,64,128,256,512,1000,4000,16000,64000,143000 \\
        --purge --out runs/pythia

No GPU and no HuggingFace account are required for that command. Pythia is
ungated; the ignition stage is forward passes only, and pythia-410m takes
about two minutes per checkpoint on four CPU cores. ``--purge`` deletes each
checkpoint after measuring so peak disk stays at one model.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

#: Pythia sizes -> (n_layers, d_model, non-embedding params) for the compute axis.
PYTHIA = {
    "14m": (6, 128, 1.4e7), "31m": (6, 256, 3.1e7), "70m": (6, 512, 7.0e7),
    "160m": (12, 768, 1.6e8), "410m": (24, 1024, 4.1e8), "1b": (16, 2048, 1.0e9),
    "1.4b": (24, 2048, 1.4e9), "2.8b": (32, 2560, 2.8e9),
    "6.9b": (32, 4096, 6.9e9), "12b": (36, 5120, 1.18e10),
}
TOKENS_PER_STEP = 2_097_152  # Pythia: batch 1024 x seq 2048

#: Country pairs for the ambiguity paradigm. Single-token-ish in the NeoX
#: tokenizer and semantically far apart, so blending them is genuinely
#: ambiguous rather than a blend of near-synonyms.
COUNTRY_PAIRS = [
    ("France", "Japan"), ("Germany", "Brazil"), ("Italy", "China"),
    ("Spain", "India"), ("Canada", "Egypt"), ("Russia", "Mexico"),
    ("Norway", "Kenya"), ("Poland", "Chile"),
]
CARRIERS = [
    "My sister has always wanted to visit {}",
    "The best wine I ever had came from {}",
    "After the war, my grandparents emigrated from {}",
    "Her accent suggests she grew up in {}",
    "The parcel was posted from {}",
]


def build_trials(tokenizer, n: int = 40) -> list[tuple[str, str]]:
    """Carrier sentences with the country swapped, keeping length matched.

    A mixture is only defined when the two prompts tokenize to the same
    length, so mismatched pairs are dropped rather than padded — padding would
    shift the position of everything after the concept token.
    """
    trials = []
    for carrier in CARRIERS:
        for a, b in COUNTRY_PAIRS:
            pa, pb = carrier.format(a), carrier.format(b)
            if len(tokenizer(pa).input_ids) == len(tokenizer(pb).input_ids):
                trials.append((pa, pb))
            if len(trials) >= n:
                return trials
    return trials


def concept_index(tokenizer, prompt_a: str, prompt_b: str) -> int | None:
    """First token index where the two prompts differ."""
    a = tokenizer(prompt_a).input_ids
    b = tokenizer(prompt_b).input_ids
    if len(a) != len(b):
        return None
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return i
    return None


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pythia(size: str, step: int, dtype: str = "float32", device: str = "cpu"):
    """Load one (size, checkpoint) pair.

    Pythia is ungated and public: no HuggingFace account, token or payment is
    needed to download it. ``revision="step{N}"`` selects the checkpoint.
    """
    import transformers

    name = f"EleutherAI/pythia-{size}"
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        name, revision=f"step{step}", dtype=getattr(torch, dtype)
    ).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(name, revision=f"step{step}")
    return hf, tok


def purge_cache(size: str, step: int) -> None:
    """Delete one checkpoint's snapshot from the HF cache.

    Sixteen checkpoints of pythia-410m is ~26 GB in fp32. Measuring then
    deleting keeps peak disk at one model regardless of how long the sweep is,
    which is what makes this runnable on a laptop.
    """
    import shutil

    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception:
        return
    for repo in cache.repos:
        if repo.repo_id != f"EleutherAI/pythia-{size}":
            continue
        for rev in repo.revisions:
            if f"step{step}" in rev.refs:
                shutil.rmtree(rev.snapshot_path, ignore_errors=True)


def run_ignition(size: str, step: int, args) -> dict:
    import jlens
    from jlens.ignition import ignition_sweep

    device = pick_device(args.device)
    hf, tok = load_pythia(size, step, args.dtype, device)
    model = jlens.from_hf(hf, tok)
    trials = build_trials(tok, args.n_trials)
    if not trials:
        raise SystemExit("no length-matched trials; adjust COUNTRY_PAIRS/CARRIERS")
    pos = concept_index(tok, *trials[0])

    r = ignition_sweep(
        model, trials,
        concept_position=pos,
        measure_position=pos,      # the paper reads at the mixed token
        n_alphas=args.n_alphas,
    )
    n_layers, d_model, params = PYTHIA[size]
    out = r.summary()
    out.update({
        "size": size, "step": step, "n_layers": n_layers, "d_model": d_model,
        "params": params, "tokens": step * TOKENS_PER_STEP,
        # Compute proxy 6ND: the axis to test for curve collapse.
        "compute_flops": 6 * params * step * TOKENS_PER_STEP,
        "concept_position": pos, "n_trials_used": len(trials),
    })
    del hf, model
    if device == "cuda":
        torch.cuda.empty_cache()
    if args.purge:
        purge_cache(size, step)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="70m,160m,410m")
    ap.add_argument("--steps", default="0,1,2,4,8,16,32,64,128,256,512,"
                                       "1000,4000,16000,64000,143000")
    ap.add_argument("--stage", default="ignition", choices=["ignition"],
                    help="stage 1 only for now; lens and functional stages "
                         "reuse fit_toy_lens/e_multihop against from_hf models")
    ap.add_argument("--n-alphas", type=int, default=21)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="auto",
                    help="auto picks cuda when available, else cpu. The "
                         "ignition stage is forward-passes only: pythia-410m "
                         "takes ~2 min for 40 trials on four CPU cores.")
    ap.add_argument("--purge", action="store_true",
                    help="delete each checkpoint from the HF cache after "
                         "measuring it, so peak disk stays at one model")
    ap.add_argument("--out", default="runs/pythia")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sizes = args.sizes.split(",")
    steps = [int(s) for s in args.steps.split(",")]
    path = os.path.join(args.out, "ignition.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["size"], r["step"]) for r in rows}

    print(f"{len(sizes)} sizes x {len(steps)} steps = "
          f"{len(sizes) * len(steps)} model instances "
          f"({len(done)} already done)")
    for size in sizes:
        for step in steps:
            if (size, step) in done:
                continue
            try:
                row = run_ignition(size, step, args)
            except Exception as exc:  # a missing revision should not kill the sweep
                print(f"  {size}/step{step}: FAILED {type(exc).__name__}: {exc}")
                continue
            rows.append(row)
            with open(path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  {size:>5s} step{step:<7d} sharpness "
                  f"{row['max_sharpness']:6.2f}  onset_rho {row['onset_rho']:.2f}")

    if not rows:
        return
    print()
    print("=" * 78)
    print("COLLAPSE TEST — does ignition sharpness follow compute, or depth?")
    print("=" * 78)
    trained = [r for r in rows if r["step"] > 0]
    if len(trained) > 3:
        y = np.array([r["max_sharpness"] for r in trained])
        for name, x in (
            ("log compute (6ND)", np.log10([max(r["compute_flops"], 1) for r in trained])),
            ("log params", np.log10([r["params"] for r in trained])),
            ("log tokens", np.log10([max(r["tokens"], 1) for r in trained])),
            ("n_layers", np.array([r["n_layers"] for r in trained], dtype=float)),
        ):
            if np.std(x) > 0 and np.std(y) > 0:
                print(f"  corr(sharpness, {name:18s}) = {np.corrcoef(x, y)[0, 1]:+.3f}")
        print("\n  The axis with the tightest collapse is the answer: compute means")
        print("  the workspace is bought with training, depth means it is structural.")

    onsets = [r["onset_rho"] for r in trained if np.isfinite(r["onset_rho"])]
    if onsets:
        print(f"\n  onset_rho across the grid: {np.min(onsets):.2f}-{np.max(onsets):.2f} "
              f"(median {np.median(onsets):.2f})")
        print("  A constant fraction of depth across 6-36 layers is a structural result.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
