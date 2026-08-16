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

The instrument confound, and what stage 2 does about it
-------------------------------------------------------
The J-lens is *fitted per model*, so lens capacity varies across the sweep and
a bigger apparent workspace in a bigger model might just be a bigger lens.
Stage 2 is built around that, via :mod:`jlens.lenscontrol`:

* One :class:`~jlens.lenscontrol.FittingBudget` for every rung — the same
  prompts, sequence length and probe tokens, so *samples per Jacobian entry* is
  constant rather than merely "same number of prompts". Its fingerprint is
  written into every row, so a re-run under different settings cannot be
  appended to the curve unnoticed.
* A fixed number of probe layers at matched relative depths, so a 6-layer and a
  24-layer model produce CKA matrices of the same size. Otherwise blockiness
  partly measures depth.
* Every geometry statistic reported against a matched null of identical shape
  and scale, and again after projecting all models to one reference width.
* An explicit instrument-calibration curve: J-lens fidelity against the logit
  lens at every rung. Rungs where the fitted transport bought nothing carry a
  warning in their own output row, and the whole sweep ends by regressing lens
  fidelity out of the headline relationship
  (:func:`jlens.scaling.partial_correlation`). If "workspace grows with
  compute" does not survive that, it was the instrument.

Stage 2 costs far more than stage 1: fitting ``J_l`` needs ``d_model``
backward passes per prompt, so budget roughly ``d_model / dim_batch`` graph
replays per prompt per checkpoint. This is the stage to put on a GPU.

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

    # stage 1 -- lens-free, CPU is fine
    python experiments/pythia_sweep.py --stage ignition \\
        --sizes 70m,160m,410m \\
        --steps 0,1,2,4,8,16,32,64,128,256,512,1000,4000,16000,64000,143000 \\
        --purge --out runs/pythia

    # stage 2 -- lens-based, wants a GPU
    python experiments/pythia_sweep.py --stage lens \\
        --sizes 70m,160m,410m --steps 0,16,512,4000,64000,143000 \\
        --purge --out runs/pythia

    # read both back, with the instrument regressed out
    python experiments/pythia_sweep.py --stage report --out runs/pythia

No GPU and no HuggingFace account are required for stage 1. Pythia is ungated;
the ignition stage is forward passes only, and pythia-410m takes about two
minutes per checkpoint on four CPU cores. ``--purge`` deletes each checkpoint
after measuring so peak disk stays at one model.
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

#: Fitting corpus for stage 2, hardcoded rather than downloaded.
#: The lens must be fitted on *identical text* at every rung — a corpus that
#: differs between models makes lens quality differ between models, which is
#: precisely the confound stage 2 exists to remove. Hardcoding also means the
#: sweep needs no dataset access and reproduces byte-for-byte.
SENTENCES = [
    "The committee met on Tuesday to review the proposal in detail.",
    "Water boils at a lower temperature as altitude increases.",
    "She put the letter down and stared out of the kitchen window.",
    "Trade between the two ports declined sharply after the treaty.",
    "The compiler reported an error on the third line of the file.",
    "Most of the birds had already left before the first frost.",
    "He argued that the evidence supported a rather different reading.",
    "Rain fell steadily through the afternoon and into the evening.",
    "The bridge was built in the last century and rebuilt twice since.",
    "Prices rose again in the spring, though not by very much.",
    "A small crowd had gathered near the entrance to the station.",
    "The results were consistent across every group they tested.",
    "Nobody remembered who had suggested the idea in the first place.",
    "Copper conducts electricity better than iron but costs more.",
    "The novel was translated into eleven languages within a decade.",
    "Their argument turned on a single word in the second paragraph.",
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


def fitting_corpus(n_prompts: int, sentences_per_prompt: int = 6) -> list[str]:
    """Deterministic paragraphs built from :data:`SENTENCES`.

    Sentences are cycled with a stride coprime to the pool size, so every
    prompt is a different mix rather than the same block repeated, and the
    whole corpus is a pure function of ``n_prompts``. Paragraphs are long
    enough that the first ``skip_first`` positions can be dropped and plenty of
    valid source positions remain.
    """
    out = []
    for i in range(n_prompts):
        idx = [(i * 5 + j) % len(SENTENCES) for j in range(sentences_per_prompt)]
        out.append(" ".join(SENTENCES[k] for k in idx))
    return out


def run_lens(size: str, step: int, args, budget) -> dict:
    """Stage 2: fit a lens under the fixed budget and measure it.

    The lens is fitted at :func:`~jlens.lenscontrol.matched_depth_layers`
    rather than at every layer, which is what makes a 6-layer and a 24-layer
    model produce comparable matrices. It is not saved: the report is small and
    the lens for pythia-410m is 24 x 1024^2 floats.
    """
    import jlens
    from jlens.lenscontrol import lens_stage, matched_depth_layers

    device = pick_device(args.device)
    hf, tok = load_pythia(size, step, args.dtype, device)
    model = jlens.from_hf(hf, tok)
    prompts = fitting_corpus(budget.n_prompts)
    layers, _ = matched_depth_layers(model.n_layers, budget.n_probe_layers)

    lens = jlens.fit(
        model, prompts,
        source_layers=layers,
        dim_batch=budget.dim_batch,
        max_seq_len=budget.max_seq_len,
        skip_first=budget.skip_first,
        checkpoint_path=None,
        log_every=0,
    )
    report = lens_stage(model, lens, prompts, budget, d_ref=args.d_ref)

    n_layers, d_model, params = PYTHIA[size]
    out = report.summary()
    out.update({
        "size": size, "step": step, "n_layers": n_layers, "d_model": d_model,
        "params": params, "tokens": step * TOKENS_PER_STEP,
        "compute_flops": 6 * params * step * TOKENS_PER_STEP,
    })
    del hf, model, lens
    if device == "cuda":
        torch.cuda.empty_cache()
    if args.purge:
        purge_cache(size, step)
    return out


# ----------------------------------------------------------------------------
# Reading the two stages back together
# ----------------------------------------------------------------------------


def _lens_measures(row: dict) -> dict:
    """Collapse one stage-2 report into the scalars worth sweeping.

    Deliberately several rather than one. Picking a single headline number
    after seeing the data is how a sweep talks itself into a transition; the
    report runs the identical control on all of them and prints them together,
    so a result that only shows up in one measure is visible as such.
    """
    geo = row.get("geometry") or []
    fid = row.get("fidelity") or {}
    kurt = [
        g["readout_kurtosis"] - g["kurtosis_transport_null"]
        for g in geo
        if np.isfinite(g.get("readout_kurtosis", np.nan))
        and np.isfinite(g.get("kurtosis_transport_null", np.nan))
    ]
    pr = [g["projected_pr_ratio"] for g in geo
          if np.isfinite(g.get("projected_pr_ratio", np.nan))]
    return {
        "lens_advantage": fid.get("advantage", float("nan")),
        "lens_best_r": fid.get("best_r", float("nan")),
        "blockiness": (row.get("blocks") or {}).get("blockiness", float("nan")),
        "max_projected_pr_ratio": max(pr) if pr else float("nan"),
        "max_kurtosis_advantage": max(kurt) if kurt else float("nan"),
    }


def report_stage(out_dir: str) -> None:
    """Join stage 1 and stage 2 and control the result for lens quality."""
    from jlens.scaling import collapse_test, compare_power_laws, partial_correlation

    ig_path = os.path.join(out_dir, "ignition.json")
    ln_path = os.path.join(out_dir, "lens.json")
    ignition = json.load(open(ig_path)) if os.path.exists(ig_path) else []
    lens_rows = json.load(open(ln_path)) if os.path.exists(ln_path) else []
    if not ignition and not lens_rows:
        raise SystemExit(f"no results in {out_dir}; run --stage ignition first")

    prints = int(len({r["budget"]["fingerprint"] for r in lens_rows})) if lens_rows else 0
    if prints > 1:
        print("!! stage 2 rows carry more than one budget fingerprint. They were")
        print("!! measured under different settings and must not be pooled.\n")

    # ---- stage 1: is there a break at all? -------------------------------
    trained = [r for r in ignition if r["step"] > 0
               and np.isfinite(r.get("max_sharpness", np.nan))]
    if len(trained) >= 6:
        print("=" * 78)
        print("STAGE 1 (lens-free) — does a break beat a smooth power law?")
        print("=" * 78)
        x = np.array([max(r["compute_flops"], 1.0) for r in trained])
        y = np.array([r["max_sharpness"] for r in trained])
        res = compare_power_laws(x, y)
        if "error" in res:
            print(f"  {res['error']}")
        else:
            print(f"  smooth R2 {res['smooth'].r2:.3f}   broken R2 {res['broken'].r2:.3f}")
            print(f"  naive F p {res['f_p_naive']:.4f}  (optimistic — ignores the "
                  f"breakpoint search)")
            print(f"  bootstrap p {res['p_bootstrap']:.4f}   dBIC {res['delta_bic']:+.1f}")
            if res["breakpoint_ci95"]:
                lo, hi = res["breakpoint_ci95"]
                print(f"  breakpoint {res['breakpoint']:.3g} FLOPs "
                      f"(95% CI {lo:.3g}-{hi:.3g})")
            print(f"  -> {res['verdict']}")
        print()
        col = collapse_test(y, {
            "compute": x,
            "params": np.array([r["params"] for r in trained], float),
            "tokens": np.array([max(r["tokens"], 1) for r in trained], float),
            "depth": np.array([r["n_layers"] for r in trained], float),
        })
        for name in ("compute", "params", "tokens", "depth"):
            v = col.get(name, {})
            if "r2" in v:
                print(f"  collapse onto {name:8s} R2 {v['r2']:+.3f} "
                      f"residual RMS {v['residual_rms']:.4f}")
        print(f"  -> tightest collapse: {col.get('best')}")
        print()

    if not lens_rows:
        print("no stage-2 rows yet; run --stage lens for the instrument control")
        return

    # ---- stage 2: the instrument, and the result net of it ---------------
    print("=" * 78)
    print("STAGE 2 — the instrument first")
    print("=" * 78)
    print(f"  {'size':>6s} {'step':>7s} {'lens r':>7s} {'vs logit':>9s} "
          f"{'blockiness':>11s} {'PR ratio':>9s}")
    measures = []
    for row in sorted(lens_rows, key=lambda r: (r["params"], r["step"])):
        m = _lens_measures(row)
        m.update({k: row[k] for k in
                  ("size", "step", "params", "tokens", "compute_flops", "n_layers")})
        measures.append(m)
        flag = "  <- lens dead" if not (m["lens_advantage"] > 0) else ""
        print(f"  {row['size']:>6s} {row['step']:>7d} {m['lens_best_r']:7.3f} "
              f"{m['lens_advantage']:+9.3f} {m['blockiness']:11.3f} "
              f"{m['max_projected_pr_ratio']:9.3f}{flag}")
    dead = [m for m in measures if not (m["lens_advantage"] > 0)]
    if dead:
        print(f"\n  {len(dead)}/{len(measures)} rungs have a lens no better than "
              f"the logit lens. Their geometry")
        print("  numbers measure the instrument rather than the model, and are "
              "excluded")
        print("  from the controlled fits below.")

    live = [m for m in measures if m["lens_advantage"] > 0
            and m["compute_flops"] > 0]
    print()
    print("=" * 78)
    print("STAGE 2 — does the result survive controlling for lens quality?")
    print("=" * 78)
    if len(live) < 6:
        print(f"  only {len(live)} usable rungs; need >=6 for the partial "
              f"correlation. Add checkpoints or sizes.")
        return
    logC = np.log10([m["compute_flops"] for m in live])
    adv = np.array([m["lens_advantage"] for m in live])
    print(f"  lens advantage itself tracks compute at r = "
          f"{np.corrcoef(logC, adv)[0, 1]:+.3f} — that is the confound, sized.")
    print()
    print(f"  {'measure':>24s} {'raw r':>7s} {'partial r':>10s} {'shrinkage':>10s} "
          f"{'left of y':>10s}")
    for name in ("blockiness", "max_projected_pr_ratio", "max_kurtosis_advantage"):
        y = np.array([m[name] for m in live], float)
        pc = partial_correlation(y, logC, {"lens_advantage": adv})
        if "error" in pc:
            print(f"  {name:>24s} {pc['error']}")
            continue
        print(f"  {name:>24s} {pc['raw']:+7.3f} {pc['partial']:+10.3f} "
              f"{pc['shrinkage']:+10.3f} {pc['residual_frac_y']:9.1%}")
        if "note" in pc:
            print(f"  {'':>24s} -> {pc['note']}")
    print()
    print("  'left of y' is the share of the measure's variance still standing "
          "after lens")
    print("  advantage is regressed out. A raw correlation that collapses to near "
          "zero was")
    print("  the instrument improving with scale, not the workspace. One that "
          "survives is a")
    print("  claim about the model.")

    # ---- triangulation ---------------------------------------------------
    onset = {(r["size"], r["step"]): r.get("onset_rho") for r in ignition}
    pairs = [
        (onset[(m["size"], m["step"])],
         (next(r for r in lens_rows
               if r["size"] == m["size"] and r["step"] == m["step"])
          .get("blocks", {}).get("boundaries_rho") or [np.nan])[0])
        for m in live if (m["size"], m["step"]) in onset
    ]
    pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) >= 4:
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        print()
        print("=" * 78)
        print("TRIANGULATION — lens-free onset vs lens-derived block boundary")
        print("=" * 78)
        print(f"  n={len(pairs)}  correlation {np.corrcoef(a, b)[0, 1]:+.3f}  "
              f"mean |difference| in rho {np.abs(a - b).mean():.3f}")
        print("  Agreement is the point: two measurements sharing no machinery")
        print("  landing in the same place is evidence neither one alone gives.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="70m,160m,410m")
    ap.add_argument("--steps", default="0,1,2,4,8,16,32,64,128,256,512,"
                                       "1000,4000,16000,64000,143000")
    ap.add_argument("--stage", default="ignition",
                    choices=["ignition", "lens", "report"],
                    help="ignition = stage 1, lens-free; lens = stage 2, fits a "
                         "J-lens per rung under a fixed budget; report = read "
                         "both back and control the result for lens quality")
    ap.add_argument("--n-alphas", type=int, default=21)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="auto",
                    help="auto picks cuda when available, else cpu. The "
                         "ignition stage is forward-passes only: pythia-410m "
                         "takes ~2 min for 40 trials on four CPU cores. The "
                         "lens stage wants a GPU.")
    ap.add_argument("--purge", action="store_true",
                    help="delete each checkpoint from the HF cache after "
                         "measuring it, so peak disk stays at one model")
    ap.add_argument("--out", default="runs/pythia")

    lens_args = ap.add_argument_group(
        "stage 2 budget",
        "Held identical across every rung. Changing any of these changes the "
        "budget fingerprint, and rows with different fingerprints must not be "
        "pooled into one curve.",
    )
    lens_args.add_argument("--lens-prompts", type=int, default=16)
    lens_args.add_argument("--lens-seq-len", type=int, default=128)
    lens_args.add_argument("--lens-skip-first", type=int, default=16)
    lens_args.add_argument("--lens-probe-layers", type=int, default=5,
                           help="set from the SHALLOWEST model in the sweep "
                                "(pythia-70m has 6 layers, so 5 source layers)")
    lens_args.add_argument("--lens-probe-tokens", type=int, default=512)
    lens_args.add_argument("--dim-batch", type=int, default=8)
    lens_args.add_argument("--d-ref", type=int, default=256,
                           help="reference width every model is projected to "
                                "before its geometry is measured; set at or "
                                "below the NARROWEST model's d_model")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.stage == "report":
        report_stage(args.out)
        return

    from jlens.lenscontrol import FittingBudget

    budget = FittingBudget(
        n_prompts=args.lens_prompts,
        max_seq_len=args.lens_seq_len,
        skip_first=args.lens_skip_first,
        n_probe_layers=args.lens_probe_layers,
        n_probe_tokens=args.lens_probe_tokens,
        dim_batch=args.dim_batch,
    )

    sizes = args.sizes.split(",")
    steps = [int(s) for s in args.steps.split(",")]
    name = "ignition" if args.stage == "ignition" else "lens"
    path = os.path.join(args.out, f"{name}.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["size"], r["step"]) for r in rows}

    print(f"stage {name}: {len(sizes)} sizes x {len(steps)} steps = "
          f"{len(sizes) * len(steps)} model instances "
          f"({len(done)} already done)")
    if args.stage == "lens":
        print(f"  budget {budget.fingerprint()}: {budget.n_prompts} prompts x "
              f"{budget.samples_per_entry // max(budget.n_prompts, 1)} positions "
              f"= {budget.samples_per_entry} samples per Jacobian entry")
        stale = {r["budget"]["fingerprint"] for r in rows} - {budget.fingerprint()}
        if stale:
            raise SystemExit(
                f"{path} already holds rows fitted under budget(s) {sorted(stale)}, "
                f"which differ from the current {budget.fingerprint()}. Pooling "
                f"them would compare budgets rather than models — write to a "
                f"different --out, or match the budget."
            )

    for size in sizes:
        for step in steps:
            if (size, step) in done:
                continue
            try:
                if args.stage == "ignition":
                    row = run_ignition(size, step, args)
                else:
                    row = run_lens(size, step, args, budget)
            except Exception as exc:  # a missing revision should not kill the sweep
                print(f"  {size}/step{step}: FAILED {type(exc).__name__}: {exc}")
                continue
            rows.append(row)
            with open(path, "w") as f:
                json.dump(rows, f, indent=2)
            if args.stage == "ignition":
                print(f"  {size:>5s} step{step:<7d} sharpness "
                      f"{row['max_sharpness']:6.2f}  onset_rho {row['onset_rho']:.2f}")
            else:
                m = _lens_measures(row)
                warn = f"   !! {row['warnings'][0]}" if row["warnings"] else ""
                print(f"  {size:>5s} step{step:<7d} lens r "
                      f"{m['lens_best_r']:5.3f}  vs logit {m['lens_advantage']:+.3f}"
                      f"  blockiness {m['blockiness']:.3f}{warn}")

    if not rows:
        return
    print(f"\nwrote {path}")
    print()
    report_stage(args.out)


if __name__ == "__main__":
    main()
