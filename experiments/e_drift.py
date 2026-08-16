"""Which depths change during training, and which stay put?

The hypothesis this tests: a transformer's *readout geometry* does not move
uniformly during pretraining. Early layers settle on a tokenizer-shaped input
code and stop; late layers settle on an unembedding-shaped output code and
stop; the middle keeps reorganising for far longer. If that is true it is a
statement about where a network's plasticity lives, and it has a direct
consequence for continual learning — the depths worth freezing and the depths
worth leaving free are identifiable from geometry alone, before you train.

What is actually measured
-------------------------
The quantity you would *want* is cross-checkpoint CKA of ``V_l`` — how much
layer ``l``'s own J-lens vectors rotated between checkpoint ``t`` and ``t+1``.
That needs the fitted lens tensors, which the open campaign does not publish
(0.02-63 GB each).

What it does publish is the layer x layer CKA matrix ``C_t`` per checkpoint. So
this measures a proxy: how much layer ``l``'s **relational profile** moved,

    drift_t(l) = || C_t[l, :] - C_{t-1}[l, :] ||_2 / sqrt(L)

i.e. how differently layer ``l`` relates to every other layer than it did at
the previous checkpoint. A layer whose geometry is frozen keeps its
relationships; a layer that reorganises does not. It is a proxy and is labelled
as one — a layer *could* rotate in a way that preserves every CKA to every
other layer, and this would miss it. That failure mode makes the measure
conservative rather than permissive: it under-reports drift, never invents it.

Two controls, because the raw curve has two obvious confounds:

**Global drift.** If the whole matrix shifts between checkpoints, every row
moves and every layer looks plastic. The centred curve subtracts the
per-checkpoint mean, so what is reported is *which depths moved more than the
model as a whole*.

**Edge effects.** Row 0 and row L-1 are not special in the norm — every row has
L entries — but the first and last layers sit adjacent to the embedding and
unembedding, which are shared across the whole vocabulary. That is a real
reason they might be stable rather than an artifact, and it is the substantive
claim, so the report gives the per-third breakdown rather than only a
correlation.

**The saturation ceiling, which turns out to decide the answer.** A row already
near 1.0 against every other layer cannot move much, whatever training does to
it. Middle rows are exactly those rows: they have the highest mean CKA and the
smallest spread. So "the middle drifts least" and "the middle has no room to
drift" predict the same raw curve, and the raw curve cannot tell them apart.
The report therefore prints ``corr(drift, row mean CKA)`` — the size of the
confound — and the headroom-normalised curve beside the raw one. On the open
campaign's three training runs the two disagree, which is the finding: this
proxy is not sufficient to settle the question, and saying so is the result.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from jlens.atlas import read_entry
from jlens.checkpoints import parse_ref


def load_series(results_dir: str) -> list[dict]:
    """Every checkpoint in one training run, in training order."""
    out = []
    for path in glob.glob(os.path.join(results_dir, "cka_*.npz")):
        stem = os.path.basename(path)[len("cka_"):-len(".npz")]
        try:
            rec = read_entry(path)
        except Exception:
            continue
        C = rec["C"]
        if np.ndim(C) != 2 or C.shape[0] != C.shape[1]:
            continue
        cp = parse_ref(stem)
        step = cp.step if cp else None
        if step is None:
            m = re.search(r"step(\d+)", stem)
            step = int(m.group(1)) if m else None
        m = re.search(r"\(([\d.]+)([KMBT])[,)]", rec["model_name"])
        tokens = (float(m.group(1)) * {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)]
                  if m else None)
        out.append({"label": stem, "step": step, "tokens": tokens,
                    "C": C, "rho": np.asarray(rec["rho"], float)})
    out.sort(key=lambda r: (r["step"] is None, r["step"] or 0))
    return out


def drift_profile(series: list[dict]) -> dict:
    """Per-layer drift between consecutive checkpoints, raw and centred."""
    if len(series) < 2:
        return {}
    L = series[0]["C"].shape[0]
    if any(r["C"].shape[0] != L for r in series):
        return {"error": "checkpoints disagree on layer count"}

    raw, centred, spans = [], [], []
    for a, b in zip(series[:-1], series[1:], strict=True):
        d = np.linalg.norm(b["C"] - a["C"], axis=1) / np.sqrt(L)
        raw.append(d)
        centred.append(d - d.mean())
        spans.append((a["label"], b["label"]))

    raw = np.array(raw)
    centred = np.array(centred)
    # Total reorganisation over the whole run, not just step to step: a layer
    # can drift and drift back, and the endpoints are what a continual-learning
    # decision would actually care about.
    total = np.linalg.norm(series[-1]["C"] - series[0]["C"], axis=1) / np.sqrt(L)
    return {
        "rho": series[0]["rho"], "L": L, "spans": spans,
        "raw": raw, "centred": centred,
        "mean_raw": raw.mean(axis=0), "mean_centred": centred.mean(axis=0),
        "total": total,
    }


def thirds(rho: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Mean of ``values`` over the shallow, middle and deep thirds of depth."""
    lo = values[rho < 1 / 3]
    mid = values[(rho >= 1 / 3) & (rho <= 2 / 3)]
    hi = values[rho > 2 / 3]
    f = lambda v: float(v.mean()) if len(v) else float("nan")
    return f(lo), f(mid), f(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                    help="clone of eliebak/open-jlens-data (the repo root or "
                         "its runs/ directory)")
    ap.add_argument("--out", default="runs/drift")
    args = ap.parse_args()

    root = args.data_root
    if os.path.isdir(os.path.join(root, "runs")):
        root = os.path.join(root, "runs")
    os.makedirs(args.out, exist_ok=True)

    runs = sorted(glob.glob(os.path.join(root, "E2_training", "*", "results")))
    if not runs:
        raise SystemExit(f"no E2_training runs under {root}")

    report = {}
    print("=" * 78)
    print("PER-LAYER J-SPACE DRIFT ACROSS TRAINING")
    print("=" * 78)
    print("drift = how much a layer's relations to every other layer moved")
    print("between consecutive checkpoints. Centred = relative to the whole")
    print("model's move at that step, so a global shift does not count.\n")

    for results_dir in runs:
        name = os.path.basename(os.path.dirname(results_dir))
        series = load_series(results_dir)
        prof = drift_profile(series)
        if not prof or "error" in prof:
            print(f"{name}: {prof.get('error', 'too few checkpoints')}\n")
            continue

        rho = prof["rho"]
        print(f"--- {name}: {len(series)} checkpoints, {prof['L']} layers ---")
        print(f"    {series[0]['label']} .. {series[-1]['label']}")

        lo, mid, hi = thirds(rho, prof["mean_raw"])
        edges = np.nanmean([lo, hi])
        print(f"\n    mean drift by depth third:")
        print(f"      shallow (rho<0.33)  {lo:.4f}")
        print(f"      middle  (0.33-0.67) {mid:.4f}")
        print(f"      deep    (rho>0.67)  {hi:.4f}")
        if np.isfinite(edges) and edges > 0:
            print(f"      middle / edges      {mid / edges:.2f}x")

        tlo, tmid, thi = thirds(rho, prof["total"])
        tedges = np.nanmean([tlo, thi])
        print(f"    end-to-end reorganisation (first -> last checkpoint):")
        print(f"      shallow {tlo:.4f}   middle {tmid:.4f}   deep {thi:.4f}"
              + (f"   ({tmid / tedges:.2f}x)" if tedges > 0 else ""))

        peak = int(np.argmax(prof["mean_raw"]))
        print(f"    peak drift at layer {peak} (rho={rho[peak]:.2f})")

        # --- the saturation ceiling ---------------------------------------
        C0 = series[0]["C"]
        row_mean, row_std = C0.mean(axis=1), C0.std(axis=1)
        r_sat = float(np.corrcoef(prof["mean_raw"], row_mean)[0, 1])
        mlo, mmid, mhi = thirds(rho, row_mean)
        norm = prof["mean_raw"] / np.maximum(row_std, 1e-9)
        nlo, nmid, nhi = thirds(rho, norm)
        nedges = np.nanmean([nlo, nhi])
        print(f"\n    CEILING CHECK")
        print(f"      corr(drift, row mean CKA) = {r_sat:+.3f}"
              f"  (negative => saturated rows cannot move)")
        print(f"      row mean CKA:  shallow {mlo:.3f}  middle {mmid:.3f}  "
              f"deep {mhi:.3f}")
        print(f"      drift / headroom:  shallow {nlo:.3f}  middle {nmid:.3f}  "
              f"deep {nhi:.3f}"
              + (f"   middle/edges {nmid / nedges:.2f}x" if nedges > 0 else ""))
        raw_ratio = mid / edges if (np.isfinite(edges) and edges > 0) else float("nan")
        norm_ratio = nmid / nedges if nedges > 0 else float("nan")
        if np.isfinite(raw_ratio) and np.isfinite(norm_ratio) and \
                (raw_ratio - 1) * (norm_ratio - 1) < 0:
            print(f"      -> raw and corrected DISAGREE ({raw_ratio:.2f}x vs "
                  f"{norm_ratio:.2f}x). This proxy cannot settle it.")

        # An inverted-U is the hypothesis; a monotone ramp is a different
        # claim. Quadratic fit in rho: a significantly negative rho^2 term is
        # the hump, and reporting it stops "the middle moves most" from resting
        # on where the thirds happen to be cut.
        if len(rho) >= 5:
            c = np.polyfit(rho, prof["mean_raw"], 2)
            shape = ("hump (middle peaks)" if c[0] < 0 else "bowl (edges peak)")
            print(f"    quadratic in rho: {c[0]:+.4f} rho^2 -> {shape}")

        report[name] = {
            "n_checkpoints": len(series),
            "L": prof["L"],
            "labels": [s["label"] for s in series],
            "rho": rho.tolist(),
            "mean_drift": prof["mean_raw"].tolist(),
            "mean_drift_centred": prof["mean_centred"].tolist(),
            "total_drift": prof["total"].tolist(),
            "thirds_mean": {"shallow": lo, "middle": mid, "deep": hi},
            "thirds_total": {"shallow": tlo, "middle": tmid, "deep": thi},
            "peak_rho": float(rho[peak]),
            "ceiling": {
                "corr_drift_row_mean": r_sat,
                "row_mean_thirds": {"shallow": mlo, "middle": mmid, "deep": mhi},
                "normalised_thirds": {"shallow": nlo, "middle": nmid, "deep": nhi},
                "raw_middle_over_edges": raw_ratio,
                "normalised_middle_over_edges": norm_ratio,
            },
        }

        # A compact profile, so the shape is visible without a plot.
        print("\n    drift vs depth:")
        m = prof["mean_raw"]
        span = max(m.max() - m.min(), 1e-12)
        step = max(len(rho) // 16, 1)
        for i in range(0, len(rho), step):
            bar = "#" * int(1 + 34 * (m[i] - m.min()) / span)
            print(f"      rho {rho[i]:.2f}  {m[i]:.4f}  {bar}")
        print()

    with open(os.path.join(args.out, "drift.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    raws = [r["ceiling"]["raw_middle_over_edges"] for r in report.values()]
    norms = [r["ceiling"]["normalised_middle_over_edges"] for r in report.values()]
    n_raw = sum(1 for v in raws if np.isfinite(v) and v > 1)
    n_norm = sum(1 for v in norms if np.isfinite(v) and v > 1)
    print(f"  middle drifts MORE than the edges:")
    print(f"    raw                {n_raw}/{len(report)} runs   "
          f"({', '.join(f'{v:.2f}x' for v in raws)})")
    print(f"    headroom-corrected {n_norm}/{len(report)} runs   "
          f"({', '.join(f'{v:.2f}x' for v in norms)})")
    if n_raw != n_norm:
        print("\n  The two corrections give different answers, so the honest")
        print("  reading is that this measure does not settle the question.")
        print("  What would: cross-checkpoint CKA of V_l itself, which needs the")
        print("  fitted lens tensors. `experiments/jspace_open.py --family pythia`")
        print("  produces them, on a family with 154 checkpoints per size rather")
        print("  than the 4-12 here.")
    print("\n  Standing caveat: this reads a layer's *relations* to other layers,")
    print("  not its vectors. A rotation preserving every pairwise CKA would be")
    print("  invisible, so it under-reports drift rather than inventing it.")
    print(f"\nwrote {args.out}/drift.json")


if __name__ == "__main__":
    main()
