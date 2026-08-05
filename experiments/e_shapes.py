"""What do the different J-space shapes actually indicate?

The viz shows one layer x layer CKA heatmap per open model, segmented into
blocks, and the models do not all look the same: some read as two blocks, some
as three, and the boundaries sit at different relative depths. The natural
reading is that these are *facts about the models* — that this model has a
wider workspace, that one an earlier sensory/workspace transition.

Before believing that, three things have to be ruled out, and all three are
testable with the CKA matrices already on disk:

**(1) Is the block count a property of the model, or of the segmenter?**
``n_blocks`` comes from ``argmax_k [blockiness(k) - penalty*(k-1)]`` with
``penalty = 0.02``. That threshold is a choice. If a model's ``k`` flips
between 2 and 3 under a small change to the penalty, then "2-block model" vs
"3-block model" is a statement about the penalty, not the model. We measure
each model's *margin*: how far the penalty can move before ``k`` changes.

**(2) Is a boundary reproducible?** The same model fitted on a different
corpus, at a different prompt count, or under a different quantization should
give the same shape if the shape is real. The repo has all three ablations.

**(3) Is the early block a regime, or is it degeneracy?** The paper flags this
as open: the J-lens may carry no early content because there *is* none, or
because the lens cannot read it (``J_l`` relates layer ``l`` to the final
layer, and early layers are far away). These leave different fingerprints in
the CKA matrix. A genuine regime should still have internal structure — layers
near each other more similar than layers far apart. Degeneracy — all early
``V_l`` collapsing onto one shared low-rank subspace — makes every early layer
similar to every other *equally*, with no distance decay. We measure the
within-block distance-decay slope and the within-block CKA spread.

Reads ``runs/E*/results/*.npz`` from a clone of ``eliebak/open-jlens-data``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from jlens.cka import best_segmentation, blockiness_of, segment_blocks


def load_matrices(root: str) -> list[dict]:
    """Every standard CKA matrix in the data repo, with its metadata."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "runs/**/*.npz"), recursive=True)):
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        keys = set(d.keys())
        if not {"C", "source_layers"} <= keys:
            continue
        C = np.asarray(d["C"], dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 6:
            continue
        n_layers = int(d["n_layers"]) if "n_layers" in keys else C.shape[0] + 1
        out.append(
            {
                "path": os.path.relpath(path, root),
                "C": C,
                "source_layers": [int(x) for x in d["source_layers"]],
                "n_layers": n_layers,
                "model_name": str(d["model_name"]) if "model_name" in keys else "",
                "experiment": os.path.relpath(path, root).split(os.sep)[1],
            }
        )
    return out


# ----------------------------------------------------------------------------
# (1) How fragile is the block count?
# ----------------------------------------------------------------------------


def penalty_margin(C: np.ndarray, source_layers, n_layers: int) -> dict:
    """Range of segmentation penalties over which ``n_blocks`` stays put.

    Sweeps the penalty and records which ``k`` wins. A model whose ``k`` only
    holds over a sliver of penalty space is not really an "``k``-block model".
    """
    ks, penalties = [], np.round(np.arange(0.0, 0.201, 0.002), 4)
    for p in penalties:
        ks.append(segment_blocks(C, source_layers, n_layers, penalty_per_block=p).n_blocks)
    ks = np.array(ks)
    k_default = int(ks[np.argmin(np.abs(penalties - 0.02))])
    same = penalties[ks == k_default]
    return {
        "k_at_default": k_default,
        "k_range": [int(ks.min()), int(ks.max())],
        "stable_penalty_lo": float(same.min()) if len(same) else float("nan"),
        "stable_penalty_hi": float(same.max()) if len(same) else float("nan"),
        "margin": float(same.max() - same.min()) if len(same) else 0.0,
        "k_by_penalty": {float(p): int(k) for p, k in zip(penalties[::10], ks[::10])},
    }


def boundary_stability(C: np.ndarray, source_layers, n_layers: int) -> dict:
    """Where the cuts land when ``k`` is *forced* to 2, 3 and 4.

    Forcing ``k`` sidesteps model selection entirely. If a boundary is real, it
    should appear at the same relative depth whether or not the segmenter was
    allowed to add another block elsewhere.
    """
    out = {}
    L = C.shape[0]
    for k in (2, 3, 4):
        if k > L:
            continue
        seg = best_segmentation(C, k)
        B, _, _ = blockiness_of(C, seg)
        out[f"k{k}"] = {
            "boundaries_rho": [source_layers[i] / (n_layers - 1) for i, _ in seg[1:]],
            "blockiness": float(B),
        }
    return out


# ----------------------------------------------------------------------------
# (3) Regime or degeneracy?
# ----------------------------------------------------------------------------


def distance_decay(C: np.ndarray, lo: int, hi: int) -> dict:
    """Does within-block CKA fall off with layer distance?

    Regresses CKA on ``|i - j|`` inside ``[lo, hi]``. A genuine processing
    regime drifts: adjacent layers are more alike than distant ones, so the
    slope is clearly negative and the spread is appreciable. A degenerate band
    — every ``V_l`` collapsed onto the same low-rank subspace — is flat and
    tight: slope near zero, spread near zero, mean near one.
    """
    idx = np.arange(lo, hi + 1)
    if len(idx) < 4:
        return {"n": len(idx), "slope": float("nan"), "mean": float("nan"),
                "std": float("nan"), "r2": float("nan")}
    sub = C[np.ix_(idx, idx)]
    iu, ju = np.triu_indices(len(idx), k=1)
    dist = (ju - iu).astype(float)
    vals = sub[iu, ju]
    A = np.vstack([dist, np.ones_like(dist)]).T
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    pred = A @ coef
    ss_res = float(((vals - pred) ** 2).sum())
    ss_tot = float(((vals - vals.mean()) ** 2).sum())
    return {
        "n": int(len(idx)),
        "slope": float(coef[0]),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def regime_vs_degeneracy(C: np.ndarray, source_layers, n_layers: int) -> dict:
    """Compare the first block against the rest on the decay diagnostic."""
    res = segment_blocks(C, source_layers, n_layers)
    segs = res.segments
    first = distance_decay(C, *segs[0])
    rest_lo, rest_hi = segs[1][0], segs[-1][1]
    rest = distance_decay(C, rest_lo, rest_hi)
    return {
        "n_blocks": res.n_blocks,
        "first_block": first,
        "later_blocks": rest,
        "first_block_rho": [
            source_layers[segs[0][0]] / (n_layers - 1),
            source_layers[segs[0][1]] / (n_layers - 1),
        ],
        # A flat, tight, near-1 first block is the degeneracy signature.
        "degeneracy_score": float(first["mean"] - first["std"] * 4 - abs(first["slope"]) * 40)
        if first["n"] >= 4
        else float("nan"),
    }


# ----------------------------------------------------------------------------
# Grouping for the reproducibility question
# ----------------------------------------------------------------------------


def model_key(entry: dict) -> str:
    """Collapse ablation variants onto the underlying model."""
    name = entry["model_name"] or entry["path"]
    name = re.sub(r"\s*\(.*", "", name).strip()
    stem = os.path.basename(entry["path"])
    for tag in ("frozenattn", "vpdiag", "vpheads", "nvfp4", "bf16", "n710", "n250"):
        stem = stem.replace(tag, "")
    return f"{name}|L{entry['n_layers']}"


def variant_label(entry: dict) -> str:
    p = entry["path"]
    for tag in (
        "frozenattn", "vpdiag", "vpheads", "nvfp4", "bf16", "n710", "n250",
        "code", "math", "finepdfs", "wikitext_orig", "wikitext", "4k",
    ):
        if tag in p:
            return tag
    return "default"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default="/workspace/eliebak/open-jlens-data",
        help="clone of eliebak/open-jlens-data",
    )
    ap.add_argument("--out", default="runs/shapes")
    args = ap.parse_args()

    entries = load_matrices(args.data_root)
    if not entries:
        raise SystemExit(f"no CKA matrices under {args.data_root}/runs")
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(entries)} CKA matrices\n")

    records = []
    for e in entries:
        base = segment_blocks(e["C"], e["source_layers"], e["n_layers"])
        rec = {
            "path": e["path"],
            "model_name": e["model_name"],
            "experiment": e["experiment"],
            "n_layers": e["n_layers"],
            "variant": variant_label(e),
            "model_key": model_key(e),
            "n_blocks": base.n_blocks,
            "blockiness": base.blockiness,
            "layer_coupling": base.layer_coupling,
            "boundaries_rho": base.boundaries_rho,
            "penalty": penalty_margin(e["C"], e["source_layers"], e["n_layers"]),
            "forced": boundary_stability(e["C"], e["source_layers"], e["n_layers"]),
            "degeneracy": regime_vs_degeneracy(e["C"], e["source_layers"], e["n_layers"]),
        }
        records.append(rec)

    # ---- (1) fragility of the block count --------------------------------
    print("=" * 78)
    print("(1) IS n_blocks A PROPERTY OF THE MODEL OR OF THE PENALTY?")
    print("=" * 78)
    margins = np.array([r["penalty"]["margin"] for r in records])
    ks = np.array([r["n_blocks"] for r in records])
    print(f"  n_blocks at default penalty 0.02: "
          f"{dict(zip(*np.unique(ks, return_counts=True)))}")
    print(f"  penalty range over which k is unchanged: "
          f"median {np.median(margins):.3f}, "
          f"min {margins.min():.3f}, max {margins.max():.3f} "
          f"(the full swept range is 0.000-0.200)")
    fragile = [r for r in records if r["penalty"]["margin"] < 0.02]
    print(f"  FRAGILE matrices (k flips within 0.02 of the default): "
          f"{len(fragile)}/{len(records)}")
    for r in fragile:
        print(f"      {(r['model_name'] or r['path'])[:60]}  k={r['n_blocks']}  "
              f"stable only over "
              f"[{r['penalty']['stable_penalty_lo']:.3f},"
              f"{r['penalty']['stable_penalty_hi']:.3f}]")
    spans = np.array([r["penalty"]["k_range"][1] - r["penalty"]["k_range"][0]
                      for r in records])
    print(f"  distinct k values reachable across the whole penalty sweep: "
          f"median span {np.median(spans):.1f}, max {spans.max()}")
    bl = np.array([r["blockiness"] for r in records])
    print(f"  blockiness at the selected k: median {np.median(bl):.3f}, "
          f"range {bl.min():.3f}-{bl.max():.3f}")
    n2 = sum(1 for r in records if r["n_blocks"] == 2)
    print(f"\n  NOTE: {n2}/{len(records)} matrices segment into TWO blocks, so their")
    print("  'sensory_end' and 'motor_start' are the SAME cut. In those models the")
    print("  segmentation has not found a separate motor band at all — a three-regime")
    print("  reading of those pictures is imposed by the reader, not by the data.")

    # ---- (2) reproducibility of the boundaries ---------------------------
    print()
    print("=" * 78)
    print("(2) DO ABLATION VARIANTS OF THE SAME MODEL GIVE THE SAME SHAPE?")
    print("=" * 78)
    groups = defaultdict(list)
    for r in records:
        groups[r["model_key"]].append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    early_spreads, late_spreads = [], []
    for key, group in sorted(multi.items()):
        k2 = [g["forced"]["k2"]["boundaries_rho"][0] for g in group if "k2" in g["forced"]]
        k3 = [g["forced"]["k3"]["boundaries_rho"] for g in group if "k3" in g["forced"]]
        early = [b[0] for b in k3 if b]
        late = [b[-1] for b in k3 if b]
        if len(k2) > 1:
            late_spreads.append(np.ptp(k2))
        if len(early) > 1:
            early_spreads.append(np.ptp(early))
        print(f"  {key[:52]:52s} n={len(group)}")
        print(f"      variants: {sorted(g['variant'] for g in group)}")
        print(f"      n_blocks seen: {sorted({g['n_blocks'] for g in group})}")
        if len(k2) > 1:
            print(f"      forced-k=2 cut rho: {np.min(k2):.3f}-{np.max(k2):.3f} "
                  f"(spread {np.ptp(k2):.3f})")
        if len(early) > 1:
            print(f"      forced-k=3 cuts rho: early "
                  f"{np.min(early):.3f}-{np.max(early):.3f} (spread {np.ptp(early):.3f}), "
                  f"late {np.min(late):.3f}-{np.max(late):.3f} (spread {np.ptp(late):.3f})")
    if early_spreads and late_spreads:
        print()
        print(f"  ACROSS GROUPS: median spread of the single forced cut  "
              f"{np.median(late_spreads):.3f} rho")
        print(f"                 median spread of the forced-k=3 early cut "
              f"{np.median(early_spreads):.3f} rho")

    # ---- (3) regime or degeneracy ---------------------------------------
    print()
    print("=" * 78)
    print("(3) IS THE FIRST BLOCK A REGIME, OR A DEGENERATE BAND?")
    print("=" * 78)
    print(f"  {'model':46s} {'first-block':>22s}   {'later blocks':>22s}")
    print(f"  {'':46s} {'mean   std   slope':>22s}   {'mean   std   slope':>22s}")
    fb_slope, lb_slope, fb_std, lb_std = [], [], [], []
    for r in sorted(records, key=lambda r: r["path"]):
        f, l = r["degeneracy"]["first_block"], r["degeneracy"]["later_blocks"]
        if not (f["n"] >= 4 and l["n"] >= 4):
            continue
        fb_slope.append(f["slope"]); lb_slope.append(l["slope"])
        fb_std.append(f["std"]); lb_std.append(l["std"])
        label = (r["model_name"] or r["path"])[:46]
        print(f"  {label:46s} {f['mean']:6.3f}{f['std']:6.3f}{f['slope']:8.4f}   "
              f"{l['mean']:6.3f}{l['std']:6.3f}{l['slope']:8.4f}")
    if fb_slope:
        print()
        print(f"  medians: first block  std {np.median(fb_std):.3f}  "
              f"slope {np.median(fb_slope):+.4f}")
        print(f"           later blocks std {np.median(lb_std):.3f}  "
              f"slope {np.median(lb_slope):+.4f}")
        flat = sum(1 for s, d in zip(fb_slope, fb_std) if abs(s) < 0.005 and d < 0.05)
        print(f"  first blocks that are flat AND tight (|slope|<0.005, std<0.05): "
              f"{flat}/{len(fb_slope)}")

    # ---- (4) a controlled comparison: same model, different optimizer ----
    opt = [r for r in records if r["experiment"] == "E11_optimizers"]
    if opt:
        print()
        print("=" * 78)
        print("(4) CONTROLLED: SAME ARCHITECTURE AND DATA, DIFFERENT OPTIMIZER")
        print("=" * 78)
        print("  If the shape were a fact about the trained model, optimizers that")
        print("  reach comparable loss should reach comparable shapes.")
        print(f"  {'run':28s} {'k':>2s} {'blockiness':>11s} {'coupling':>9s} "
              f"{'k=2 cut':>8s} {'k=3 cuts':>18s}")
        for r in sorted(opt, key=lambda r: r["path"]):
            c2 = r["forced"].get("k2", {}).get("boundaries_rho", [float("nan")])
            c3 = r["forced"].get("k3", {}).get("boundaries_rho", [])
            print(f"  {r['model_name'][:28]:28s} {r['n_blocks']:2d} "
                  f"{r['blockiness']:11.3f} {r['layer_coupling']:9.3f} "
                  f"{c2[0]:8.3f} "
                  f"{'  '.join(f'{x:.3f}' for x in c3):>18s}")
        cuts2 = np.array([r["forced"]["k2"]["boundaries_rho"][0] for r in opt])
        blk = np.array([r["blockiness"] for r in opt])
        print(f"\n  forced-k=2 cut across {len(opt)} optimizers: "
              f"{cuts2.min():.3f}-{cuts2.max():.3f} (spread {np.ptp(cuts2):.3f} rho)")
        print(f"  blockiness across the same runs:  "
              f"{blk.min():.3f}-{blk.max():.3f} (spread {np.ptp(blk):.3f})")

    # ---- (5) does the shape track model depth? ---------------------------
    print()
    print("=" * 78)
    print("(5) DOES THE SHAPE TRACK MODEL DEPTH?")
    print("=" * 78)
    depths = np.array([r["n_layers"] for r in records], dtype=float)
    blk = np.array([r["blockiness"] for r in records])
    coup = np.array([r["layer_coupling"] for r in records])
    cut = np.array([
        r["forced"]["k2"]["boundaries_rho"][0] if "k2" in r["forced"] else np.nan
        for r in records
    ])
    ok = np.isfinite(cut)
    print(f"  depth range {depths.min():.0f}-{depths.max():.0f} layers "
          f"over {len(records)} matrices")
    print(f"  corr(n_layers, blockiness)      {np.corrcoef(depths, blk)[0, 1]:+.3f}")
    print(f"  corr(n_layers, layer_coupling)  {np.corrcoef(depths, coup)[0, 1]:+.3f}")
    print(f"  corr(n_layers, forced-k=2 cut)  "
          f"{np.corrcoef(depths[ok], cut[ok])[0, 1]:+.3f}")
    print()
    print(f"  {'depth':>6s} {'n':>3s} {'blockiness':>11s} {'coupling':>9s} {'k=2 cut':>8s}")
    for d in sorted(set(depths)):
        sel = depths == d
        print(f"  {d:6.0f} {sel.sum():3d} {blk[sel].mean():11.3f} "
              f"{coup[sel].mean():9.3f} "
              f"{np.nanmean(cut[sel]) if np.isfinite(cut[sel]).any() else float('nan'):8.3f}")

    with open(os.path.join(args.out, "shapes.json"), "w") as f:
        json.dump(records, f, indent=2, default=float)
    print(f"\nwrote {args.out}/shapes.json")


if __name__ == "__main__":
    main()
