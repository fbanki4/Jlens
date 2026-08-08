"""Making a phase transition earn its place.

If you sweep scale looking for the moment a workspace switches on, you will
find one. Apparent emergent abilities in language models are very often an
artifact of the metric rather than a fact about the model (Schaeffer, Miranda
& Koyejo): a discontinuous score turns a smooth underlying curve into a cliff.
The defence is procedural, and it has three parts, all implemented here.

**1. Fit the smooth law first.** A power law in log-log space is the null. It
gets fitted before anything else and it keeps the benefit of the doubt.

**2. Make the break win a comparison that accounts for the search.** A broken
power law has more parameters *and* its breakpoint is chosen by looking at the
data, so a textbook nested F-test is optimistic — it prices the extra
parameters but not the searching. :func:`compare_power_laws` therefore also
runs a parametric bootstrap: simulate data from the fitted smooth law, refit
*both* models including the full breakpoint search, and ask how often a break
looks this good by chance. That p-value is the one to quote.

**3. Check the break survives a change of metric.** :func:`metric_robustness`
refits on a monotone transform of the same measurements. A real transition
survives; an artifact of where a threshold happened to fall does not.

And separately, :func:`collapse_test` answers the question that is usually more
interesting than any threshold: *which axis* do the curves collapse onto —
compute, parameters, or depth? Whichever gives the tightest single curve is
telling you what the phenomenon is made of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PowerLawFit:
    """``y = 10^intercept * x^slope`` fitted in log-log space."""

    slope: float
    intercept: float
    rss: float
    n: int
    k: int = 2

    @property
    def r2(self) -> float:
        return self._r2

    _r2: float = field(default=float("nan"), repr=False)

    @property
    def aic(self) -> float:
        return self.n * np.log(max(self.rss / self.n, 1e-300)) + 2 * self.k

    @property
    def bic(self) -> float:
        return self.n * np.log(max(self.rss / self.n, 1e-300)) + self.k * np.log(self.n)

    def predict(self, x) -> np.ndarray:
        return 10 ** (self.intercept + self.slope * np.log10(np.asarray(x, float)))


@dataclass
class BrokenPowerLawFit:
    """Continuous two-segment power law with a fitted breakpoint."""

    breakpoint: float
    slope_lo: float
    slope_hi: float
    intercept: float
    rss: float
    n: int
    k: int = 4
    _r2: float = field(default=float("nan"), repr=False)

    @property
    def r2(self) -> float:
        return self._r2

    @property
    def aic(self) -> float:
        return self.n * np.log(max(self.rss / self.n, 1e-300)) + 2 * self.k

    @property
    def bic(self) -> float:
        return self.n * np.log(max(self.rss / self.n, 1e-300)) + self.k * np.log(self.n)

    def predict(self, x) -> np.ndarray:
        lx = np.log10(np.asarray(x, float))
        b = np.log10(self.breakpoint)
        ly = self.intercept + self.slope_lo * (lx - b)
        hi = lx > b
        ly[hi] = self.intercept + self.slope_hi * (lx[hi] - b)
        return 10**ly


def _clean(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return np.log10(x[ok]), np.log10(y[ok])


def fit_power_law(x, y) -> PowerLawFit:
    """Least squares in log-log space. The null model."""
    lx, ly = _clean(x, y)
    if len(lx) < 3:
        raise ValueError(f"need >=3 positive finite points, got {len(lx)}")
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (intercept + slope * lx)
    rss = float(resid @ resid)
    tss = float(((ly - ly.mean()) ** 2).sum())
    fit = PowerLawFit(float(slope), float(intercept), rss, len(lx))
    fit._r2 = 1 - rss / tss if tss > 0 else float("nan")
    return fit


def _fit_broken_loglog(lx, ly, n_grid: int = 200):
    """Continuous piecewise-linear fit in log space, breakpoint by grid search.

    Continuity is enforced by construction (both segments meet at the break),
    so the model has 4 parameters, not 6.
    """
    lo, hi = np.percentile(lx, 15), np.percentile(lx, 85)
    if not np.isfinite(lo) or hi <= lo:
        return None
    best = None
    for b in np.linspace(lo, hi, n_grid):
        left = np.minimum(lx - b, 0.0)
        right = np.maximum(lx - b, 0.0)
        A = np.column_stack([np.ones_like(lx), left, right])
        if np.linalg.matrix_rank(A) < 3:
            continue
        coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
        rss = float(((ly - A @ coef) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, b, coef)
    return best


def fit_broken_power_law(x, y, n_grid: int = 200) -> BrokenPowerLawFit:
    """Two power laws joined at a fitted breakpoint."""
    lx, ly = _clean(x, y)
    if len(lx) < 5:
        raise ValueError(f"need >=5 points for a broken fit, got {len(lx)}")
    best = _fit_broken_loglog(lx, ly, n_grid)
    if best is None:
        raise ValueError("breakpoint grid degenerate")
    rss, b, coef = best
    tss = float(((ly - ly.mean()) ** 2).sum())
    fit = BrokenPowerLawFit(
        breakpoint=float(10**b), slope_lo=float(coef[1]), slope_hi=float(coef[2]),
        intercept=float(coef[0]), rss=rss, n=len(lx),
    )
    fit._r2 = 1 - rss / tss if tss > 0 else float("nan")
    return fit


def compare_power_laws(
    x, y, *, n_boot: int = 500, n_grid: int = 200, seed: int = 0
) -> dict:
    """Does a break beat a smooth power law by more than chance?

    Reports three things, in increasing order of how much you should trust them:

    * ``f_p_naive`` — the textbook nested F-test. **Optimistic**, because the
      breakpoint was chosen by looking at the data and the test does not price
      that search.
    * ``delta_bic`` — smooth minus broken. Positive favours the break;
      conventionally >10 is strong.
    * ``p_bootstrap`` — simulate from the fitted *smooth* law with resampled
      residuals, refit both models including the full breakpoint search, and
      count how often the break improves by as much as observed. This is the
      number to quote, because it accounts for the search.

    Also returns a bootstrap confidence interval on the breakpoint itself. A
    break whose location is not pinned down is not a phase transition you can
    report.
    """
    lx, ly = _clean(x, y)
    smooth = fit_power_law(x, y)
    try:
        broken = fit_broken_power_law(x, y, n_grid)
    except ValueError as exc:
        return {"error": str(exc), "smooth": smooth}

    df1, df2 = broken.k - smooth.k, len(lx) - broken.k
    if df2 > 0 and broken.rss > 0:
        f_stat = ((smooth.rss - broken.rss) / df1) / (broken.rss / df2)
        from scipy import stats

        f_p = float(1 - stats.f.cdf(f_stat, df1, df2))
    else:
        f_stat, f_p = float("nan"), float("nan")

    # Parametric bootstrap under the smooth model, with the search included.
    rng = np.random.default_rng(seed)
    resid = ly - (smooth.intercept + smooth.slope * lx)
    observed_gain = smooth.rss - broken.rss
    n_better = 0
    n_ok = 0
    for _ in range(n_boot):
        sim = smooth.intercept + smooth.slope * lx + rng.choice(
            resid, size=len(lx), replace=True
        )
        s_slope, s_int = np.polyfit(lx, sim, 1)
        s_rss = float(((sim - (s_int + s_slope * lx)) ** 2).sum())
        b = _fit_broken_loglog(lx, sim, n_grid)
        if b is None:
            continue
        n_ok += 1
        n_better += (s_rss - b[0]) >= observed_gain
    p_boot = (n_better + 1) / (n_ok + 1) if n_ok else float("nan")

    # Non-parametric CI on the breakpoint.
    breaks = []
    idx = np.arange(len(lx))
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(lx[take])) < 5:
            continue
        b = _fit_broken_loglog(lx[take], ly[take], n_grid)
        if b is not None:
            breaks.append(10 ** b[1])

    return {
        "smooth": smooth,
        "broken": broken,
        "delta_aic": smooth.aic - broken.aic,
        "delta_bic": smooth.bic - broken.bic,
        "f_stat": float(f_stat),
        "f_p_naive": f_p,
        "p_bootstrap": float(p_boot),
        "breakpoint": broken.breakpoint,
        "breakpoint_ci95": (
            [float(np.percentile(breaks, 2.5)), float(np.percentile(breaks, 97.5))]
            if len(breaks) > 20 else None
        ),
        "verdict": (
            "break justified" if (p_boot < 0.05 and smooth.bic - broken.bic > 10)
            else "smooth power law not rejected"
        ),
    }


def metric_robustness(x, y, transforms: dict | None = None, **kw) -> dict:
    """Refit under monotone transforms of ``y``.

    A real transition survives a change of metric; one manufactured by where a
    threshold happened to land does not. Transforms must be monotone, or you
    are testing a different quantity rather than the same one differently
    expressed.
    """
    transforms = transforms or {
        "identity": lambda v: v,
        "sqrt": np.sqrt,
        "log1p": np.log1p,
        "rank": lambda v: np.argsort(np.argsort(v)) + 1.0,
    }
    out = {}
    for name, fn in transforms.items():
        try:
            t = np.asarray(fn(np.asarray(y, float)), float)
            res = compare_power_laws(x, t, **kw)
            out[name] = {
                "verdict": res.get("verdict"),
                "p_bootstrap": res.get("p_bootstrap"),
                "breakpoint": res.get("breakpoint"),
            }
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    survived = sum(1 for v in out.values() if v.get("verdict") == "break justified")
    out["_survived"] = f"{survived}/{len(transforms)} transforms"
    return out


def collapse_test(y, axes: dict) -> dict:
    """Which axis do the measurements collapse onto?

    Usually more interesting than any threshold. Fit a power law of ``y``
    against each candidate axis and report how tight the collapse is. Compute
    winning means the workspace is bought with training; depth winning means
    it is structural.

    Args:
        y: The measured quantity.
        axes: ``{name: values}`` — e.g. compute (6ND), parameters, tokens, depth.

    Returns:
        ``{name: {r2, residual_rms, slope}}`` plus ``best``.
    """
    out = {}
    for name, x in axes.items():
        try:
            fit = fit_power_law(x, y)
            lx, ly = _clean(x, y)
            resid = ly - (fit.intercept + fit.slope * lx)
            out[name] = {
                "r2": fit.r2,
                "residual_rms": float(np.sqrt((resid**2).mean())),
                "slope": fit.slope,
                "n": fit.n,
            }
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    scored = {k: v for k, v in out.items() if "residual_rms" in v}
    if scored:
        out["best"] = min(scored, key=lambda k: scored[k]["residual_rms"])
    return out
