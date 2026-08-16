"""The J-space: sparse decomposition over J-lens vectors, and interventions.

The J-lens gives one vector per vocabulary token. Since ``n_vocab > d_model``
those vectors span the whole residual stream, so "the J-space" cannot just mean
their span — everything would be in it. What makes it a *subspace-like* object
is a sparsity constraint:

    J-space = { sum of at most k J-lens vectors, non-negative coefficients }

Geometrically that is a union of ``k``-dimensional polyhedral cones, one per
choice of ``k`` vectors — a "sparse subframe". A point's *J-space component* is
its nearest point in that union; the rest is outside the workspace. Empirically
only a couple of dozen vectors are meaningfully active at once, and the J-space
component carries under 10% of activation variance: the workspace is a thin
slice of what the model is actually computing.

This module implements:

* :func:`gradient_pursuit` / :func:`nn_omp` — find the sparse non-negative
  combination (the "bag of concepts" at a position).
* :func:`occupancy` — how many vectors are active before a random control set
  would do just as well.
* :func:`fraction_variance_explained` — how much of the activation is in there.
* :func:`steer`, :func:`ablate`, :func:`swap_coordinates` — writing to the
  workspace rather than reading it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ----------------------------------------------------------------------------
# Sparse non-negative decomposition
# ----------------------------------------------------------------------------


@dataclass
class Decomposition:
    """Result of a sparse non-negative fit of an activation onto J-lens vectors.

    Attributes:
        indices: Selected vector indices, in selection order.
        coefficients: Matching non-negative coefficients.
        residual: ``x - V[indices].T @ coefficients``.
        reconstruction_error: ``||residual|| / ||x||`` after each selection
            step, length ``len(indices) + 1`` (entry 0 is 1.0).
    """

    indices: torch.Tensor
    coefficients: torch.Tensor
    residual: torch.Tensor
    reconstruction_error: torch.Tensor

    def top(self, n: int = 10) -> list[tuple[int, float]]:
        order = self.coefficients.argsort(descending=True)[:n]
        return [(int(self.indices[i]), float(self.coefficients[i])) for i in order]


def _normalized(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norms = V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return V / norms, norms.squeeze(1)


def gradient_pursuit(
    x: torch.Tensor,
    V: torch.Tensor,
    k: int,
    *,
    exclude: torch.Tensor | None = None,
) -> Decomposition:
    """Sparse non-negative approximation of ``x`` by ``k`` rows of ``V``.

    Gradient pursuit (Blumensath & Davies, 2008): like matching pursuit, but
    instead of updating only the newest atom's coefficient, take one exact
    line-search step along the gradient restricted to the selected support.
    Cheaper than the least-squares re-solve of orthogonal matching pursuit and
    close enough in practice — which matters here because the dictionary is the
    whole vocabulary.

    Selection is restricted to *positive* correlations: a J-lens vector encodes
    a disposition to say a token, and "negative presence" is not a thing we
    want to read as a concept being active.

    Args:
        x: ``[d_model]`` activation to decompose.
        V: ``[n_vectors, d_model]`` dictionary (J-lens vectors, unnormalised).
        k: Number of atoms to select.
        exclude: Optional indices never to select.

    Returns:
        A :class:`Decomposition`.
    """
    x = x.float()
    Vn, _ = _normalized(V.float())
    residual = x.clone()
    x_norm = x.norm().clamp_min(1e-12)
    selected: list[int] = []
    coefficients = torch.zeros(0)
    errors = [1.0]
    mask = torch.zeros(V.shape[0], dtype=torch.bool)
    if exclude is not None:
        mask[exclude] = True

    for _ in range(k):
        scores = Vn @ residual
        scores[mask] = -float("inf")
        best = int(scores.argmax())
        if scores[best] <= 0:
            break
        selected.append(best)
        mask[best] = True

        Vs = Vn[selected]                       # [s, d]
        coefficients = torch.cat([coefficients, torch.zeros(1)])
        # One exact line search along the support-restricted gradient.
        g = Vs @ (x - Vs.T @ coefficients)      # [s]
        Vg = Vs.T @ g                           # [d]
        denom = Vg.dot(Vg).clamp_min(1e-12)
        coefficients = (coefficients + (g.dot(g) / denom) * g).clamp_min(0.0)
        residual = x - Vs.T @ coefficients
        errors.append(float(residual.norm() / x_norm))

    return Decomposition(
        indices=torch.tensor(selected, dtype=torch.long),
        coefficients=coefficients,
        residual=residual,
        reconstruction_error=torch.tensor(errors),
    )


def nn_omp(
    x: torch.Tensor, V: torch.Tensor, k: int, *, n_inner: int = 40
) -> Decomposition:
    """Non-negative orthogonal matching pursuit: slower, more accurate.

    Re-solves the non-negative least-squares problem over the whole support at
    every step (by projected gradient descent). Use it when the exact J-space
    component matters — e.g. variance-explained numbers — and
    :func:`gradient_pursuit` when speed matters.
    """
    x = x.float()
    Vn, _ = _normalized(V.float())
    residual = x.clone()
    x_norm = x.norm().clamp_min(1e-12)
    selected: list[int] = []
    coefficients = torch.zeros(0)
    errors = [1.0]
    mask = torch.zeros(V.shape[0], dtype=torch.bool)

    for _ in range(k):
        scores = Vn @ residual
        scores[mask] = -float("inf")
        best = int(scores.argmax())
        if scores[best] <= 0:
            break
        selected.append(best)
        mask[best] = True
        Vs = Vn[selected]
        # Projected gradient descent on ||x - Vs^T c||^2 subject to c >= 0.
        c = torch.cat([coefficients, torch.zeros(1)])
        G = Vs @ Vs.T
        b = Vs @ x
        step = 1.0 / torch.linalg.eigvalsh(G).max().clamp_min(1e-9)
        for _ in range(n_inner):
            c = (c - step * (G @ c - b)).clamp_min(0.0)
        coefficients = c
        residual = x - Vs.T @ coefficients
        errors.append(float(residual.norm() / x_norm))

    return Decomposition(
        indices=torch.tensor(selected, dtype=torch.long),
        coefficients=coefficients,
        residual=residual,
        reconstruction_error=torch.tensor(errors),
    )


# ----------------------------------------------------------------------------
# Capacity
# ----------------------------------------------------------------------------


def random_control_dictionary(
    V: torch.Tensor, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A control dictionary with ``V``'s shape and scale but no structure.

    Gaussian directions matched to ``V``'s per-row norm distribution. Any
    dictionary of enough random directions reconstructs an activation
    eventually; the control is what turns "reconstruction improved" into
    "reconstruction improved *more than chance*".
    """
    R = torch.randn(V.shape, generator=generator)
    R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return R * V.norm(dim=1, keepdim=True)


def occupancy(
    x: torch.Tensor,
    V: torch.Tensor,
    *,
    k_max: int = 64,
    n_controls: int = 3,
    generator: torch.Generator | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """How many J-lens vectors are meaningfully active in ``x``.

    Decompose with the real dictionary and with random controls, then find the
    largest ``k`` at which the real dictionary's *marginal* reconstruction gain
    still beats the control's. Past that point you are fitting noise.

    Returns:
        ``(occupancy, real_errors, control_errors)``; the error curves have
        length ``k_max + 1``.
    """
    real = gradient_pursuit(x, V, k_max).reconstruction_error
    ctrl = torch.stack(
        [
            gradient_pursuit(
                x, random_control_dictionary(V, generator=generator), k_max
            ).reconstruction_error
            for _ in range(n_controls)
        ]
    ).mean(dim=0)
    n = min(len(real), len(ctrl))
    real, ctrl = real[:n], ctrl[:n]
    real_gain, ctrl_gain = -real.diff(), -ctrl.diff()
    better = (real_gain > ctrl_gain).nonzero().flatten()
    return (int(better[-1]) + 1 if len(better) else 0), real, ctrl


def fraction_variance_explained(
    activations: torch.Tensor, V: torch.Tensor, k: int, *, n_controls: int = 1
) -> tuple[float, float]:
    """Share of activation variance the top-``k`` J-space component captures.

    Returns ``(fve, fve_random_control)``. The gap between them is the number
    to quote: the raw FVE of *any* ``k``-atom dictionary is not zero.
    """

    def run(dictionary: torch.Tensor) -> float:
        total = residual = 0.0
        for x in activations:
            d = gradient_pursuit(x, dictionary, k)
            total += float(x.pow(2).sum())
            residual += float(d.residual.pow(2).sum())
        return 1.0 - residual / max(total, 1e-12)

    fve = run(V)
    ctrl = sum(run(random_control_dictionary(V)) for _ in range(n_controls)) / n_controls
    return fve, ctrl


def participation_ratio(V: torch.Tensor) -> float:
    """Effective dimensionality of a dictionary: ``(sum l)^2 / sum l^2`` over
    the eigenvalues of its covariance.

    This is the cheap capacity proxy: it needs only the fitted lens, no forward
    passes, so it can be swept across model scales or training checkpoints for
    free. It measures how many directions the *dictionary* spreads over, which
    is an upper bound on how many independent concepts it could hold.
    """
    Vc = V.float() - V.float().mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.svdvals(Vc).pow(2)
    return float(eigenvalues.sum().pow(2) / eigenvalues.pow(2).sum().clamp_min(1e-30))


# ----------------------------------------------------------------------------
# Writing to the workspace
# ----------------------------------------------------------------------------


def steer(h: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """``h + alpha * v``. Negative ``alpha`` suppresses."""
    return h + alpha * v.to(h.device)


def ablate(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project out the component of ``h`` along ``v`` entirely."""
    v = v.to(h.device)
    unit = v / v.norm().clamp_min(1e-12)
    return h - (h @ unit).unsqueeze(-1) * unit


def ablate_subspace(h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Project out the span of several vectors at once (e.g. the top-``k``
    J-space contents at this position — the "suppress the workspace" ablation).
    """
    V = V.to(h.device).float()
    Q, _ = torch.linalg.qr(V.T)
    return h - (h @ Q) @ Q.T


def swap_coordinates(
    h: torch.Tensor,
    v_source: torch.Tensor,
    v_target: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Exchange two concepts while leaving the rest of the activation alone.

    Reads ``h``'s coordinates in the (non-orthogonal, 2-vector) frame
    ``[v_source, v_target]`` via the pseudoinverse, swaps them, and writes the
    difference back::

        c = V^+ h;   h' = h + V (sigma(c) - c)

    The component of ``h`` orthogonal to ``span{v_source, v_target}`` is
    untouched, which is what makes this a clean test: if the model's answer
    follows the swap, that concept was genuinely the argument being consumed
    downstream, not a correlate.
    """
    V = torch.stack([v_source, v_target], dim=1).to(h.device).float()  # [d, 2]
    c = torch.linalg.pinv(V) @ h.float()
    swapped = torch.stack([c[1], c[0]])
    return h + alpha * (V @ (swapped - c))
