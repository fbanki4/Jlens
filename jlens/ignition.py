"""Ignition: an all-or-none transition, measured without the lens.

Every other measurement in this repo goes *through* ``J_l``, and that is a
problem for any question about training or scale. ``J_l`` is an expectation of
``dh_final/dh_l``. Early in training the model is close to a unigram predictor
and that map is degenerate — not weak, degenerate. So a reading of "no
workspace at step 512" cannot be distinguished from "the instrument had
nothing to measure". Run a lens-based sweep across a training trajectory and
you will find a transition, and it may be the instrument warming up.

This module implements the paper's ambiguity paradigm, which touches the raw
residual stream and never builds a lens:

1. Replace one token's input embedding with a mixture
   ``(1 - a) * e_B + a * e_A`` and sweep ``a`` from 0 to 1.
2. At every layer, record where the activation sits along the line joining
   that trial's own pure-B activation to its pure-A activation:

       share = <h_a - h_0, h_1 - h_0> / ||h_1 - h_0||^2

   which is 0 at one end of the sweep and 1 at the other by construction.
3. In early layers ``share`` tracks ``a`` roughly proportionally. Past the
   workspace onset it sits near one endpoint or the other and switches sharply
   at a threshold — the all-or-none "ignition" global workspace theory
   predicts.

The reported statistic is the **sharpness** of that switch: the maximum slope
of ``share`` against ``a``, in units where proportional tracking is exactly
1.0. It is continuous, it needs no threshold, and it is comparable across
models of different width and depth.

``onset_rho`` is then the relative depth at which sharpness first reaches half
its within-model maximum. Normalising within each model matters: a fixed
cross-model cutoff would manufacture exactly the phase transitions
:mod:`experiments.e_scaling` is trying to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jlens.protocol import LensModel


@dataclass
class IgnitionResult:
    """Per-layer ignition statistics for one model.

    Attributes:
        layers: Layer indices measured (block outputs).
        rho: Relative depth of each layer.
        sharpness: Max slope of share-vs-alpha, averaged over trials. 1.0 is
            proportional tracking; larger is more step-like.
        bimodality: Mean over trials at the most ambiguous alpha of
            ``1 - 2*min(share, 1-share)``. 0 means the activation sits midway
            between the two interpretations; 1 means it has committed to one.
        shares: ``[n_trials, n_alphas, n_layers]`` raw projection shares.
        alphas: The sweep values.
    """

    layers: list[int]
    rho: np.ndarray
    sharpness: np.ndarray
    bimodality: np.ndarray
    shares: np.ndarray
    alphas: np.ndarray

    def onset_rho(self, frac: float = 0.5) -> float:
        """Relative depth where sharpness first reaches ``frac`` of its own max.

        Within-model normalisation is deliberate — see the module docstring.
        """
        s = self.sharpness
        if not np.isfinite(s).any():
            return float("nan")
        lo, hi = np.nanmin(s), np.nanmax(s)
        if hi - lo <= 1e-9:
            return float("nan")
        target = lo + frac * (hi - lo)
        hit = np.where(s >= target)[0]
        return float(self.rho[hit[0]]) if len(hit) else float("nan")

    def summary(self) -> dict:
        return {
            "layers": self.layers,
            "rho": self.rho.tolist(),
            "sharpness": self.sharpness.tolist(),
            "bimodality": self.bimodality.tolist(),
            "max_sharpness": float(np.nanmax(self.sharpness)),
            "onset_rho": self.onset_rho(),
            "n_trials": int(self.shares.shape[0]),
        }


@torch.no_grad()
def _block0_input(model: LensModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Capture what the first residual block receives (embedding + positions).

    Uses a forward pre-hook so no assumption is made about where the model
    keeps its embedding table — only that ``layers`` are the residual blocks,
    which the LensModel protocol already guarantees.
    """
    captured = {}

    def pre_hook(module, args, kwargs=None):
        captured["x"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

    handle = model.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        model.forward(input_ids)
    finally:
        handle.remove()
    return captured["x"]


@torch.no_grad()
def _run_with_injected(
    model: LensModel, input_ids: torch.Tensor, x: torch.Tensor, at: list[int]
) -> dict[int, torch.Tensor]:
    """Run the stack with block 0's input replaced by ``x``; record ``at``."""
    from jlens.hooks import ActivationRecorder

    def pre_hook(module, args, kwargs=None):
        if args:
            return (x,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs or {})
        kwargs["hidden_states"] = x
        return args, kwargs

    handle = model.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with ActivationRecorder(model.layers, at=at) as rec:
            model.forward(input_ids)
            return {l: rec.activations[l].detach().clone() for l in at}
    finally:
        handle.remove()


@torch.no_grad()
def ignition_sweep(
    model: LensModel,
    trials: list[tuple[str, str]],
    *,
    concept_position: int = 0,
    measure_position: int | None = None,
    n_alphas: int = 21,
    layers: list[int] | None = None,
) -> IgnitionResult:
    """Measure ignition sharpness by layer.

    Args:
        model: Any :class:`~jlens.protocol.LensModel`.
        trials: ``(prompt_a, prompt_b)`` pairs, identical except for the token
            at ``concept_position``. The two prompts must tokenize to the same
            length — the mixture is only well defined if everything else lines
            up.
        concept_position: Index of the token whose embedding is blended.
        measure_position: Where to read the residual stream. Defaults to
            ``concept_position``, which is what the paper does; the final
            position is the interesting alternative when you care about the
            answer rather than the concept.
        n_alphas: Sweep resolution. More alphas resolve a sharper switch, and
            sharpness is normalised by the step size so the numbers stay
            comparable.

    Returns:
        An :class:`IgnitionResult`.
    """
    if layers is None:
        layers = list(range(model.n_layers))
    measure_position = (
        concept_position if measure_position is None else measure_position
    )
    alphas = np.linspace(0.0, 1.0, n_alphas)

    shares = np.full((len(trials), n_alphas, len(layers)), np.nan)
    for t, (prompt_a, prompt_b) in enumerate(trials):
        ids_a = model.encode(prompt_a)
        ids_b = model.encode(prompt_b)
        if ids_a.shape != ids_b.shape:
            continue  # differing tokenization: the mixture is not defined
        x_a, x_b = _block0_input(model, ids_a), _block0_input(model, ids_b)
        # The two prompts differ at one token and block 0's input is
        # position-local, so blending the whole tensor blends only that token.
        endpoints = {}
        for name, x, ids in (("b", x_b, ids_b), ("a", x_a, ids_a)):
            acts = _run_with_injected(model, ids, x, layers)
            endpoints[name] = {l: acts[l][0, measure_position].float() for l in layers}

        for i, alpha in enumerate(alphas):
            x_mix = (1 - alpha) * x_b + alpha * x_a
            acts = _run_with_injected(model, ids_b, x_mix, layers)
            for j, l in enumerate(layers):
                h = acts[l][0, measure_position].float()
                base = endpoints["b"][l]
                axis = endpoints["a"][l] - base
                denom = float(axis.dot(axis))
                if denom > 1e-12:
                    shares[t, i, j] = float((h - base).dot(axis) / denom)

    # sharpness: max |d share / d alpha|, normalised so proportional == 1.0
    d_alpha = alphas[1] - alphas[0]
    with np.errstate(invalid="ignore"):
        slopes = np.abs(np.diff(shares, axis=1)) / d_alpha
        sharpness = np.nanmean(np.nanmax(slopes, axis=1), axis=0)
        mid = n_alphas // 2
        s_mid = shares[:, mid, :]
        bimodality = np.nanmean(1.0 - 2.0 * np.minimum(s_mid, 1.0 - s_mid), axis=0)

    n_layers_total = model.n_layers
    return IgnitionResult(
        layers=list(layers),
        rho=np.array([l / max(n_layers_total - 1, 1) for l in layers]),
        sharpness=sharpness,
        bimodality=bimodality,
        shares=shares,
        alphas=alphas,
    )
