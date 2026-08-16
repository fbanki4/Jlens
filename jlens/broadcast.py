"""Broadcast breadth: how many downstream components actually read a direction.

Global workspace theory's central structural claim is that workspace content is
*broadcast* — many specialised consumers can read it, not just the one circuit
that produced it. The paper measures this as a weight-space property: J-lens
vectors compose with downstream input weights more broadly than other
directions do.

That measure is correlational. A direction can be aligned with a component's
input weights and still have no effect on what that component does, because
alignment ignores whether the direction is ever active and ignores everything
non-linear between here and there. So this module measures it **causally**:
perturb along the direction at layer ``l``, and watch how far the disturbance
actually reaches.

The controls are the whole game
-------------------------------
Any large enough perturbation moves everything downstream, so a raw count of
"components that changed" measures perturbation size, not broadcast. Every
number here is therefore a *ratio* against a random direction of identical
norm injected at the same layer and position. A direction with breadth 1.0
reaches exactly as far as noise of the same size.

Working at block granularity is deliberate: the :class:`~jlens.protocol.LensModel`
contract only guarantees ``layers``, so a block-level profile runs on any model
without architecture-specific plumbing. :func:`submodule_broadcast` goes finer
when you can name the submodules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@dataclass
class BroadcastProfile:
    """How far a perturbation at one layer travels.

    Attributes:
        source_layer: Where the direction was injected.
        downstream: Downstream block indices measured.
        effect: Relative change in each downstream block's output.
        control: Same, for a random direction of matched norm.
        ratio: ``effect / control``. 1.0 means "reaches as far as noise".
    """

    source_layer: int
    downstream: list[int]
    effect: np.ndarray
    control: np.ndarray
    ratio: np.ndarray

    @property
    def breadth(self) -> float:
        """Mean log-ratio over downstream blocks, exponentiated.

        A single continuous number, no threshold. Geometric rather than
        arithmetic mean because the ratios are scale-free and heavy-tailed —
        one block reaching 40x should not swamp eight blocks at 1.0.
        """
        r = self.ratio[np.isfinite(self.ratio) & (self.ratio > 0)]
        return float(np.exp(np.mean(np.log(r)))) if len(r) else float("nan")

    @property
    def reach(self) -> float:
        """How deep the effect persists: the ratio at the *last* downstream
        block, which is what "available to arbitrary later consumers" means."""
        finite = self.ratio[np.isfinite(self.ratio)]
        return float(finite[-1]) if len(finite) else float("nan")

    def summary(self) -> dict:
        return {
            "source_layer": self.source_layer,
            "downstream": self.downstream,
            "breadth": self.breadth,
            "reach": self.reach,
            "ratio": self.ratio.tolist(),
        }


@torch.no_grad()
def _run_capturing(
    model: LensModel,
    input_ids: torch.Tensor,
    at: Sequence[int],
    inject: tuple[int, int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    """Forward pass, optionally adding a vector at (layer, position)."""
    handle = None
    if inject is not None:
        layer, position, vec = inject

        def hook(module, args, output):
            t = output if torch.is_tensor(output) else output[0]
            patched = t.clone()
            patched[0, position] = patched[0, position] + vec.to(t.device, t.dtype)
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        handle = model.layers[layer].register_forward_hook(hook)
    try:
        with ActivationRecorder(model.layers, at=list(at)) as rec:
            model.forward(input_ids)
            return {i: rec.activations[i].detach().float().clone() for i in at}
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def broadcast_profile(
    model: LensModel,
    direction: torch.Tensor,
    source_layer: int,
    prompts: Sequence[str],
    *,
    position: int = -1,
    alpha: float = 1.0,
    n_controls: int = 3,
    max_seq_len: int = 128,
    generator: torch.Generator | None = None,
) -> BroadcastProfile:
    """Causal broadcast profile of ``direction`` injected at ``source_layer``.

    Args:
        direction: ``[d_model]``. Scaled to unit norm then multiplied by
            ``alpha * ||h||`` at the injection site, so the perturbation is a
            fixed fraction of the residual's own magnitude rather than an
            arbitrary absolute size. Without that, comparing directions with
            different natural norms compares perturbation sizes.
        prompts: Averaged over, because broadcast is a property of the
            direction and not of one sentence.
        n_controls: Random directions per prompt for the denominator.

    Returns:
        A :class:`BroadcastProfile`.
    """
    downstream = [l for l in range(source_layer + 1, model.n_layers)]
    if not downstream:
        raise ValueError(f"source_layer={source_layer} has no downstream blocks")
    unit = (direction / direction.norm().clamp_min(1e-12)).float()
    d = unit.shape[0]

    eff_acc = np.zeros(len(downstream))
    ctl_acc = np.zeros(len(downstream))
    n_used = 0
    for prompt in prompts:
        ids = model.encode(prompt, max_length=max_seq_len)
        clean = _run_capturing(model, ids, downstream + [source_layer])
        scale = alpha * float(clean[source_layer][0, position].norm())
        if scale <= 0:
            continue

        pert = _run_capturing(model, ids, downstream,
                              inject=(source_layer, position, scale * unit))
        eff = np.array([
            float((pert[l] - clean[l]).norm() / clean[l].norm().clamp_min(1e-12))
            for l in downstream
        ])

        ctl = np.zeros(len(downstream))
        for _ in range(n_controls):
            r = torch.randn(d, generator=generator)
            r = r / r.norm().clamp_min(1e-12)
            p = _run_capturing(model, ids, downstream,
                               inject=(source_layer, position, scale * r))
            ctl += np.array([
                float((p[l] - clean[l]).norm() / clean[l].norm().clamp_min(1e-12))
                for l in downstream
            ])
        eff_acc += eff
        ctl_acc += ctl / n_controls
        n_used += 1

    if n_used == 0:
        raise ValueError("no usable prompts")
    effect, control = eff_acc / n_used, ctl_acc / n_used
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = effect / np.where(control > 0, control, np.nan)
    return BroadcastProfile(
        source_layer=source_layer, downstream=downstream,
        effect=effect, control=control, ratio=ratio,
    )


@torch.no_grad()
def broadcast_by_layer(
    model: LensModel,
    lens,
    token_ids: Sequence[int],
    prompts: Sequence[str],
    *,
    layers: Sequence[int] | None = None,
    **kwargs,
) -> dict[int, float]:
    """Mean broadcast breadth of a set of J-lens vectors, per source layer.

    This is the curve to compare against depth: global workspace theory
    predicts it should peak in the workspace band rather than rising
    monotonically toward the output.
    """
    from jlens.cka import build_probe

    P, _ = build_probe(model, token_ids=torch.arange(len(model.tokenizer)))
    layers = list(layers if layers is not None else lens.source_layers)
    out = {}
    for layer in layers:
        if layer >= model.n_layers - 1:
            continue
        V = (P @ lens.jacobians[layer]).float()
        breadths = [
            broadcast_profile(model, V[t], layer, prompts, **kwargs).breadth
            for t in token_ids
        ]
        out[layer] = float(np.nanmean(breadths))
    return out


@torch.no_grad()
def submodule_broadcast(
    model: LensModel,
    direction: torch.Tensor,
    source_layer: int,
    named_submodules: Sequence[tuple[str, nn.Module]],
    prompts: Sequence[str],
    *,
    position: int = -1,
    alpha: float = 1.0,
    n_controls: int = 3,
    max_seq_len: int = 128,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Finer-grained version: per named submodule (attention heads, MLPs).

    Pass e.g. ``[(f"L{i}.mlp", blk.mlp) for i, blk in enumerate(model.layers)]``.
    Same ratio-against-matched-noise construction as
    :func:`broadcast_profile`; the return value is ``effect / control`` per
    submodule.
    """
    unit = (direction / direction.norm().clamp_min(1e-12)).float()
    d = unit.shape[0]
    names = [n for n, _ in named_submodules]
    eff = {n: 0.0 for n in names}
    ctl = {n: 0.0 for n in names}
    n_used = 0

    def capture(ids, inject):
        got: dict[str, torch.Tensor] = {}
        handles = []

        def make(name):
            def hook(module, args, output):
                t = output if torch.is_tensor(output) else output[0]
                got[name] = t.detach().float().clone()
            return hook

        for name, mod in named_submodules:
            handles.append(mod.register_forward_hook(make(name)))
        inj = None
        if inject is not None:
            layer, pos, vec = inject

            def ihook(module, args, output):
                t = output if torch.is_tensor(output) else output[0]
                p = t.clone()
                p[0, pos] = p[0, pos] + vec.to(t.device, t.dtype)
                return p if torch.is_tensor(output) else (p, *output[1:])

            inj = model.layers[layer].register_forward_hook(ihook)
        try:
            model.forward(ids)
        finally:
            for h in handles:
                h.remove()
            if inj is not None:
                inj.remove()
        return got

    for prompt in prompts:
        ids = model.encode(prompt, max_length=max_seq_len)
        clean = capture(ids, None)
        with ActivationRecorder(model.layers, at=[source_layer]) as rec:
            model.forward(ids)
            scale = alpha * float(rec.activations[source_layer][0, position].norm())
        if scale <= 0:
            continue

        p = capture(ids, (source_layer, position, scale * unit))
        for n in names:
            if n in clean and n in p:
                eff[n] += float((p[n] - clean[n]).norm()
                                / clean[n].norm().clamp_min(1e-12))
        for _ in range(n_controls):
            r = torch.randn(d, generator=generator)
            r = r / r.norm().clamp_min(1e-12)
            q = capture(ids, (source_layer, position, scale * r))
            for n in names:
                if n in clean and n in q:
                    ctl[n] += float((q[n] - clean[n]).norm()
                                    / clean[n].norm().clamp_min(1e-12)) / n_controls
        n_used += 1

    return {
        n: (eff[n] / ctl[n]) if ctl[n] > 0 else float("nan") for n in names
    } if n_used else {}
