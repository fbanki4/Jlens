"""Keeping the instrument out of the result.

This is the confound that will quietly ruin a scale sweep. The J-lens is
*fitted per model*. So when a bigger model shows a bigger apparent workspace,
there are two explanations and they look identical in the output:

1. the model has a bigger workspace, or
2. the lens has more parameters, more estimation noise, and more room to
   manufacture structure.

Nothing downstream can separate those. They have to be separated here, and
there are three distinct leaks to plug.

**Budget.** ``J_l`` is a ``d x d`` matrix and every entry is an average over
``n_prompts x n_valid_positions`` samples. Fixing the prompt count and the
sequence length therefore fixes *samples per entry*, which is the thing that
controls estimation noise. :class:`FittingBudget` holds those knobs together
and hashes them, so two runs with different budgets cannot be silently pooled
into one curve.

**Shape.** A 6-layer model probed at every layer gives a 6x6 CKA matrix; a
36-layer model gives 36x36. Blockiness, band width and effective dimension all
move with matrix size, so comparing them across depths compares depths.
:func:`matched_depth_layers` probes a *fixed number* of layers at matched
relative depths instead.

**Ambient dimension.** Participation ratio and friends have a ``d``-dependent
floor: a purely random dictionary in 2048 dimensions spreads over more
directions than a purely random one in 512, and reports a bigger "workspace"
for it. Two independent fixes, because they fail differently:
:func:`capacity_controlled_geometry` divides through by a matched null (so the
number becomes "how much more concentrated than unstructured"), and also
random-projects every model to one reference dimension before measuring (so
every model is measured on the same ceiling).

**And then the calibration curve.** Even with all of that, a lens fitted to a
step-0 checkpoint may simply have nothing to transport — ``dh_final/dh_l`` is
near-degenerate when the model is near a unigram predictor. A lens-derived
"no workspace here" and "the instrument was dead here" are the same
measurement. :func:`lens_fidelity` measures the instrument directly, against
the logit lens as a floor, so instrument quality becomes an explicit covariate
you can plot the result against — or regress out with
:func:`jlens.scaling.partial_correlation` — rather than an unstated assumption.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from jlens.jspace import participation_ratio, random_control_dictionary
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

# ----------------------------------------------------------------------------
# Matched shape
# ----------------------------------------------------------------------------


def matched_depth_layers(
    n_layers: int, n_probe: int, *, target_layer: int | None = None
) -> tuple[list[int], np.ndarray]:
    """Probe layers at matched relative depths, ``n_probe`` of them.

    Every layer-resolved statistic in this repo (CKA blockiness, band width,
    onset depth) is computed on a matrix whose size is the number of probed
    layers. Probing "all layers" therefore hands a 36-layer model a different
    measurement than a 6-layer one, and the difference between them is partly
    just matrix size.

    Args:
        n_probe: How many layers to probe. Set this from the *shallowest* model
            in the sweep — a 6-layer model has only 5 legal source layers, so
            asking for 8 silently returns 5 and breaks the matching.

    Returns:
        ``(layers, rho)``. ``len(layers) < n_probe`` means the model was too
        shallow to honour the request; check it before pooling models.
    """
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 1:
        raise ValueError(f"need target_layer >= 1, got {target}")
    if n_probe < 1:
        raise ValueError(f"n_probe must be >= 1, got {n_probe}")
    # Sources must sit strictly below the target, so the legal range is
    # [0, target - 1] and the deepest probe lands just under the readout.
    raw = np.linspace(0.0, float(target - 1), n_probe)
    layers = sorted({int(round(v)) for v in raw})
    return layers, np.array([l / (n_layers - 1) for l in layers], dtype=float)


@dataclass(frozen=True)
class FittingBudget:
    """Everything that must be held identical across a scale sweep.

    The fields are the ones that change how much evidence goes into each
    Jacobian entry, or how big the resulting measurement matrices are. Pass one
    of these to every rung of the sweep and record :meth:`fingerprint` in every
    output row.

    Note ``n_probe_layers`` should come from the *shallowest* model you intend
    to include, not the deepest.
    """

    n_prompts: int = 16
    max_seq_len: int = 128
    skip_first: int = 16
    n_probe_layers: int = 5
    n_probe_tokens: int = 512
    dim_batch: int = 8

    @property
    def samples_per_entry(self) -> int:
        """Gradient samples averaged into each ``J_l`` entry.

        This — not the prompt count alone — is what fixes estimation noise, and
        it is why ``max_seq_len`` belongs in the budget.
        """
        return self.n_prompts * max(self.max_seq_len - self.skip_first - 1, 0)

    def fingerprint(self) -> str:
        """Short hash of the budget.

        Recorded per row so a sweep re-run with different settings cannot be
        appended to an existing curve without the mismatch being visible.
        """
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def describe(self) -> dict:
        return {
            **asdict(self),
            "samples_per_entry": self.samples_per_entry,
            "fingerprint": self.fingerprint(),
        }


# ----------------------------------------------------------------------------
# Matched ambient dimension
# ----------------------------------------------------------------------------


def transport_null(
    P: torch.Tensor, J: torch.Tensor, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """``P @ J_random`` with ``J_random`` matched to ``J``'s Frobenius norm.

    The null that answers the question actually being asked. ``P`` already has
    structure — it is the unembedding — so a dictionary null that scrambles
    everything (:func:`~jlens.jspace.random_control_dictionary`) tests whether
    ``V`` has *any* structure, which it trivially does. This one holds ``P``
    fixed and randomises only the learned transport, so the comparison isolates
    what fitting ``J`` bought over an arbitrary linear map of the same size.

    Note the result depends on ``J`` only through its Frobenius norm, and the
    statistics computed on it are scale-invariant. So with a fixed seed this
    null is *the same for every layer of a model* — which is what you want from
    a normaliser: a single per-model baseline rather than a moving target.
    """
    d = J.shape[0]
    R = torch.randn(d, d, generator=generator)
    R = R * (J.float().norm() / R.norm().clamp_min(1e-12))
    return (P.float() @ R).contiguous()


def _project(V: torch.Tensor, d_ref: int, generator: torch.Generator) -> torch.Tensor:
    """Seeded Johnson-Lindenstrauss projection to a fixed reference width."""
    d = V.shape[1]
    if d <= d_ref:
        return V
    R = torch.randn(d, d_ref, generator=generator) / np.sqrt(d_ref)
    return V.float() @ R


def capacity_controlled_geometry(
    V: torch.Tensor,
    P: torch.Tensor | None = None,
    J: torch.Tensor | None = None,
    *,
    d_ref: int = 256,
    n_controls: int = 3,
    seed: int = 0,
) -> dict:
    """Dictionary geometry with the width confound removed, two ways.

    Args:
        V: ``[n_tokens, d_model]`` lens vectors.
        P, J: Pass both to enable the transport null, which is the informative
            one. Without them only the dictionary null is computed.
        d_ref: Reference width every model is projected to before measuring.
            Must be at or below the *narrowest* model in the sweep, or it does
            nothing on the narrow end and the widths stay unmatched. It reduces
            the width dependence rather than abolishing it — a wider model
            carries more noise directions, and projection cannot unmix noise
            that is already in the dictionary — so use it alongside the null
            ratios, not instead of them.

    Returns:
        Raw statistics, their null values, and the ratios. Read the ratios, and
        note that the two nulls point in opposite directions by construction.
        The dictionary null spreads over everything it can, so
        ``pr_vs_dictionary_null`` is well below 1 and *lower* means more
        concentrated. The transport null is roughly the probe's own geometry
        rotated, so ``pr_vs_transport_null`` sits near 1 and *above* 1 means the
        fitted transport spread the dictionary wider than an arbitrary linear
        map would have.
    """
    gen = torch.Generator().manual_seed(seed)
    V = V.float()
    out: dict = {
        "n_tokens": int(V.shape[0]),
        "d_model": int(V.shape[1]),
        "participation_ratio": participation_ratio(V),
    }

    dict_nulls = [
        participation_ratio(random_control_dictionary(V, generator=gen))
        for _ in range(n_controls)
    ]
    out["pr_dictionary_null"] = float(np.mean(dict_nulls))
    out["pr_vs_dictionary_null"] = out["participation_ratio"] / max(
        out["pr_dictionary_null"], 1e-12
    )

    if P is not None and J is not None:
        t_nulls = [
            participation_ratio(transport_null(P, J, generator=gen))
            for _ in range(n_controls)
        ]
        out["pr_transport_null"] = float(np.mean(t_nulls))
        out["pr_vs_transport_null"] = out["participation_ratio"] / max(
            out["pr_transport_null"], 1e-12
        )

    # Same measurements again, with every model forced into one width.
    pgen = torch.Generator().manual_seed(seed + 1)
    Vp = _project(V, d_ref, pgen)
    out["d_ref"] = int(min(d_ref, V.shape[1]))
    out["projected_participation_ratio"] = participation_ratio(Vp)
    pnull = [
        participation_ratio(random_control_dictionary(Vp, generator=gen))
        for _ in range(n_controls)
    ]
    out["projected_pr_null"] = float(np.mean(pnull))
    out["projected_pr_ratio"] = out["projected_participation_ratio"] / max(
        out["projected_pr_null"], 1e-12
    )
    return out


# ----------------------------------------------------------------------------
# Calibrating the instrument
# ----------------------------------------------------------------------------


@dataclass
class LensFidelity:
    """How much of the model's own output the lens actually reconstructs.

    Attributes:
        layers: Source layers measured.
        rho: Relative depth of each.
        jlens_r: Correlation between lens logits and the model's real final
            logits, per layer. Centred first, because softmax is invariant to
            a constant shift and an uncentred correlation mostly measures the
            shared mean.
        logit_r: Same for the logit lens (``J = I``) — the floor.
        jlens_top1 / logit_top1: Argmax agreement. Secondary, and thresholded,
            so it can manufacture a transition where the continuous number
            shows a ramp. Reported for comparison, not for conclusions.
    """

    layers: list[int]
    rho: np.ndarray
    jlens_r: np.ndarray
    logit_r: np.ndarray
    jlens_top1: np.ndarray
    logit_top1: np.ndarray
    n_prompts: int = 0

    @property
    def best_r(self) -> float:
        """Best fidelity over layers: is the instrument alive anywhere?"""
        return float(np.nanmax(self.jlens_r)) if len(self.jlens_r) else float("nan")

    @property
    def advantage(self) -> float:
        """Largest margin of the J-lens over the logit lens.

        The number to treat as instrument quality. At or below zero means the
        fitted transport bought nothing, and no lens-derived claim from this
        checkpoint should be reported without that caveat attached.
        """
        if not len(self.jlens_r):
            return float("nan")
        return float(np.nanmax(self.jlens_r - self.logit_r))

    def summary(self) -> dict:
        return {
            "layers": self.layers,
            "rho": self.rho.tolist(),
            "jlens_r": self.jlens_r.tolist(),
            "logit_r": self.logit_r.tolist(),
            "jlens_top1": self.jlens_top1.tolist(),
            "logit_top1": self.logit_top1.tolist(),
            "best_r": self.best_r,
            "advantage": self.advantage,
            "n_prompts": self.n_prompts,
        }


def _centered_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson r over the vocabulary axis, after centring both sides."""
    x = (a.float() - a.float().mean()).double()
    y = (b.float() - b.float().mean()).double()
    denom = (x.norm() * y.norm()).clamp_min(1e-30)
    return float((x @ y) / denom)


@torch.no_grad()
def lens_fidelity(
    model: LensModel,
    lens: JacobianLens,
    prompts: Sequence[str],
    *,
    layers: Sequence[int] | None = None,
    position: int = -1,
    max_seq_len: int = 128,
) -> LensFidelity:
    """Measure the instrument, with the logit lens as the floor.

    A J-lens fitted on an early checkpoint can be degenerate: ``dh_final/dh_l``
    is close to trivial when the model is barely more than a unigram predictor.
    Any lens-derived sweep will then show a transition at the point where the
    *lens* starts working, which is not the claim anyone wants to make. Running
    this alongside the sweep turns that into something you can see and control
    for.

    Args:
        position: Where to compare. The last token is the natural choice — it
            is where the model is actually about to predict.

    Returns:
        A :class:`LensFidelity`.
    """
    layers = list(layers if layers is not None else lens.source_layers)
    jr = {l: [] for l in layers}
    lr = {l: [] for l in layers}
    jt = {l: [] for l in layers}
    lt = {l: [] for l in layers}

    for prompt in prompts:
        jl, true_logits, _ = lens.apply(
            model, prompt, layers=layers, positions=[position],
            max_seq_len=max_seq_len,
        )
        ll, _, _ = lens.apply(
            model, prompt, layers=layers, positions=[position],
            max_seq_len=max_seq_len, use_jacobian=False,
        )
        truth = true_logits[0]
        gold = int(truth.argmax())
        for l in layers:
            jr[l].append(_centered_corr(jl[l][0], truth))
            lr[l].append(_centered_corr(ll[l][0], truth))
            jt[l].append(float(int(jl[l][0].argmax()) == gold))
            lt[l].append(float(int(ll[l][0].argmax()) == gold))

    def col(d):
        return np.array([np.mean(d[l]) if d[l] else np.nan for l in layers])

    return LensFidelity(
        layers=layers,
        rho=np.array([l / (model.n_layers - 1) for l in layers], dtype=float),
        jlens_r=col(jr), logit_r=col(lr),
        jlens_top1=col(jt), logit_top1=col(lt),
        n_prompts=len(prompts),
    )


# ----------------------------------------------------------------------------
# Putting it together
# ----------------------------------------------------------------------------


@dataclass
class LensReport:
    """One rung of a sweep, measured under a declared budget."""

    budget: dict
    layers: list[int]
    rho: list[float]
    fidelity: dict
    geometry: list[dict] = field(default_factory=list)
    blocks: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "budget": self.budget, "layers": self.layers, "rho": self.rho,
            "fidelity": self.fidelity, "geometry": self.geometry,
            "blocks": self.blocks, "warnings": self.warnings,
        }


def lens_stage(
    model: LensModel,
    lens: JacobianLens,
    prompts: Sequence[str],
    budget: FittingBudget,
    *,
    probe_token_ids: np.ndarray | torch.Tensor | None = None,
    d_ref: int = 256,
    seed: int = 0,
) -> LensReport:
    """Every lens-derived measurement for one model, capacity-controlled.

    Runs the instrument calibration first and *records a warning on the report*
    when the lens has no advantage over the logit lens, so a degenerate rung
    carries its own caveat into the output file instead of relying on whoever
    reads the plot to remember.

    Returns:
        A :class:`LensReport`.
    """
    from jlens.cka import (
        build_probe,
        cka_matrix,
        lens_vectors,
        readout_kurtosis,
        segment_blocks,
    )
    from jlens.hooks import ActivationRecorder

    layers = list(lens.source_layers)
    P, _ = build_probe(model, probe_token_ids, n_ids=budget.n_probe_tokens)

    warnings: list[str] = []
    if len(layers) != budget.n_probe_layers:
        # Either direction breaks the comparison. Too few means the model was
        # too shallow to honour the budget; too many means the lens was fitted
        # outside it, and every layer-resolved statistic below is computed on a
        # differently sized matrix than the other rungs'.
        warnings.append(
            f"lens has {len(layers)} source layers, budget declares "
            f"{budget.n_probe_layers}: layer-resolved statistics are not "
            f"shape-matched to the rest of the sweep"
        )

    fid = lens_fidelity(model, lens, prompts, layers=layers,
                        max_seq_len=budget.max_seq_len)
    if not np.isfinite(fid.advantage) or fid.advantage <= 0:
        warnings.append(
            f"lens advantage over the logit lens is {fid.advantage:.3f}: the "
            f"fitted transport bought nothing here, so every geometry number "
            f"below is uninterpretable"
        )

    # One forward pass supplies the activations for the kurtosis readout.
    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(model.encode(prompts[0], max_length=budget.max_seq_len))
        acts = {l: rec.activations[l][0, -1].detach().float().cpu() for l in layers}

    geometry = []
    for i, layer in enumerate(layers):
        J = lens.jacobians[layer]
        V = lens_vectors(P, J)
        row = capacity_controlled_geometry(V, P, J, d_ref=d_ref, seed=seed)
        row.update({
            "layer": layer,
            "rho": layer / (model.n_layers - 1),
            "readout_kurtosis": readout_kurtosis(V, acts[layer]),
            "kurtosis_transport_null": readout_kurtosis(
                transport_null(P, J, generator=torch.Generator().manual_seed(seed + i)),
                acts[layer],
            ),
        })
        geometry.append(row)

    blocks: dict = {}
    if len(layers) >= 3:
        C = cka_matrix(lens, P)
        result = segment_blocks(C, layers, n_layers=model.n_layers)
        blocks = {
            "n_blocks": result.n_blocks,
            "blockiness": result.blockiness,
            "boundaries_rho": result.boundaries_rho,
            "layer_coupling": result.layer_coupling,
        }

    return LensReport(
        budget=budget.describe(),
        layers=layers,
        rho=[l / (model.n_layers - 1) for l in layers],
        fidelity=fid.summary(),
        geometry=geometry,
        blocks=blocks,
        warnings=warnings,
    )
