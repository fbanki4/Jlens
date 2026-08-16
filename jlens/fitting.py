"""Fitting the Jacobian lens.

The lens reads an intermediate residual ``h_l`` by linearly transporting it
into the final-layer basis with an *average* input-output Jacobian, then
decoding with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )        J_l = E[ d h_final / d h_l ]

Two things make this more than a curiosity:

1. **It is causal, not correlational.** ``J_l`` is a derivative of the model's
   own output w.r.t. the activation. A tuned lens fits a linear map to *match*
   the output distribution and, being correlational, tends to skip ahead to the
   answer instead of surfacing the intermediate steps.
2. **It is averaged over contexts.** A single-prompt Jacobian conflates "this
   direction generally disposes the model to say X" with "in this one context
   the model is about to say X". Averaging over a corpus keeps the former. That
   is what makes the readout *verbalizable* content rather than *verbalized*
   content.

The estimator
-------------
For one prompt, inject a one-hot cotangent at output dimension ``i`` **at every
valid target position at once** and backprop. Because attention is causal, the
gradient landing at source position ``p`` is

    sum_{p' >= p}  d h_final[p'] / d h_l[p]

i.e. the effect of ``h_l[p]`` on the present *and every future* position's
final residual. Mean over source positions ``p``, then mean over prompts. One
forward and ``ceil(d_model / dim_batch)`` backwards per prompt, since each
backward carries ``dim_batch`` distinct one-hots on the batch axis.

The collapse over target positions is what makes this cheap — and it is also
why the forward *distance* ``Delta = p' - p`` is integrated out. To resolve it
you need one backward per target position; :func:`delta_jacobian_for_prompt`
does that, and is what an experiment on the temporal reach of the workspace
needs.

Provenance
----------
The estimator, the ``skip_first`` convention and the checkpoint layout follow
Anthropic's reference implementation released with *Verbalizable
Representations Form a Global Workspace in Language Models* (Gurnee et al.,
Transformer Circuits, 2026), Apache-2.0. This is an independent implementation
written to be read; it is deliberately file-format-compatible with that one.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import DeltaJacobianLens, JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Positions before this index are excluded from the average. Early positions
#: act as attention sinks and have wildly atypical residual statistics; the
#: final position has no next-token target.
SKIP_FIRST_N_POSITIONS = 16

#: Default forward-distance bucket edges for :func:`fit_delta`. Bucket ``b``
#: covers ``DELTA_BINS[b] <= Delta < DELTA_BINS[b+1]``, last bucket is open.
DELTA_BINS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64)


def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    """Boolean mask over sequence positions to include in the average.

    Raises:
        ValueError: If ``skip_first`` is negative, or the prompt is too short
            to leave any valid position.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if not mask.any():
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask


def resolve_layers(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Normalise layer indices: fill defaults, wrap negatives, bounds-check,
    and enforce that every source layer sits strictly below the target."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(f"target_layer={target_layer} out of range for {n_layers} layers")
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for {n_layers} layers"
        )
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


# ----------------------------------------------------------------------------
# The collapsed (paper) estimator
# ----------------------------------------------------------------------------


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    self_only: bool = False,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Per-prompt Jacobian estimate ``J_l`` for every requested source layer.

    Args:
        model: Model to differentiate through.
        prompt: Input text.
        source_layers: Layers ``l`` to estimate ``J_l`` at.
        target_layer: Layer to differentiate *from*. Defaults to the last
            block. Targeting the penultimate block often gives a cleaner lens:
            the final block is heavily specialised for calibrating next-token
            probabilities and carries less semantic content.
        dim_batch: Output dimensions per backward pass. The prompt is
            replicated this many times along the batch axis, so this trades
            memory for wall-clock; total backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude.
        self_only: If ``True``, place the cotangent only at the source position
            itself (``p' == p``), giving the "present-token" variant that is
            closest in spirit to the logit lens. Costs one backward per
            *source position*, so it is far more expensive than the default.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)`` with ``jacobians[l]`` a
        ``[d_model, d_model]`` float32 CPU tensor.
    """
    if self_only:
        return _self_jacobian_for_prompt(
            model,
            prompt,
            source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
        )

    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = resolve_layers(source_layers, target_layer, n_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated dim_batch times; the retained
        # graph is reused by every backward below.
        model.forward(input_ids.expand(dim_batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[l] for l in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                positions = valid_positions.to(grad.device, non_blocking=True)
                rows = grad[:n_dims, positions, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    return jacobians, seq_len, n_valid


def _self_jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """``self_only`` variant: cotangent at the source position only."""
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = resolve_layers(source_layers, target_layer, n_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    positions = position_mask.nonzero(as_tuple=True)[0]
    n_valid = len(positions)

    sums = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[l] for l in source_layers]
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for t_idx, t in enumerate(positions.tolist()):
            for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
                n_dims = min(dim_batch, d_model - dim_start)
                cotangent.zero_()
                cotangent[
                    batch_indices[:n_dims], t, dim_start + batch_indices[:n_dims]
                ] = 1.0
                last = t_idx == len(positions) - 1 and pass_idx == n_passes - 1
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=cotangent,
                    retain_graph=not last,
                )
                for layer, grad in zip(source_layers, grads, strict=True):
                    rows = grad[:n_dims, t, :].float()
                    sums[layer][dim_start : dim_start + n_dims, :] += rows.cpu()
                del grads

    return {l: J / n_valid for l, J in sums.items()}, seq_len, n_valid


# ----------------------------------------------------------------------------
# The Delta-resolved estimator
# ----------------------------------------------------------------------------


def _target_positions(
    seq_len: int, valid: torch.Tensor, *, stride: int, log_targets: int | None
) -> list[int]:
    """Pick which target positions to enumerate.

    Deterministic in ``seq_len`` alone, so shards and requeues agree.
    ``log_targets`` keeps that many log-spaced positions (covering every
    ``Delta`` octave at a fraction of the cost); otherwise every ``stride``-th
    valid position is used.
    """
    candidates = valid.nonzero(as_tuple=True)[0].tolist()
    if log_targets is not None and log_targets < len(candidates):
        lo, hi = candidates[0], candidates[-1]
        picks = torch.logspace(
            math.log10(lo + 1), math.log10(hi + 1), log_targets
        ).round().long() - 1
        return sorted({int(min(max(p, lo), hi)) for p in picks.tolist()})
    return candidates[::stride]


def delta_jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    delta_bins: Sequence[int] = DELTA_BINS,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    target_stride: int = 1,
    log_targets: int | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, int, int]:
    """Jacobian estimate resolved by forward distance ``Delta = p' - p``.

    The collapsed estimator sums over target positions inside a single backward
    pass, which destroys the ``Delta`` information irrecoverably. Recovering it
    requires one backward per (dim chunk, *target position*) — roughly
    ``n_targets`` times the cost. Use ``target_stride`` or ``log_targets`` to
    subsample; per-bucket means stay unbiased under subsampling, but summing
    the buckets then reproduces the *subsampled* rather than the full average.

    Returns:
        ``(bucket_sums, bucket_counts, seq_len, n_valid)``. ``bucket_sums[l]``
        has shape ``[n_buckets, d_model, d_model]`` and holds the sum of
        ``dh_final[p'] / dh_l[p]`` over enumerated pairs in each bucket;
        ``bucket_counts`` is the matching ``[n_buckets]`` pair count. Divide to
        get per-pair means.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = resolve_layers(source_layers, target_layer, n_layers)
    edges = torch.tensor(sorted(delta_bins), dtype=torch.long)
    n_buckets = len(edges)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    source_positions = position_mask.nonzero(as_tuple=True)[0]
    n_valid = len(source_positions)
    targets = _target_positions(
        seq_len, position_mask, stride=target_stride, log_targets=log_targets
    )

    bucket_sums = {
        layer: torch.zeros(n_buckets, d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    bucket_counts = torch.zeros(n_buckets, dtype=torch.float64)
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[l] for l in source_layers]
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        # Sources that can reach target t' are exactly those with p <= t'.
        for t_idx, t_prime in enumerate(targets):
            reaching = source_positions[source_positions <= t_prime]
            if len(reaching) == 0:
                continue
            deltas = t_prime - reaching
            bucket_of = torch.bucketize(deltas, edges, right=True) - 1
            bucket_counts += torch.bincount(bucket_of, minlength=n_buckets).double()

            for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
                n_dims = min(dim_batch, d_model - dim_start)
                cotangent.zero_()
                cotangent[
                    batch_indices[:n_dims], t_prime, dim_start + batch_indices[:n_dims]
                ] = 1.0
                last = t_idx == len(targets) - 1 and pass_idx == n_passes - 1
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=cotangent,
                    retain_graph=not last,
                )
                for layer, grad in zip(source_layers, grads, strict=True):
                    rows = grad[:n_dims, reaching.to(grad.device), :].float().cpu()
                    # rows: [n_dims, n_reaching, d_model] -> scatter into buckets
                    for b in range(n_buckets):
                        sel = bucket_of == b
                        if sel.any():
                            bucket_sums[layer][
                                b, dim_start : dim_start + n_dims, :
                            ] += rows[:, sel, :].sum(dim=1)
                del grads

    return bucket_sums, bucket_counts, seq_len, n_valid


# ----------------------------------------------------------------------------
# Corpus-level fitting
# ----------------------------------------------------------------------------


def _atomic_save(obj: object, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    self_only: bool = False,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 25,
    resume: bool = True,
    min_prompts: int = 0,
    stop_at_delta: float | None = None,
    stop_window: int = 10,
    log_every: int = 25,
) -> JacobianLens:
    """Average per-prompt Jacobians over a corpus into a :class:`JacobianLens`.

    Quality saturates fast: the lens beats logit- and tuned-lens baselines with
    as few as ~10 prompts, and improvements past a few hundred are marginal.
    The convergence diagnostic (relative shift in the running mean, which falls
    like ``1/n`` once settled) drives the optional early stop.

    Args:
        prompts: Corpus to average over. A generic pretraining-like
            distribution is the right default — the point is to marginalise
            over contexts, not to model any particular one.
        min_prompts: Never early-stop before this many prompts.
        stop_at_delta: Stop once the mean relative shift over the last
            ``stop_window`` prompts falls below this. ``None`` disables.
        checkpoint_every: Write a resumable checkpoint every N prompts. The
            checkpoint is ``len(source_layers) * d_model**2 * 4`` bytes, so
            raise this for wide models.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = resolve_layers(source_layers, target_layer, n_layers)

    logger.info(
        "fit: n_layers=%d d_model=%d, %d source layers (target=L%d) over %d prompts",
        n_layers, d_model, len(source_layers), target_layer, len(prompts),
    )

    jacobian_sum: dict[int, torch.Tensor]
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"{checkpoint_path} was fitted with {key}={state[key]!r}, "
                    f"not {expected!r}; pass resume=False to discard it"
                )
        jacobian_sum = state["jacobian_sum"]
        n_done, next_idx = state["n_done"], state["next_idx"]
        logger.info("  resuming at prompt %d/%d", next_idx, len(prompts))
    else:
        jacobian_sum = {
            l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers
        }
        n_done = next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    recent_deltas: list[float] = []
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start = time.perf_counter()
        try:
            per_prompt, seq_len, n_valid = jacobian_for_prompt(
                model, prompt, source_layers,
                target_layer=target_layer, dim_batch=dim_batch,
                max_seq_len=max_seq_len, skip_first=skip_first, self_only=self_only,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        prompt_norm = max(per_prompt[l].norm().item() for l in source_layers) / sqrt_d
        if n_done > 0:
            rel_change = max(
                (
                    (per_prompt[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
            recent_deltas.append(rel_change)
            recent_deltas[:] = recent_deltas[-stop_window:]
        else:
            rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        if log_every and (n_done % log_every == 0 or n_done == 1):
            logger.info(
                "  prompt %d/%d seq=%d valid=%d %.1fs  |J|/sqrt(d)=%.3f  d_mean=%.2e",
                prompt_idx + 1, len(prompts), seq_len, n_valid,
                time.perf_counter() - start, prompt_norm, rel_change,
            )
        if checkpoint_every and next_idx % checkpoint_every == 0:
            write_checkpoint()
        if (
            stop_at_delta is not None
            and n_done >= max(min_prompts, stop_window)
            and len(recent_deltas) == stop_window
            and sum(recent_deltas) / stop_window < stop_at_delta
        ):
            logger.info(
                "  early stop at %d prompts (mean d_mean %.2e < %.2e)",
                n_done, sum(recent_deltas) / stop_window, stop_at_delta,
            )
            break

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    logger.info("fit: done, %d prompts", n_done)
    return JacobianLens(
        jacobians={l: jacobian_sum[l] / n_done for l in source_layers},
        n_prompts=n_done,
        d_model=d_model,
    )


def fit_delta(
    model: LensModel,
    prompts: Sequence[str],
    *,
    delta_bins: Sequence[int] = DELTA_BINS,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    target_stride: int = 1,
    log_targets: int | None = None,
    log_every: int = 10,
) -> DeltaJacobianLens:
    """Fit a forward-distance-resolved lens over a corpus.

    Each prompt's bucket sums are normalised by its valid-source count before
    accumulation, so every prompt contributes equally — matching :func:`fit`'s
    weighting, and making ``sum_b`` over buckets reproduce :func:`fit`'s ``J_l``
    when no target subsampling is used.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = resolve_layers(source_layers, target_layer, n_layers)
    edges = tuple(sorted(delta_bins))

    total_sums = {
        l: torch.zeros(len(edges), d_model, d_model, dtype=torch.float32)
        for l in source_layers
    }
    total_counts = torch.zeros(len(edges), dtype=torch.float64)
    n_done = 0
    for prompt_idx, prompt in enumerate(prompts):
        start = time.perf_counter()
        try:
            sums, counts, seq_len, n_valid = delta_jacobian_for_prompt(
                model, prompt, source_layers, delta_bins=edges,
                target_layer=target_layer, dim_batch=dim_batch,
                max_seq_len=max_seq_len, skip_first=skip_first,
                target_stride=target_stride, log_targets=log_targets,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            continue
        for layer in source_layers:
            total_sums[layer] += sums[layer] / n_valid
        total_counts += counts / n_valid
        n_done += 1
        if log_every and n_done % log_every == 0:
            logger.info(
                "  delta prompt %d/%d seq=%d %.1fs",
                prompt_idx + 1, len(prompts), seq_len, time.perf_counter() - start,
            )
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    return DeltaJacobianLens(
        bucket_sums={l: total_sums[l] / n_done for l in source_layers},
        bucket_weights=total_counts / n_done,
        delta_bins=edges,
        n_prompts=n_done,
        d_model=d_model,
    )
