"""Applying a fitted Jacobian lens.

:class:`JacobianLens` holds the per-layer ``J_l`` and the readout. Two views of
the same object are useful and are both exposed here:

* **Readout view.** ``lens_l(h) = unembed(J_l @ h)`` gives a score over the
  whole vocabulary for an activation. Sort it and you get a human-readable
  description of what that activation is disposed to make the model say.
* **Vector view.** The rows of ``W_U @ J_l`` are the *J-lens vectors*: one
  direction in residual-stream space per vocabulary token. Reading out is an
  inner product against these; writing (steering, ablation, concept swaps) is
  arithmetic on them. :meth:`JacobianLens.lens_vectors` builds them.

The vector view is the one that makes "the J-space" a thing: those vectors are
an overcomplete frame, and the J-space is the set of points expressible as a
*sparse non-negative* combination of them. See :mod:`jlens.jspace`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


class JacobianLens:
    """A fitted Jacobian lens.

    Attributes:
        jacobians: ``{layer: Tensor[d_model, d_model]}``; ``J_l`` maps a
            residual at layer ``l`` into the final-layer basis.
        source_layers: Sorted fitted layer indices.
        n_prompts: Prompts averaged over.
        d_model: Residual width.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {int(l): J.float() for l, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)}))"
        )

    # -- persistence ---------------------------------------------------------

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path``. Stored as fp16 by default: entries are O(1) so
        range is not a constraint, and fp16's extra mantissa beats bf16 here."""
        torch.save(
            {
                "J": {l: J.to(dtype) for l, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "JacobianLens":
        """Load a lens saved by :meth:`save`, or an interrupted
        :func:`jlens.fitting.fit` checkpoint (whose running sum is divided by
        its prompt count so partial fits stay usable)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "J" in ckpt:
            return cls(
                jacobians=ckpt["J"],
                n_prompts=int(ckpt["n_prompts"]),
                d_model=int(ckpt["d_model"]),
            )
        if "jacobian_sum" in ckpt:
            n_done = int(ckpt["n_done"])
            if n_done <= 0:
                raise ValueError(f"{path}: fit checkpoint with n_done=0")
            jac = {int(l): J / n_done for l, J in ckpt["jacobian_sum"].items()}
            return cls(
                jacobians=jac,
                n_prompts=n_done,
                d_model=int(next(iter(jac.values())).shape[0]),
            )
        raise ValueError(f"{path} is not a lens file (keys: {sorted(ckpt)})")

    @classmethod
    def merge(cls, lenses: Sequence["JacobianLens"]) -> "JacobianLens":
        """Combine lenses fitted on disjoint prompt slices (prompt-weighted
        mean). This is how you parallelise fitting across machines."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
            ):
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(l.n_prompts for l in lenses)
        merged = {
            layer: sum(l.jacobians[layer] * l.n_prompts for l in lenses) / n_total
            for layer in first.source_layers
        }
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    # -- reading -------------------------------------------------------------

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """``J_l @ h``: map a residual at ``layer`` into the final-layer basis.

        Args:
            residual: ``[..., d_model]``.
        """
        return residual @ self.jacobians[layer].to(residual.device).T

    def lens_vectors(
        self,
        model: LensModel,
        layer: int,
        *,
        token_ids: torch.Tensor | None = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """The J-lens vectors at ``layer``: rows of ``W_U @ J_l``.

        Row ``i`` is the residual-stream direction whose presence disposes the
        model to say token ``i`` — the object you steer, ablate and swap along.

        Args:
            token_ids: Restrict to these vocabulary ids (``None`` = all). The
                full matrix is ``[vocab, d_model]`` and can be large.
            normalize: Return unit-norm rows. Useful when comparing directions;
                *not* what you want when reconstructing an activation.

        Returns:
            ``[n_tokens, d_model]`` float32.
        """
        J = self.jacobians[layer]
        d = J.shape[0]
        eye = torch.eye(d, dtype=torch.float32)
        # unembed() applies the model's own final norm + head. Feeding it the
        # identity gives W_U (with the norm gain folded in) without needing to
        # know where the model keeps its weights.
        with torch.no_grad():
            W = model.unembed(eye.to(_model_device(model))).float().cpu()  # [d, vocab]
        W = W.T  # [vocab, d]
        if token_ids is not None:
            W = W[token_ids]
        V = W @ J
        if normalize:
            V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return V

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ``model`` on ``prompt`` and read out lens logits.

        Args:
            layers: Layers to read at (default: all fitted).
            positions: Token positions (Python indexing; negatives count from
                the end). ``None`` returns every position.
            use_jacobian: ``False`` skips the transport, giving the plain
                logit-lens baseline. Handy for A/B-ing the two.

        Returns:
            ``(lens_logits, model_logits, input_ids)`` where ``lens_logits[l]``
            is ``[n_positions, vocab]``.
        """
        if layers is None:
            layers = self.source_layers
        bad = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if bad:
            raise ValueError(f"layers {bad} out of range for {model.n_layers} layers")
        unknown = set(layers) - set(self.source_layers)
        if use_jacobian and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not fitted; have {self.source_layers}"
            )
        final = model.n_layers - 1
        record_at = sorted(set(layers) | {final})

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as rec:
            model.forward(input_ids)
            acts = {i: rec.activations[i].detach() for i in record_at}

        def select(layer: int) -> torch.Tensor:
            full = acts[layer][0]
            return (full if positions is None else full[list(positions)]).float()

        lens_logits = {}
        for layer in layers:
            h = select(layer)
            if use_jacobian:
                h = self.transport(h, layer)
            lens_logits[layer] = model.unembed(h).float().cpu()
        return lens_logits, model.unembed(select(final)).float().cpu(), input_ids

    @torch.no_grad()
    def readout(
        self,
        model: LensModel,
        prompt: str,
        *,
        layer: int,
        position: int = -1,
        top_k: int = 10,
        max_seq_len: int = 512,
    ) -> list[tuple[str, float]]:
        """Convenience: the top-``k`` lens tokens at one (layer, position)."""
        lens_logits, _, _ = self.apply(
            model, prompt, layers=[layer], positions=[position], max_seq_len=max_seq_len
        )
        scores = lens_logits[layer][0]
        top = scores.topk(top_k)
        return [
            (model.tokenizer.decode([int(i)]), float(v))
            for i, v in zip(top.indices, top.values)
        ]


class DeltaJacobianLens:
    """A Jacobian lens resolved by forward token distance.

    ``bucket_sums[l][b]`` is the mean (over prompts, of the per-prompt
    source-position mean) Jacobian restricted to pairs whose distance falls in
    bucket ``b``. Summing over buckets recovers the ordinary ``J_l``.
    """

    def __init__(
        self,
        bucket_sums: dict[int, torch.Tensor],
        *,
        bucket_weights: torch.Tensor,
        delta_bins: Sequence[int],
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.bucket_sums = {int(l): B.float() for l, B in bucket_sums.items()}
        self.bucket_weights = bucket_weights.double()
        self.delta_bins = tuple(delta_bins)
        self.source_layers = sorted(self.bucket_sums)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def collapsed(self) -> JacobianLens:
        """Sum over distance buckets: the ordinary lens."""
        return JacobianLens(
            jacobians={l: B.sum(dim=0) for l, B in self.bucket_sums.items()},
            n_prompts=self.n_prompts,
            d_model=self.d_model,
        )

    def per_pair_mean(self, layer: int) -> torch.Tensor:
        """``[n_buckets, d, d]``: mean Jacobian per (source, target) pair in
        each bucket, i.e. corrected for how many pairs each bucket contains."""
        w = self.bucket_weights.clamp_min(1e-12).float()
        return self.bucket_sums[layer] / w[:, None, None]

    def effect_curve(self, layer: int, *, per_pair: bool = True) -> torch.Tensor:
        """Frobenius norm of the bucketed Jacobian vs distance: how much
        influence an activation at this layer still exerts ``Delta`` tokens
        later. ``[n_buckets]``."""
        B = self.per_pair_mean(layer) if per_pair else self.bucket_sums[layer]
        return B.flatten(1).norm(dim=1)

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(
            {
                "bucket_sums": {l: B.to(dtype) for l, B in self.bucket_sums.items()},
                "bucket_weights": self.bucket_weights,
                "delta_bins": list(self.delta_bins),
                "n_prompts": self.n_prompts,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DeltaJacobianLens":
        c = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            bucket_sums=c["bucket_sums"],
            bucket_weights=c["bucket_weights"],
            delta_bins=c["delta_bins"],
            n_prompts=int(c["n_prompts"]),
            d_model=int(c["d_model"]),
        )


def _model_device(model: LensModel) -> torch.device:
    for p in model.layers[0].parameters():
        return p.device
    return torch.device("cpu")
