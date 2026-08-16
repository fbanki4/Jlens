"""Forward hooks for capturing the residual stream."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn


class ActivationRecorder:
    """Capture residual-stream tensors at given block indices.

    Registers a forward hook on each requested block on ``__enter__`` and
    removes them on ``__exit__``. After a forward pass each block's output
    lands in :attr:`activations`, keyed by block index. Tensors are *not*
    detached, so they can be handed straight to :func:`torch.autograd.grad`.

    Args:
        blocks: The residual blocks (e.g. ``model.layers``).
        at: Block indices to record.
        start_graph_at: If given, the tensor captured at this index is marked
            ``requires_grad_(True)`` before downstream blocks consume it. With
            all model parameters frozen (``requires_grad=False``), that makes
            this residual the only graph leaf, so the retained graph spans just
            this block onward instead of the whole network. This is what keeps
            the memory cost of fitting proportional to depth-below-the-shallowest
            -source-layer rather than to the full model.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> None:
            # HF blocks sometimes return (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> "ActivationRecorder":
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
