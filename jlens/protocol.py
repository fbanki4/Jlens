"""The model interface the lens is typed against.

The lens never touches a specific model library. It needs five things from a
model: the residual blocks (to hook), their count and width, a tokenizer, a
forward pass that builds an autograd graph through those blocks, and an
unembedding. Implement :class:`LensModel` and everything else here works.

Two adapters ship with this package:

* :func:`jlens.hf.from_hf` — any HuggingFace decoder (Llama, Qwen, Gemma,
  OLMo, GPT-2, Pythia, ...). This is the one to use on a real open model.
* :class:`jlens.toy.model.ToyLensModel` — a from-scratch GPT small enough to
  train and lens on a laptop CPU, used by the experiments in this repo.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn


class LensModel(Protocol):
    """What the Jacobian lens needs from a model.

    Attributes:
        n_layers: Number of residual blocks.
        d_model: Residual-stream width.
        layers: The residual blocks, integer-indexable. These are what
            :class:`~jlens.hooks.ActivationRecorder` hooks; block ``i``'s
            *output* is taken to be the residual stream at layer ``i``.
        tokenizer: Anything with ``decode(ids) -> str``. Only the
            visualisation and phrase code touch it; fitting never does.
    """

    n_layers: int
    d_model: int
    layers: Sequence[nn.Module]
    tokenizer: Any

    def encode(self, text: str, *, max_length: int = ...) -> torch.Tensor:
        """Tokenize to ``input_ids`` of shape ``[1, seq_len]`` on the model's
        input device."""
        ...

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Run the residual stack (no LM head).

        Must build an autograd graph through :attr:`layers` when grad is
        enabled, and must be deterministic across batch elements — the fitting
        estimator replicates one prompt along the batch axis and relies on
        every replica producing the same activations. Put the model in eval
        mode and turn dropout off.
        """
        ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Map ``[..., d_model]`` to logits ``[..., vocab_size]``, applying the
        model's own final norm and LM head."""
        ...
