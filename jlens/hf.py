"""HuggingFace adapter — run the lens on a real open-weights model.

Model loading stays the caller's job (``from_pretrained``, device map, dtype).
:func:`from_hf` only locates the residual stack, the final norm and the LM head
inside whatever it is handed, and exposes them as a
:class:`~jlens.protocol.LensModel`.

Usage::

    import transformers, jlens
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = jlens.from_hf(hf, tok)

    lens = jlens.fit(model, prompts, checkpoint_path="out/ckpt.pt")
    lens.save("out/lens.pt")

Sizing note: fitting costs one forward and ``ceil(d_model / dim_batch)``
backward passes per prompt, so it scales as ``params * d_model * n_prompts``.
A 1-2B model with 200 prompts is a single-GPU afternoon; a 30B model wants
prompt-sharding across GPUs and :meth:`~jlens.lens.JacobianLens.merge`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _resolve(obj: Any, dotted: str) -> Any:
    return functools.reduce(getattr, dotted.split("."), obj)


@dataclass(frozen=True)
class Layout:
    """Where the lens-relevant submodules live inside a HF causal LM.

    Attributes:
        path: Dotted path from the ``*ForCausalLM`` to the bare text decoder.
        layers: Attribute on the decoder holding the residual blocks.
        norm: Final pre-unembed norm.
        embed: Input token embedding.
        lm_head: Unembedding, on the ``*ForCausalLM``.
    """

    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


#: Tried in order; the first whose ``path`` resolves and whose decoder has all
#: of layers/norm/embed wins.
LAYOUTS: tuple[Layout, ...] = (
    Layout("model"),                                    # Llama/Qwen/Mistral/Gemma/OLMo
    Layout("model.language_model"),                     # multimodal wrappers
    Layout("language_model"),
    Layout("model", norm="final_layernorm"),            # Phi
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),          # GPT-2
    Layout("gpt_neox", norm="final_layer_norm", embed="embed_in",
           lm_head="embed_out"),                                          # Pythia
    Layout("transformer", layers="blocks", norm="ln_f", embed="wte"),
)


def find_layout(hf_model: nn.Module) -> Layout:
    for layout in LAYOUTS:
        try:
            decoder = _resolve(hf_model, layout.path)
        except AttributeError:
            continue
        if all(hasattr(decoder, a) for a in (layout.layers, layout.norm, layout.embed)):
            return layout
    raise ValueError(
        f"could not locate the residual stack in {type(hf_model).__name__} "
        f"(tried {len(LAYOUTS)} layouts); pass layout= explicitly"
    )


class HFLensModel:
    """:class:`~jlens.protocol.LensModel` over a HuggingFace causal LM."""

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        layout: Layout | None = None,
        freeze: bool = True,
    ) -> None:
        self.hf_model = hf_model.eval()
        self.layout = layout or find_layout(hf_model)
        self.decoder = _resolve(hf_model, self.layout.path)
        self.layers = getattr(self.decoder, self.layout.layers)
        self.norm = getattr(self.decoder, self.layout.norm)
        self.head = getattr(hf_model, self.layout.lm_head, None)
        if self.head is None:
            self.head = _resolve(hf_model, self.layout.lm_head)
        self.tokenizer = tokenizer
        self.n_layers = len(self.layers)
        self.d_model = int(
            getattr(self.decoder, self.layout.embed).weight.shape[1]
        )
        if freeze:
            # Fitting only needs gradients w.r.t. activations. Freezing lets
            # ActivationRecorder's start_graph_at make the captured residual
            # the sole graph leaf, which keeps the retained graph small.
            for p in self.hf_model.parameters():
                p.requires_grad_(False)

    def __repr__(self) -> str:
        return (
            f"HFLensModel({type(self.hf_model).__name__}, n_layers={self.n_layers}, "
            f"d_model={self.d_model}, layout={self.layout.path})"
        )

    @property
    def input_device(self) -> torch.device:
        return getattr(self.decoder, self.layout.embed).weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).input_ids
        return ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        # Call the bare decoder, not the *ForCausalLM: the LM head would
        # materialise a [batch, seq, vocab] tensor on every fitting forward
        # pass, which for a large vocabulary dwarfs everything else.
        return self.decoder(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(residual.to(self.head.weight.dtype)))


def from_hf(
    hf_model: nn.Module,
    tokenizer: Any,
    *,
    layout: Layout | None = None,
    freeze: bool = True,
) -> HFLensModel:
    """Wrap an already-loaded HuggingFace model for the lens."""
    return HFLensModel(hf_model, tokenizer, layout=layout, freeze=freeze)
