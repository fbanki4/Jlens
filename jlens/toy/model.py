"""A small from-scratch decoder transformer that implements ``LensModel``.

Deliberately vanilla — pre-norm, RMSNorm, causal attention, GELU MLP, learned
positional embeddings — so that nothing about the J-lens results depends on an
exotic architecture. It exists so the whole pipeline (train -> fit -> read ->
intervene) runs on a laptop CPU in minutes, and so training checkpoints are
available for the "when does the J-space form" question.

The one interface obligation worth stating: ``layers`` must be the residual
blocks, and block ``i``'s output must *be* the residual stream at layer ``i``.
The lens hooks those outputs and differentiates between them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ToyConfig:
    vocab_size: int = 160
    d_model: int = 96
    n_layers: int = 12
    n_heads: int = 4
    d_ff: int | None = None       # default 4 * d_model
    max_seq_len: int = 64
    dropout: float = 0.0

    @property
    def ff(self) -> int:
        return self.d_ff or 4 * self.d_model

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.freeze_qk = False  # set by the frozen-attention control

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        if self.freeze_qk:
            # Stop-gradient through the attention pattern: perturbing x can no
            # longer change *where* a head looks, only *what* it moves. This is
            # the paper's frozen-QK ablation, and it separates "what gets
            # attended to" from "what gets transported".
            q, k = q.detach(), k.detach()
        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """One residual block. Its *output* is the residual stream at this layer."""

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ff, cfg.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ToyTransformer(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def residual_stack(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(pos)[None]
        for block in self.layers:
            x = block(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.residual_stack(input_ids)))


class WordTokenizer:
    """Whitespace tokenizer over a fixed vocabulary."""

    def __init__(self, words: list[str]) -> None:
        self.itos = list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[w] for w in text.split() if w in self.stoi]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)

    def token_id(self, word: str) -> int:
        return self.stoi[word]


class ToyLensModel:
    """:class:`~jlens.protocol.LensModel` adapter for :class:`ToyTransformer`.

    Freezes parameters, since fitting only ever needs gradients w.r.t.
    activations — with everything frozen, ``start_graph_at`` makes the captured
    residual the sole graph leaf and the retained graph stays small.
    """

    def __init__(self, model: ToyTransformer, tokenizer: WordTokenizer) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        self.layers = model.layers

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer.encode(text)[: min(max_length, self.model.cfg.max_seq_len)]
        return torch.tensor(ids, dtype=torch.long)[None]

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self.model.residual_stack(input_ids)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(self.model.norm(residual))

    def set_frozen_attention(self, frozen: bool) -> None:
        """Toggle the frozen-QK control on every block."""
        for block in self.model.layers:
            block.attn.freeze_qk = frozen


def save_toy(model: ToyTransformer, tokenizer: WordTokenizer, path: str) -> None:
    torch.save(
        {
            "config": asdict(model.cfg),
            "state_dict": model.state_dict(),
            "vocab": tokenizer.itos,
        },
        path,
    )


def load_toy(path: str) -> tuple[ToyTransformer, WordTokenizer]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ToyConfig(**ckpt["config"])
    model = ToyTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), WordTokenizer(ckpt["vocab"])
