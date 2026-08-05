"""Train the toy model, saving log-spaced checkpoints.

Checkpoints are the point: the "when and how does the J-space form" question
needs the lens fitted at many points along a training trajectory, and on a real
model that means a checkpointed pretraining run somebody else paid for. Here we
get the whole trajectory for a few CPU-minutes.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from jlens.toy.model import ToyConfig, ToyTransformer, WordTokenizer, save_toy
from jlens.toy.world import (
    COMPOSED_RELATIONS,
    Chain,
    Split,
    World,
    build_corpus,
    composed_prompt,
    vocabulary,
)


@dataclass
class TrainConfig:
    steps: int = 6000
    batch_size: int = 128
    lr: float = 3e-3
    warmup: int = 200
    weight_decay: float = 0.01
    seq_len: int = 16
    n_checkpoints: int = 16
    seed: int = 0
    log_every: int = 500


def pad_corpus(
    items: list[tuple[str, int]], tokenizer: WordTokenizer, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One sentence per row, right-padded, with a loss mask.

    Deliberately *not* the usual concatenate-and-chop packing: a two-hop chain
    item is ~35 tokens, and packing would routinely slice one across a window
    boundary so the model never sees the premises and the query together. With
    a synthetic corpus there is no reason to pay that cost.

    Each item carries a ``supervise_from`` word index; positions before it are
    masked out of the loss. See :func:`jlens.toy.world.build_corpus` for why
    that matters.

    Returns ``(ids [n, seq_len], loss_mask [n, seq_len])`` where the mask marks
    positions that are valid *targets*.
    """
    pad, eos = tokenizer.token_id("<pad>"), tokenizer.token_id("<eos>")
    rows, masks = [], []
    for text, supervised in items:
        ids = tokenizer.encode(text) + [eos]
        if len(ids) > seq_len:
            continue
        n_real = len(ids)
        if supervised is None:
            mask = [1] * n_real + [0] * (seq_len - n_real)
        else:
            mask = [0] * seq_len
            for i in supervised:
                if i < n_real:
                    mask[i] = 1
        rows.append(ids + [pad] * (seq_len - n_real))
        masks.append(mask)
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
    )


def checkpoint_steps(total: int, n: int) -> list[int]:
    """Log-spaced steps, so the early phase — where the interesting geometry
    changes happen — is sampled densely."""
    raw = torch.logspace(0, math.log10(total), n).round().long().tolist()
    return sorted({min(int(s), total) for s in raw})


@torch.no_grad()
def composed_accuracy(
    model: ToyTransformer,
    tokenizer: WordTokenizer,
    world: World,
    subjects: dict[str, list[str]],
) -> dict[str, float]:
    """Greedy next-token accuracy on composed queries, per relation."""
    model.eval()
    out = {}
    for name in COMPOSED_RELATIONS:
        table = world.composed_map(name)
        subs = subjects[name]
        if not subs:
            out[name] = float("nan")
            continue
        # Prompt lengths vary (some entity names are two words), so no batching.
        hits = 0
        for s in subs:
            ids = torch.tensor(
                tokenizer.encode(composed_prompt(name, s)), dtype=torch.long
            )[None]
            hits += int(model(ids)[0, -1].argmax()) == tokenizer.token_id(table[s])
        out[name] = hits / len(subs)
    return out


@torch.no_grad()
def atomic_accuracy(
    model: ToyTransformer, tokenizer: WordTokenizer, world: World, relation: str
) -> float:
    """Parametric recall of an atomic fact, e.g. ``alice lives in`` -> her city.

    This is the prerequisite for parametric composition: the model has to know
    both single hops cold before chaining them is even possible.
    """
    from jlens.toy.world import ATOMIC_RELATIONS

    verb, _, _ = ATOMIC_RELATIONS[relation]
    table = world.atomic[relation]
    hits = 0
    for subject, obj in table.items():
        ids = torch.tensor(tokenizer.encode(f"{subject} {verb}"), dtype=torch.long)[None]
        hits += int(model(ids)[0, -1].argmax()) == tokenizer.token_id(obj.split()[0])
    return hits / len(table)


@torch.no_grad()
def chain_accuracy(
    model: ToyTransformer, tokenizer: WordTokenizer, chains: list[Chain]
) -> float:
    """Greedy accuracy on in-context two-hop items."""
    model.eval()
    correct = 0
    for c in chains:
        ids = torch.tensor(tokenizer.encode(c.prompt), dtype=torch.long)[None]
        pred = int(model(ids)[0, -1].argmax())
        correct += pred == tokenizer.token_id(c.answer)
    return correct / max(1, len(chains))


def train(
    world: World,
    split: Split,
    *,
    model_cfg: ToyConfig | None = None,
    train_cfg: TrainConfig | None = None,
    out_dir: str = "runs/toy",
    device: str = "cpu",
) -> tuple[ToyTransformer, WordTokenizer, list[dict]]:
    """Train and checkpoint. Returns ``(model, tokenizer, history)``."""
    tcfg = train_cfg or TrainConfig()
    torch.manual_seed(tcfg.seed)
    vocab = vocabulary(world)
    tokenizer = WordTokenizer(vocab)
    cfg = model_cfg or ToyConfig(vocab_size=len(vocab), max_seq_len=tcfg.seq_len)
    cfg.vocab_size, cfg.max_seq_len = len(vocab), tcfg.seq_len

    sentences = build_corpus(world, split, n_sentences=120_000, seed=tcfg.seed)
    data, loss_mask = pad_corpus(sentences, tokenizer, tcfg.seq_len)
    data, loss_mask = data.to(device), loss_mask.to(device)

    model = ToyTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
        betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < tcfg.warmup:
            return tcfg.lr * step / max(1, tcfg.warmup)
        p = (step - tcfg.warmup) / max(1, tcfg.steps - tcfg.warmup)
        return tcfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    os.makedirs(out_dir, exist_ok=True)
    ckpt_at = set(checkpoint_steps(tcfg.steps, tcfg.n_checkpoints))
    history: list[dict] = []
    gen = torch.Generator().manual_seed(tcfg.seed)
    start = time.perf_counter()

    for step in range(1, tcfg.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = torch.randint(0, data.shape[0], (tcfg.batch_size,), generator=gen)
        batch, mask = data[idx], loss_mask[idx]
        logits = model(batch[:, :-1])
        targets = batch[:, 1:].clone()
        targets[~mask[:, 1:]] = -100  # padding contributes no gradient
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=-100
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in ckpt_at or step == tcfg.steps:
            acc_train = composed_accuracy(model, tokenizer, world, split.train)
            acc_held = composed_accuracy(model, tokenizer, world, split.heldout)
            acc_atomic = atomic_accuracy(model, tokenizer, world, "lives_in")
            model.train()
            path = os.path.join(out_dir, f"ckpt_step{step:06d}.pt")
            save_toy(model, tokenizer, path)
            record = {
                "step": step,
                "tokens": step * tcfg.batch_size * tcfg.seq_len,
                "loss": float(loss.detach()),
                "acc_train": acc_train,
                "acc_heldout": acc_held,
                "acc_atomic_lives_in": acc_atomic,
                "path": path,
            }
            history.append(record)
            print(
                f"  step {step:6d}  loss {float(loss.detach()):.4f}  "
                f"atomic {acc_atomic:.2f}  "
                f"2hop train {acc_train['hails_from']:.2f} "
                f"held {acc_held['hails_from']:.2f}  "
                f"({time.perf_counter() - start:.0f}s)",
                flush=True,
            )
        elif tcfg.log_every and step % tcfg.log_every == 0:
            print(f"  step {step:6d}  loss {float(loss.detach()):.4f}", flush=True)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"n_params": n_params, "history": history}, f, indent=2)
    save_toy(model, tokenizer, os.path.join(out_dir, "final.pt"))
    return model.eval(), tokenizer, history
