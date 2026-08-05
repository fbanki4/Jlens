"""Multi-token concepts without training anything.

The J-lens produces one vector per *vocabulary token*, so the concepts it can
name are exactly the concepts the tokenizer happens to spell in one piece.
"blackmail" shows up as ``black``; "photosynthesis" as ``phot``. The two known
fixes both need an auxiliary artifact built first:

* the **template lens** needs an LLM to author ~100 bespoke passages per word,
  a vocabulary enumerated in advance, and a few hundred forward passes per
  word — and, being a fitted linear predictor of the output, it inherits the
  tuned lens's habit of skipping ahead to the answer, plus a set of
  high-frequency junk words that have to be filtered out by hand;
* the **oracle lens** fine-tunes two auxiliary models and then RLs one of them.

Both are trained. This module asks whether that is necessary, and implements
two constructions that are not.

Method A — the phrase Jacobian
------------------------------
The J-lens definition already generalises to phrases. For a phrase
``w = (w_1..w_n)``::

    v_w(l) = E_{prompt, t} [ d log p(w | prompt) / d h_{l,t} ]

Teacher-force ``w`` onto the end of each corpus prompt, sum its token
log-probabilities, and backprop to every source position at once — causality
means one backward pass gives every ``t`` simultaneously, exactly as in the
ordinary J-lens fit. For ``n = 1`` this *is* the J-lens (up to taking the
gradient at the probabilities rather than at the final residual, which is the
same deviation the paper notes for the template lens). Cost is one
forward+backward per (prompt, phrase) — cheaper than the template lens's
hundreds of forwards per word, with no authored data.

Method B — the debiased token sum
---------------------------------
Cheaper still, and it reuses an already-fitted ``J_l`` with no new backward
passes. Chain-rule Method A through the final layer::

    d log p(w)/d h_{l,t} ~= sum_i J_l^T . d log p(w_i)/d h_final
                          = sum_i [ v_{w_i} - E_{p_i}[v] ]

where ``v_t`` is the ordinary J-lens vector for token ``t`` and ``p_i`` is the
model's own predictive distribution at the ``i``-th phrase position. The first
term is the naive token-sum baseline; the second is the correction it is
missing. It subtracts the *generic continuation* — whatever usually follows
``w_1..w_{i-1}`` — leaving only what is specific to this phrase. That is why
``black`` alone is a poor stand-in for ``blackmail``: most of ``black``'s
J-lens vector is the part shared with ``blackboard`` and ``blackbird``, and
that part is exactly what ``E_{p_i}[v]`` removes.

Method C — proposing the dictionary
-----------------------------------
Neither method above says *which* phrases to test. :func:`propose_phrases`
gets candidates from the model itself: decompose the activation over
single-token vectors, take the top tokens, let the model continue each one in
the actual context, and keep the continuations that improve reconstruction.
Propose-and-verify with no oracle to train.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from jlens.hooks import ActivationRecorder
from jlens.jspace import gradient_pursuit
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Baselines (what the J-lens can do today)
# ----------------------------------------------------------------------------


def first_token_vector(V: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """The paper's default fallback: use the phrase's first token's vector."""
    return V[token_ids[0]].clone()


def token_average_vector(V: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Mean of the constituent tokens' vectors. Reported in the paper as *not*
    helping over the first token — kept here as the control it is."""
    return V[list(token_ids)].mean(dim=0)


# ----------------------------------------------------------------------------
# Method A — phrase Jacobian
# ----------------------------------------------------------------------------


def phrase_logprob_gradient(
    model: LensModel,
    context_ids: torch.Tensor,
    phrase_ids: Sequence[int],
    source_layers: Sequence[int],
    *,
    skip_first: int = 0,
) -> dict[int, torch.Tensor]:
    """``d log p(phrase | context) / d h_{l,t}``, averaged over source ``t``.

    One forward over ``[context, phrase]`` and one backward. Because attention
    is causal, the single backward yields the gradient at every context
    position at once.

    Args:
        context_ids: ``[1, T]``.
        phrase_ids: The phrase, teacher-forced after the context.
        skip_first: Leading context positions to exclude from the average.

    Returns:
        ``{layer: [d_model] float32}``.
    """
    T = context_ids.shape[1]
    n = len(phrase_ids)
    full = torch.cat(
        [
            context_ids,
            torch.tensor(list(phrase_ids), dtype=torch.long, device=context_ids.device)[
                None
            ],
        ],
        dim=1,
    )
    with (
        ActivationRecorder(
            model.layers, at=list(source_layers), start_graph_at=min(source_layers)
        ) as rec,
        torch.enable_grad(),
    ):
        residual = model.forward(full)
        logits = model.unembed(residual).float()
        # Position T-1+i predicts phrase token i.
        log_probs = logits.log_softmax(dim=-1)[0]
        score = sum(
            log_probs[T - 1 + i, phrase_ids[i]] for i in range(n)
        )
        sources = [rec.activations[l] for l in source_layers]
        grads = torch.autograd.grad(score, sources)

    out = {}
    for layer, g in zip(source_layers, grads, strict=True):
        rows = g[0, skip_first:T, :].float()
        out[layer] = rows.mean(dim=0).detach().cpu()
    return out


def fit_phrase_vectors(
    model: LensModel,
    phrases: Sequence[str],
    prompts: Sequence[str],
    source_layers: Sequence[int],
    *,
    max_seq_len: int = 128,
    skip_first: int = 4,
    log_every: int = 0,
) -> dict[str, dict[int, torch.Tensor]]:
    """Method A over a corpus: ``{phrase: {layer: vector}}``.

    The averaging over ``prompts`` is the same move that makes the ordinary
    J-lens read *verbalizable* rather than *verbalized* content: a single
    context tells you what the model is about to say here, the average tells
    you what this direction disposes it to say in general.
    """
    encoded = {p: model.tokenizer.encode(p) for p in phrases}
    sums = {
        p: {l: torch.zeros(model.d_model) for l in source_layers} for p in phrases
    }
    counts = {p: 0 for p in phrases}
    for prompt_idx, prompt in enumerate(prompts):
        ctx = model.encode(prompt, max_length=max_seq_len)
        if ctx.shape[1] <= skip_first + 1:
            continue
        for phrase in phrases:
            ids = encoded[phrase]
            if not ids:
                continue
            grads = phrase_logprob_gradient(
                model, ctx, ids, source_layers, skip_first=skip_first
            )
            for layer, g in grads.items():
                sums[phrase][layer] += g
            counts[phrase] += 1
        if log_every and (prompt_idx + 1) % log_every == 0:
            logger.info("  phrase-jacobian prompt %d/%d", prompt_idx + 1, len(prompts))
    return {
        p: {l: v / max(counts[p], 1) for l, v in layers.items()}
        for p, layers in sums.items()
    }


# ----------------------------------------------------------------------------
# Method B — debiased token sum
# ----------------------------------------------------------------------------


@torch.no_grad()
def phrase_continuation_distributions(
    model: LensModel,
    phrase_ids: Sequence[int],
    contexts: Sequence[str],
    *,
    max_seq_len: int = 128,
) -> torch.Tensor:
    """``p_i``: the model's predictive distribution at each phrase position.

    Averaged over ``contexts`` so the correction is a property of the phrase
    rather than of one sentence. Returns ``[n_phrase_tokens, vocab]``.
    """
    total = None
    used = 0
    for text in contexts:
        ctx = model.encode(text, max_length=max_seq_len)
        if ctx.shape[1] < 2:
            continue
        full = torch.cat(
            [
                ctx,
                torch.tensor(list(phrase_ids), dtype=torch.long, device=ctx.device)[
                    None
                ],
            ],
            dim=1,
        )
        T = ctx.shape[1]
        logits = model.unembed(model.forward(full)).float()[0]
        probs = logits[T - 1 : T - 1 + len(phrase_ids)].softmax(dim=-1)
        total = probs if total is None else total + probs
        used += 1
    if total is None:
        raise ValueError("no usable contexts")
    return total / used


def debiased_token_sum(
    V: torch.Tensor,
    phrase_ids: Sequence[int],
    continuation_probs: torch.Tensor,
) -> torch.Tensor:
    """Method B: ``sum_i ( v_{w_i} - E_{p_i}[v] )``.

    Args:
        V: ``[vocab, d_model]`` J-lens vectors at one layer.
        phrase_ids: The phrase's token ids.
        continuation_probs: ``[n_tokens, vocab]`` from
            :func:`phrase_continuation_distributions`.

    Returns:
        ``[d_model]``.
    """
    V = V.float()
    out = torch.zeros(V.shape[1])
    for i, tid in enumerate(phrase_ids):
        out += V[tid] - continuation_probs[i].float() @ V
    return out


def build_debiased_vectors(
    model: LensModel,
    lens: JacobianLens,
    phrases: Sequence[str],
    contexts: Sequence[str],
    source_layers: Sequence[int],
    *,
    max_seq_len: int = 128,
) -> dict[str, dict[int, torch.Tensor]]:
    """Method B over a phrase list. One forward pass per (phrase, context) and
    no backward passes at all — the ``J_l`` you already fitted does the rest."""
    from jlens.cka import build_probe

    P_full, _ = build_probe(model, token_ids=torch.arange(len(model.tokenizer)))
    V_by_layer = {l: (P_full @ lens.jacobians[l]).float() for l in source_layers}
    out: dict[str, dict[int, torch.Tensor]] = {}
    for phrase in phrases:
        ids = model.tokenizer.encode(phrase)
        if not ids:
            continue
        probs = phrase_continuation_distributions(
            model, ids, contexts, max_seq_len=max_seq_len
        )
        out[phrase] = {
            l: debiased_token_sum(V_by_layer[l], ids, probs) for l in source_layers
        }
    return out


# ----------------------------------------------------------------------------
# Method C — proposing candidate phrases without an oracle
# ----------------------------------------------------------------------------


@dataclass
class PhraseProposal:
    phrase: str
    token_ids: list[int]
    seed_token: str
    variance_explained: float


@torch.no_grad()
def propose_phrases(
    model: LensModel,
    lens: JacobianLens,
    activation: torch.Tensor,
    layer: int,
    *,
    n_seeds: int = 12,
    max_phrase_tokens: int = 4,
    context: str | None = None,
    top_k_continuations: int = 2,
) -> list[PhraseProposal]:
    """Propose multi-token concepts for one activation, without an oracle model.

    The loop:

    1. Decompose ``activation`` over the single-token J-lens vectors and take
       the top ``n_seeds`` tokens. These are the concepts the plain lens can
       already see — including the truncated ones (``black``, ``phot``).
    2. Let the model itself continue each seed token greedily, in ``context``
       if given. The model's own continuation of ``black`` in a blackmail
       context is ``mail``; this is where the multi-token concept comes from,
       and it costs a few forward passes rather than a fine-tune.
    3. Score each candidate by how much of the activation its debiased vector
       explains, and return them ranked.

    This is the untrained counterpart to the oracle lens: propose from the
    model's own next-token distribution, verify by reconstruction.
    """
    from jlens.cka import build_probe

    vocab_ids = torch.arange(len(model.tokenizer))
    P_full, _ = build_probe(model, token_ids=vocab_ids)
    V = (P_full @ lens.jacobians[layer]).float()

    decomposition = gradient_pursuit(activation.float(), V, n_seeds)
    seeds = [int(i) for i in decomposition.indices]

    prefix_ids = model.encode(context)[0].tolist() if context is not None else []
    proposals: list[PhraseProposal] = []
    x = activation.float()
    x_energy = max(float(x.pow(2).sum()), 1e-12)

    for seed in seeds:
        # Greedily extend the seed token using the model's own continuation.
        chain = [seed]
        for _ in range(max_phrase_tokens - 1):
            ids = torch.tensor([prefix_ids + chain], dtype=torch.long)
            logits = model.unembed(model.forward(ids)).float()[0, -1]
            chain.append(int(logits.argmax()))
        # Score every prefix length; the best one is the concept's extent.
        for length in range(1, max_phrase_tokens + 1):
            candidate = chain[:length]
            v = V[candidate].sum(dim=0)
            unit = v / v.norm().clamp_min(1e-12)
            proposals.append(
                PhraseProposal(
                    phrase=model.tokenizer.decode(candidate),
                    token_ids=list(candidate),
                    seed_token=model.tokenizer.decode([seed]),
                    variance_explained=float((x @ unit) ** 2) / x_energy,
                )
            )
    proposals.sort(key=lambda p: -p.variance_explained)
    return proposals


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------


def rank_of_vector(
    vectors: dict[str, torch.Tensor], target: str, activation: torch.Tensor
) -> int:
    """Rank of ``target`` when all candidate phrase vectors are scored against
    ``activation`` by cosine similarity. Rank 1 is best.

    Cosine rather than raw inner product because the constructions differ in
    natural scale, and a readout that just prefers whichever method produces
    bigger vectors would tell us nothing.
    """
    names = list(vectors)
    M = torch.stack([vectors[n].float() for n in names])
    M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    h = activation.float()
    scores = M @ (h / h.norm().clamp_min(1e-12))
    ranked = [names[int(i)] for i in scores.argsort(descending=True)]
    return ranked.index(target) + 1
