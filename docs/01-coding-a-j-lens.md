# Coding a J-lens for an open-source model

Everything here is implemented in `jlens/`. This document explains *why* each
piece is the way it is, because most of the J-lens's value comes from three
choices that look like details and are not.

---

## 1. The idea in one line

```
lens_l(h) = unembed( J_l @ h )        J_l = E[ ∂h_final / ∂h_l ]
```

Take an activation from the middle of the network. You want to know what it
means. The model already has a machine for turning residual-stream vectors into
words — the unembedding — but it only works on the *final* layer's vectors.
So: linearly transport your mid-stack vector into the final layer's basis
first, then unembed it.

The transport is the average input-output Jacobian. `J_l` answers "if I nudge
the residual at layer `l`, which way does the final residual move, on average
over the kinds of text this model sees?"

### Why not just unembed directly?

That is the **logit lens** (`J_l = I`). It works in late layers because
residual connections mean late activations are already roughly in output
coordinates, and it degrades earlier. The J-lens is the principled correction:
instead of *assuming* the coordinates are shared, measure the map.

You can A/B them in one line here — `lens.apply(..., use_jacobian=False)` is
the logit lens.

### Why not fit a linear map to match the output distribution?

That is the **tuned lens**, and the objection is that it is *correlational*.
It is trained to predict what the model outputs, so on exactly the prompts you
care about — the ones with an unverbalized intermediate step — it learns to
skip the intermediate and jump to the answer. The J-lens never sees a
prediction target. It is a derivative of the model's own computation, so it
cannot be trained to shortcut anything.

---

## 2. The three choices that matter

### (a) The averaging is the whole point

A Jacobian computed on one prompt tells you what this activation is doing
*here*. Averaged over a thousand unrelated prompts, the context-specific part
cancels and what survives is the model's standing disposition: which output
this direction pushes toward, whatever else is going on.

That distinction has a name in the paper — **verbalizable** versus
**verbalized**. Averaging is what buys it. It is also why the fitting corpus
should be generic web-like text and *not* the task you plan to study: fit on
nothing but multi-hop prompts and you bake the task into the transport.

Corpus size saturates fast. The paper uses 1000 sequences of 128 tokens;
it beats logit- and tuned-lens baselines from about 10, and gains past a few
hundred are marginal. `fit()` prints a convergence diagnostic (the relative
shift in the running mean, which decays like `1/n` once settled) and can
early-stop on it.

### (b) Sum over future positions, mean over source positions

The estimator in `jlens/fitting.py`:

```python
cotangent[batch, valid_target_positions, dim] = 1.0
grads = torch.autograd.grad(target_activation, source_activations, cotangent)
J[dim_start:dim_start+n, :] = grads[:, valid_positions, :].mean(dim=1)
```

One one-hot cotangent per output dimension, planted at **every valid target
position at once**. Because attention is causal, the gradient that lands at
source position `p` is automatically

```
Σ_{p' ≥ p}  ∂h_final[p'] / ∂h_l[p]
```

— the effect of this activation on the current position's output *and every
future position's*. That is deliberate: a workspace representation is one that
is available to later computation, not just to the immediate next token.

The trick is also why one backward pass per output dimension suffices, and
why the forward *distance* `Δ = p' − p` is unrecoverable afterwards. If you
want to resolve it (how long does a nudge keep steering the output?) you need
one backward per target position — `delta_jacobian_for_prompt` does exactly
that, at roughly `n_targets` times the cost.

### (c) Skip the first ~16 positions

Early positions are attention sinks with wildly atypical residual norms, and
they will dominate a mean if you let them. `SKIP_FIRST_N_POSITIONS = 16`.

### Variants worth knowing

| Knob | Default | Alternative |
|---|---|---|
| target layer | last block | **penultimate** — the final block is specialised for calibrating next-token probabilities and adds noise |
| attention gradients | flow through QK | **frozen-QK** — stop-grad the attention pattern, isolating "what gets moved" from "what gets attended to" |
| target positions | all `t' ≥ t` | `t' = t` (logit-lens-like) or `t' > t` (pure broadcast) |
| aggregation | mean | median, or drop prompts with outlier Jacobian norms |

Results are reported to be robust across these; mean-over-penultimate is a
small win for surfacing intermediates, and frozen-QK can increase causal
effect.

---

## 3. Cost, and how to size a run

Per prompt: **one forward** and **`ceil(d_model / dim_batch)` backwards**.
Total scales as `params × d_model × n_prompts`.

- `dim_batch` trades memory for wall-clock. The prompt is replicated that many
  times on the batch axis and the retained graph is reused; total backward
  FLOPs are unchanged. Raise it until you run out of memory.
- `ActivationRecorder(..., start_graph_at=min(source_layers))` marks the
  shallowest source activation as the graph root. With model parameters frozen
  it becomes the *only* leaf, so the retained graph spans only that block
  onward rather than the whole network. Freezing parameters is not an
  optimisation detail here — it is what makes the memory tractable.
- Fitting is embarrassingly parallel over prompts. Run `fit()` on disjoint
  slices and combine with `JacobianLens.merge()`.

A 1–2B model with a few hundred prompts is a single-GPU afternoon. A 30B model
wants prompt-sharding across GPUs.

---

## 4. Running it on a real open model

```python
import transformers, jlens

hf  = transformers.AutoModelForCausalLM.from_pretrained(
          "Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda")
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = jlens.from_hf(hf, tok)          # locates the residual stack for you

lens = jlens.fit(model, prompts, dim_batch=64, max_seq_len=128,
                 checkpoint_path="out/ckpt.pt")
lens.save("out/lens.pt")

print(lens.readout(model, "The capital of the country shaped like a boot is",
                   layer=12, top_k=8))
```

`jlens/hf.py` handles Llama / Qwen / Mistral / Gemma / OLMo / Phi / GPT-2 /
Pythia layouts. Two things it does that matter:

- It calls the **bare decoder**, not the `*ForCausalLM`. Running the LM head on
  every fitting forward pass would materialise a `[batch, seq, vocab]` tensor
  which, for a 150k vocabulary, dwarfs everything else.
- It freezes parameters, for the graph-root reason above.

**Gemma caveat.** Gemma's RMSNorm stores its weight as an offset from 1, so the
effective gain is `1 + γ`. If you build a probe by reading weights out of
safetensors yourself, you must handle that; dropping the final-norm gain
degenerates the geometry on a large fraction of open models. `build_probe()`
sidesteps the whole issue by pushing the identity through the model's own
`unembed`, which folds the gain in automatically — for an RMSNorm model that
returns exactly `√d · W_U · diag(γ)`, and the global scalar is invisible to
both CKA and rank-based readouts.

---

## 5. Reading a slice

`jlens/vis.py` renders the layer × position view: rows are layers (deepest at
top), columns are token positions, each cell is the top-ranked lens token
there. Click a cell to pin its token and every cell recolours by that token's
rank.

What to look for, and roughly where:

- **Early third** — mostly junk. Either there is genuinely nothing verbalizable
  yet, or the lens cannot read it. This is an open question, not a settled
  fact (see `docs/03-open-problems.md`).
- **Middle band** — the interesting part. Abstract, task-relevant content that
  is in neither the prompt nor the output: the unspoken intermediate of a
  multi-hop question, `empty`/`ERROR` at a buggy call site, `nose` at the `^`
  of an ASCII face.
- **Last few layers** — the readout collapses onto the actual next token. This
  is the "motor" regime; it is no longer telling you about intermediate
  computation.

---

## 6. From reading to writing

The rows of `W_U J_l` are the **J-lens vectors**: one residual-stream direction
per vocabulary token (`lens.lens_vectors(model, layer)`). Reading is an inner
product against them. Writing is arithmetic on them (`jlens/jspace.py`):

- **steer** `h ← h + α v` (negative `α` suppresses)
- **ablate** project out `v`, or the span of the top-k active vectors
- **swap** `c = V⁺h; h ← h + V(σ(c) − c)` with `V = [v_source, v_target]` —
  exchanges two concepts while leaving everything orthogonal to their span
  untouched

The swap is the one that carries evidential weight. A concept showing up in a
readout means it is *present*; only an intervention that redirects the model's
answer shows it was the argument being consumed downstream. Always run the
control: swap two irrelevant tokens at the same layers and magnitude, and check
the answer does *not* move.

---

## 7. The J-space

Since `n_vocab > d_model`, the J-lens vectors span the entire residual stream —
so "their span" cannot be the definition of the workspace, or everything would
be in it. What makes it a real object is a **sparsity constraint**:

> the J-space is the set of points expressible as a sum of at most `k`
> J-lens vectors with non-negative coefficients

— a union of `k`-dimensional cones, one per choice of `k` vectors. A point's
J-space component is its nearest point in that union; the remainder lives
outside the workspace.

`jlens/jspace.py` finds it by `gradient_pursuit` (matching pursuit with an
exact line search on the selected support, non-negative). Two derived numbers:

- **occupancy** — the `k` beyond which a same-sized *random* dictionary
  reconstructs just as well. Around 25 in the paper's model, near zero in the
  first third of layers.
- **fraction of variance explained** — never more than ~10% over a random
  control. The workspace is a thin slice of what the model is computing.

Always quote both against the random control. Any dictionary of enough
directions reconstructs anything eventually.
