# MATS application — draft

> **Before you send this:** read the two notes at the bottom. One is about
> mapping this to the actual application prompts, and one is about disclosing
> AI assistance. Neither is optional.

---

## What I did

I reimplemented the Jacobian lens from the Gurnee et al. global-workspace paper
against a `LensModel` protocol that runs on any HuggingFace decoder, built a
1.4M-parameter toy transformer with a synthetic relational world so ground
truth is available, and attacked the four open problems the paper names.

Everything below is a number from a run in
[the repo](https://github.com/fbanki4/Jlens), not a recollection.

### 1. A bag of concepts without training a probe

The paper gets its concept vectors from a trained sparse autoencoder. I wanted
a version that trains nothing.

The construction: instead of averaging token embeddings, take the Jacobian of
the *phrase's* log-probability with respect to the residual,
`v_w(ℓ) = E[∂ log p(w | prompt) / ∂h_{ℓ,t}]`. Because it differentiates the
whole phrase's probability, a multi-token concept is not a special case.

Top-1 decoding accuracy on held-out items, by concept length:

| method | 1 token | 2 tokens |
|---|---|---|
| **phrase Jacobian** | **0.79** | **0.76** |
| first-token embedding | 0.56 | 0.65 |
| token-average embedding | 0.54 | **0.15** |
| template readout | 0.17 | 0.22 |

The interesting cell is token-averaging at length 2. It is competitive on
single-token concepts and then collapses to 0.15 — barely above the 0.02
chance floor — exactly where the concept stops being a single token. The
phrase Jacobian does not degrade. That is the whole argument for it: the
standard cheap trick fails precisely on the cases a workspace claim needs.

### 2. When the J-space forms, relative to the capability

Fitting a lens at every training checkpoint of the toy model: two-hop accuracy
reaches 90% of its final value at **step 145**, but the J-space geometry does
not stop moving until **step 2301** — 16× later. The model can do the task
long before the representation that supports it settles.

### 3. What the shapes in the open visualisation actually indicate

I ran the frozen segmentation rule over all 67 CKA matrices in the open
measurement campaign, and swept the block-count penalty it depends on.

- **62 of 67 models segment into exactly two blocks.** Their reported
  `sensory_end` and `motor_start` are therefore the *same cut*. A
  three-regime reading of those pictures is supplied by the reader, not by the
  data.
- The block count is robust to the penalty: only **1 of 67** flips within
  ±0.02 of the default.
- **Depth does not predict blockiness** (r = +0.04). Whatever the shape is
  about, it is not how deep the model is.
- Changing only the optimizer moves the boundary by 0.16 in relative depth —
  comparable to the spread across entirely different model families.

### 4. A lens-free test for ignition, and a null result

Every measure above depends on a lens I fitted, so a "no workspace here"
result cannot be told from "my instrument was dead here". I built an
ambiguity-blend paradigm that never touches the lens: blend two concept
embeddings, sweep the mixture, and measure how sharply the residual commits to
one reading. Proportional tracking reads 1.0; a perfect all-or-none switch
reads 20.

**Peak sharpness across every depth and every checkpoint: 2.73.** No ignition,
on a measure that could not have been faked by a bad lens.

I also implemented broadcast breadth causally — perturb along a direction, see
how far the disturbance travels, ratio against matched-norm noise — rather
than as weight alignment. It peaks **outside** the workspace band (1.67 vs
1.47 inside), which is the opposite of the global-workspace prediction.

---

## The part I actually want you to read

Two of my headline findings were wrong, and I found both myself.

**The first.** My workspace band was `[l for l in layers if 0.3 <= l/(L-1) <= 0.8]`.
On a 2-layer model that selects nothing. Downstream code initialised
"best rank" to a `10**9` sentinel, and the specificity ratio printed as
`1e9/1e9 = 1.0`. I had written it up as the headline result — "read the first
row twice" — before noticing that a suspiciously clean 1.0 was an empty list
divided by itself. The real value is 1.3×. The fix is a shared
`workspace_band()` that falls back to all fitted layers and never returns
empty.

**The second.** I reported that causal swap success "appears between depth 4
and 8". The actual sequence across the depth ladder was 0.086 → 0.000 →
0.094 → 0.000. That is noise, and I had drawn a threshold through it.

I mention these because the failure mode they share — a metric that produces a
clean number for a degenerate reason — is the one I now design against first.
Every measurement in the repo carries its own control: broadcast is a ratio
against matched-norm noise, occupancy is scored against a random dictionary of
the same shape, and the scale-sweep code refuses to pool runs whose fitting
budgets hash differently.

The sharpest version of that is the instrument control. The J-lens is fitted
per model, so on a scale sweep lens quality varies across the grid and is
itself correlated with scale — meaning "workspace structure grows with
compute" and "my instrument gets better with compute" produce the same
correlation. So I measure lens fidelity against the logit lens at every rung
and regress it out. On synthetic data where the measure is *purely* the
instrument, this takes a raw r of +0.97 to a partial r of +0.00; where there
is a real effect underneath the same instrument ramp, it survives at +0.99.
Rungs whose lens never beat the logit lens are marked dead and excluded.

Building that also caught a bug in my own control: when the covariate
explained a measure *entirely*, the naive residual-variance guard returned
`NaN` — hiding the exact finding the control exists to produce behind a
missing value.

---

## What I would work on

See `docs/06-nested-learning-proposal.md` for the full version. In one
paragraph:

Google's Nested Learning frames a model as nested optimisation problems whose
components update at different frequencies, and its Continuum Memory System
assigns those frequencies **by hand**. The J-lens can measure which depths of a
network are still reorganising and which have settled — so the frequencies
could be *measured* rather than chosen. I have run the first version of that
measurement on public training checkpoints, found it confounded by a
saturation ceiling, and know what would settle it.

---

## Notes for me before submitting

**1. Map this to the real prompts.** This is written as a research summary. The
actual application asks specific questions with specific word limits, and this
should be cut down and reshaped to answer *those*, not delivered as-is. The
strongest material is section 4 (the null result) and "the part I actually want
you to read" — lead with those if there is a choice.

**2. Disclose the AI assistance.** This work was done in an extended session
with an AI assistant that wrote most of the code and ran the experiments. That
needs stating plainly, in whatever form the application asks for — and if it
doesn't ask, say it anyway. Two reasons, one principled and one practical: it
is straightforwardly the truth about how the work was produced, and MATS
interviews go deep on submitted work. Every number above is reproducible from
the repo, but be able to derive the cotangent trick at a whiteboard, explain
why the estimator sums over target positions `p' ≥ p`, and say why linear CKA
is the right similarity measure here — because you will be asked.

**3. Decide what is yours.** The research questions, the direction, and the
nested-learning connection are yours. The implementation is largely not. Be
specific about which is which rather than letting the reader assume.
