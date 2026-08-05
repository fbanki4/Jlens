# Four open problems, investigated

Results from `experiments/`. Two substrates:

* **Real open models** — 67 layer × layer CKA matrices from a clone of
  `eliebak/open-jlens-data`, covering 1B–1T-parameter models, a 14-run
  optimizer study, training trajectories, corpus and quantization ablations.
  Used for the "shapes" question, which needs many real models.
* **A toy model** — 1.4M parameters, 12 layers, 96-wide, trained here in
  12 minutes on a synthetic relational world. Used for everything that needs
  ground truth about what the latent concept *is*, or a full training
  trajectory with matched capability evals.

The toy model's task: it has memorised `person lives in city` and
`city sits in country`, and was trained on the composed `person hails from
country` for 72 of 96 people. On the 24 held-out people it can only answer by
composing. It reaches **0.92 on held-out subjects against 0.083 chance**, with
1.00 atomic recall. The prompt `alice hails from` is three tokens, so the
intermediate city appears nowhere in it — the same structure as the paper's
"the color of the planet fourth from the sun is", where `Mars` is neither
asked for nor answered.

Scale caveats are stated where they bite. Several of them bite hard, and that
is itself a result.

---

## 0. First, does any of this reproduce?

Yes, on the read side, and the sanity check the paper uses passes cleanly.
J-lens readouts agree with the model's own next token far more often than the
logit lens at every depth:

| ρ | J-lens top-1 == model | logit lens | cos(J_ℓ, I) |
|---|---|---|---|
| 0.00 | 0.08 | 0.00 | 0.926 |
| 0.36 | 0.57 | 0.27 | 0.957 |
| 0.55 | 0.70 | 0.52 | 0.965 |
| 0.73 | 0.95 | 0.82 | 0.972 |
| 0.91 | 1.00 | 1.00 | 0.986 |

And the latent shows up. On `abel hails from` (a held-out subject; answer
`germany`, intermediate `berlin`), the readout at the query position goes:

```
L0   clam  german  munich  bulgarian  germany  olive
L3   germany  german  olive  bulgarian  clam  berlin
L5   germany  german  berlin  olive  little  europe
L8   germany  berlin  german  europe  sits  utters
```

`berlin` is in neither the prompt nor the output. Across 94 solved subjects,
the latent city reaches median rank 3 in the middle band while five randomly
chosen other cities sit at median rank 52 — **17× specificity**, and the
latent outranks the distractors on 100% of items. Held-out subjects do
slightly *better* than trained ones (68% vs 61% top-3), which is what you want:
the effect is not memorisation.

**Two things did not reproduce, and both are informative.**

*The logit lens is not worse here.* It puts the latent at median rank 1 with
30× specificity — better than the J-lens. This is not a bug: `cos(J_ℓ, I)`
never drops below 0.93, so a 12-layer model's coordinates barely rotate, and
the paper is explicit that the two lenses diverge in *early layers of deep
models*. There is also a conceptual reason to expect it. `J_ℓ` transports
toward `h_final`, i.e. toward what the model will actually *say*; a fully
consumed intermediate contributes little to `h_final`, so the transport
systematically damps exactly the content we are hunting for. On a shallow
model, where the logit lens is already well-calibrated, that damping is a net
loss. This is worth keeping in mind when applying the J-lens to small models.

*Steering is weak.* Swapping the latent's J-lens vector for another city's
redirects the answer on 9.6% of items (irrelevant-token control: 0.0%, so the
effect is real but small), against 45–68% for single-token concepts in the
paper. Sweeping the intervention band (L1–4, L4–8, L1–10) and α ∈ {1, 2, 4}
does not rescue it. The likely cause is visible in the read table: the answer
is already rank 1 by ρ=0.36, so most of the computation is finished before any
plausible "workspace band" begins. A 12-layer model doing a 2-hop lookup has
almost no room for a workspace.

---

## 1. A non-trained way to get the bag of concepts

**The problem.** The J-lens gives one vector per *token*, so it can only name
concepts the tokenizer spells in one piece — `blackmail` surfaces as `black`,
`photosynthesis` as `phot`. The paper's two fixes both build a trained
artifact first: the **template lens** needs an LLM to author ~100 passages per
word over a vocabulary enumerated in advance, and inherits the tuned lens's
habit of skipping to the answer; the **oracle lens** fine-tunes two auxiliary
models and then RLs one of them.

**Two constructions that need neither.** Implemented in `jlens/phrases.py`.

**Method A — the phrase Jacobian.** The J-lens definition already generalises:

```
v_w(ℓ) = E_{prompt, t} [ ∂ log p(w | prompt) / ∂ h_{ℓ,t} ]
```

Teacher-force the phrase onto the end of each corpus prompt, sum its token
log-probabilities, backprop once. Causality means that single backward yields
the gradient at *every* source position simultaneously, exactly as in the
ordinary fit. For a one-token phrase this *is* the J-lens (up to taking the
gradient at the probabilities rather than at the final residual — the same
deviation the paper notes for the template lens). Cost: one forward+backward
per (prompt, phrase), against the template lens's few hundred forwards per
word plus the authoring.

**Method B — the debiased token sum.** Chain-rule Method A through the final
layer and it collapses to something computable from an already-fitted `J_ℓ`
with no backward passes at all:

```
v_w(ℓ) ≈ Σ_i [ v_{w_i} − E_{p_i}[v] ]
```

where `p_i` is the model's own predictive distribution at the i-th phrase
position. The first term is the naive token-sum baseline; the second is the
correction it is missing. It subtracts the *generic continuation* — whatever
usually follows `w_1..w_{i-1}` — leaving what is specific to this phrase. That
is why `black` is a poor stand-in for `blackmail`: most of its vector is the
part shared with `blackboard` and `blackbird`, and that is precisely what
`E_{p_i}[v]` removes.

**Method C — proposing the dictionary.** Neither says *which* phrases to test.
`propose_phrases` gets candidates from the model itself: decompose the
activation over single-token vectors, let the model greedily continue each top
token in context, keep the continuations that improve reconstruction. The
oracle lens's job, done by propose-and-verify with nothing trained.

### Results

Half the toy world's cities are two words, so half the latents are genuinely
multi-token. Decoding ranks the true latent against all 176 world entities,
best over the workspace band:

| method | 1-token top-1 | top-3 | 2-token top-1 | top-3 |
|---|---|---|---|---|
| first-token (baseline) | 56% | 92% | 65% | 96% |
| token-average (baseline) | 54% | 94% | **15%** | **35%** |
| template (trained-data reference) | 17% | 35% | 22% | 70% |
| **phrase-Jacobian (untrained)** | **79%** | **98%** | **76%** | 87% |
| debiased-sum (untrained) | 17% | 79% | 33% | 74% |

The phrase Jacobian is the best decoder at both lengths, and — the point that
matters — it **does not degrade on multi-token concepts** (79% → 76%), where
token-averaging collapses (54% → 15%). It costs one backward pass per
(prompt, phrase) and no training.

The debiased sum is mid-table: clearly better than token-averaging on
multi-token latents (33% vs 15% top-1) and it is nearly free, but it does not
match the full gradient. That gap is the price of the first-order
approximation.

**Caveats, both real.** The template lens underperforms here (17–22% top-1),
worse than I expected. Its contexts were mined from the corpus rather than
authored, and the whitening uses a ridge of 1e-2 on a 96-dimensional
covariance estimated from limited samples — either could be responsible. I
would not read this table as evidence that the template lens is a bad method,
only that the untrained gradient does not need it. And on the causal side no
method does well (best 10.6%), for the depth reason in §0, so this comparison
rests on decoding alone.

---

## 2. When and how do J-spaces form?

The paper compares base against post-trained and finds the workspace already
present in the base. Open checkpoint series (OLMo, SmolLM3) let you watch the
geometry, but not the capability — those runs do not ship per-checkpoint evals
matched to what the workspace is supposed to support. Training the model here
gives both, which makes the sharper question askable: **does the geometry
settle before, with, or after the ability that routes through it?**

Sixteen log-spaced checkpoints, a lens fitted at each:

| step | held-out 2-hop | blockiness | coupling | lock-in | PR peak |
|---|---|---|---|---|---|
| 1 | 0.00 | 0.142 | 0.801 | 0.116 | 60.1 |
| 28 | 0.00 | 0.184 | 0.840 | 0.249 | 61.6 |
| 48 | 0.00 | 0.053 | 0.958 | 0.413 | 29.9 |
| 83 | 0.04 | 0.087 | 0.938 | 0.448 | 8.9 |
| **145** | **0.83** | 0.092 | 0.920 | 0.490 | 14.2 |
| 252 | 0.96 | 0.076 | 0.932 | 0.608 | 20.8 |
| 761 | 0.75 | 0.022 | 0.985 | 0.758 | 9.2 |
| 1324 | 0.96 | 0.000 | 0.991 | 0.838 | 7.8 |
| **2301** | 0.96 | 0.000 | 0.987 | **0.957** | 12.0 |
| 4000 | 0.92 | 0.021 | 0.985 | 1.000 | 12.5 |

*lock-in* = matched-depth CKA against the final checkpoint.

**Finding 1 — the capability arrives long before the geometry settles.**
Two-hop accuracy hits 90% of its final value at **step 145**; lock-in hits 90%
at **step 2301**, sixteen times later. At the moment the model can first do the
task, its J-space geometry is less than half of the way to where it ends up
(lock-in 0.49) and then keeps rotating for another order of magnitude of
training. The two are correlated across checkpoints (+0.90) only because both
rise monotonically; the timing says they are not the same event. Whatever the
late geometry change is doing, it is not what makes the model able to compose.

**Finding 2 — at this scale the block structure dissolves rather than forms.**
Blockiness falls from 0.14 to 0.02 and layer coupling rises from 0.80 to 0.99;
at steps 1324 and 2301 the segmenter finds **no blocks at all** (k=1). The
J-lens geometry becomes *more* uniform across depth with training, not more
differentiated.

That is the opposite of the real trajectories. On OLMo-3-32B, blockiness rises
0.17 → 0.36 over the first ~1T tokens and coupling falls 0.87 → 0.78; SmolLM3-3B
rises 0.22 → 0.30 then relaxes to 0.20 through its mid-training stages. So the
direction of the effect flips somewhere between 1.4M and 3B parameters. The
paper asks whether small models have "an equally rich workspace, a
proportionally smaller one, a less reliable one, or none at all." At 1.4M
parameters the answer looks like **none**: no block structure, occupancy ≈ 0,
and J-space variance-explained *below* a random-dictionary control at every
layer but the last. This is a scale threshold, and locating it is a concrete
follow-up.

**Finding 3 — a methodological warning.** The paper's capacity measures do not
survive the trip down. Occupancy is defined as the `k` beyond which a
same-sized *random* dictionary reconstructs as well; with 217 dictionary
vectors in 96 dimensions the random dictionary wins outright, because real
J-lens vectors are strongly anisotropic and 25 of them span less than 25
isotropic directions do. The comparison measures dimensionality, not
structure. Anyone porting occupancy/FVE to a small model needs a different
control — `explained_by_random` in `experiments/e_structure.py` uses a
matched-dimension random subspace instead.

---

## 3. Is the J-space more than a bag of concepts?

The paper's caveat: a readout containing *spider*, *legs* and *eight* says
those concepts are present, not how they are bound. Composed relations let two
differences be isolated cleanly.

**Latent axis** — two subjects with the *same* answer country but different
intermediate cities. `h(A) − h(B)` differs in the unverbalized intermediate
with the output held fixed. (184 pairs.)

**Role axis** — the same city under two relations, `paris utters` (→ language)
and `paris stands within` (→ continent). Same subject, same latent country,
different operation in flight. (16 pairs.)

| ρ | latent: 2 city vectors | chance | ratio | role: 4 named vectors | chance | ratio |
|---|---|---|---|---|---|---|
| 0.36 | 0.161 | 0.020 | 7.9× | 0.257 | 0.047 | 5.5× |
| 0.55 | 0.184 | 0.021 | 9.0× | 0.292 | 0.046 | 6.3× |
| 0.73 | 0.220 | 0.021 | 10.5× | 0.403 | 0.046 | 8.8× |

Both are far above the matched-dimension chance level, so the J-lens vectors
are genuinely aligned with both differences. But both are also *small in
absolute terms*: the two cities' own vectors account for only **19%** of the
latent difference, and the tokens a bag-of-concepts readout could name account
for **32%** of the role difference. **81% and 68% respectively lie outside the
directions the readout would name.**

Two readings, and the data does not separate them. Either the workspace really
does carry binding and role structure that a flat concept list cannot express —
the paper's own suspicion — or a substantial part of the difference is
non-workspace processing that happens to co-vary. What the numbers do settle
is that "read the top-k tokens" is not a complete description of what changed:
it is capturing a fifth to a third of it. Notably the role axis is *better*
explained than the latent axis, which is the opposite of what a pure
bag-of-concepts account predicts (concept identity accounted for, relational
role not).

---

## 4. What do the different J-space shapes indicate?

The viz shows one segmented CKA heatmap per open model, and they do not all
look the same — some read as two blocks, some three, boundaries at different
depths. The natural reading is that these are facts about the models. Across
67 real matrices, that reading survives in one place and fails in three.

**The block count is not a segmenter artifact.** Sweeping the selection
penalty from 0 to 0.2, `n_blocks` is unchanged over the entire range for
66 of 67 matrices. Only `gemma-3-1b-pt` is fragile (k=3 holds only over
[0.016, 0.024]). So the reported `k` is robust.

**But 62 of 67 matrices segment into *two* blocks** — meaning `sensory_end`
and `motor_start` are the *same cut*. In those models the segmentation has not
identified a separate motor band at all. A three-regime
sensory/workspace/motor reading of those pictures is supplied by the reader,
not by the data. Median blockiness is 0.247, i.e. within-block CKA exceeds
between-block CKA by about a quarter — visible, not dramatic.

**The late boundary is reproducible; the early one is not.** Forcing k=3 on
every ablation variant of the same model:

| model | variants | early cut spread (ρ) | late cut spread (ρ) |
|---|---|---|---|
| gemma-3-1b-pt | default, 4k, frozen-attn, vp-diag, vp-heads | 0.040 | **0.000** |
| gemma-3-1b-it | default, 4k, frozen-attn | 0.280 | **0.000** |
| Qwen3-1.7B | default, 4k | 0.074 | **0.000** |

Across all multi-variant groups the median early-cut spread is 0.152 ρ against
0.097 for the single forced cut. The workspace→motor transition lands in the
same place regardless of fitting corpus, prompt count or attention ablation.
The sensory→workspace transition moves by up to 28% of the model's depth.
**Read late boundaries as model properties; treat early boundaries as
measurement-dependent.**

**Changing only the optimizer moves the shape as much as changing the model.**
Fourteen runs, same architecture, same data, same recipe, different optimizer:

```
forced-k=2 cut across 14 optimizers: 0.419 – 0.581   (spread 0.162 ρ)
blockiness across the same runs:     0.171 – 0.261   (spread 0.090)
```

A 0.16 ρ spread from the optimizer alone is larger than the difference between
many models people would compare as different. Two heatmaps differing by that
much are not evidence of different internal organisation.

**And depth does not explain the differences.** Over 26–66 layers:
corr(depth, blockiness) = **+0.04**, corr(depth, coupling) = +0.34,
corr(depth, forced-k=2 cut) = −0.39. Deeper models put the single cut slightly
*earlier* in relative depth; blockiness is essentially uncorrelated with depth.

**Is the early block a regime, or is the lens just blind there?** The paper
lists this as open. The two accounts differ in the CKA fingerprint: a genuine
processing regime drifts (adjacent layers more similar than distant ones),
while a degenerate band with every `V_ℓ` collapsed onto one low-rank subspace
is flat and tight. Measuring the within-block distance-decay slope and spread:

```
median first block:  std 0.066  slope -0.0138
median later blocks: std 0.061  slope -0.0156
first blocks that are flat AND tight: 0/64
```

The first block drifts with depth just as much as the later ones. **The
"lens is blind early" account predicts a signature that does not appear in any
of the 64 matrices.** Whatever makes the early band different, it is not
absence of structure. That is a point in favour of the paper's other reading —
that early layers compute something real which simply is not workspace-shaped.

### Summary

| claim about a shape | verdict |
|---|---|
| the block count | robust (66/67 penalty-invariant) |
| three regimes visible in the picture | mostly not — 62/67 have only two blocks |
| the late (workspace→motor) boundary | reproducible across corpus, prompts, ablations |
| the early (sensory→workspace) boundary | moves by up to 0.28 ρ; treat as measurement-dependent |
| shape differences between models | confounded — the optimizer alone moves the cut by 0.16 ρ |
| depth explains the differences | no (r = +0.04 with blockiness) |
| the early block is lens degeneracy | not supported (0/64 show the signature) |

---

## Reproducing

```bash
python -m experiments.run_all           # trains, fits, and runs 1-3
python experiments/e_shapes.py --data-root <clone of eliebak/open-jlens-data>
python -m tests.test_core               # 10 correctness tests
```

Raw outputs land in `runs/{multihop,phrase,formation,structure,shapes}/`.
