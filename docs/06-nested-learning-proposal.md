# Measuring update frequencies instead of choosing them

**A MATS project proposal: connecting Nested Learning to J-space geometry.**

---

## The gap

Google's *Nested Learning* (Behrouz et al., NeurIPS 2025) reframes a model as a
set of nested optimisation problems, each with its own context flow and its own
**update frequency**. Its Continuum Memory System generalises the usual
short-term/long-term memory split into a spectrum: a chain of blocks, each
updating at a different rate, so that fast-moving components absorb new data
while slow-moving ones hold what was already learned. That is a direct attack
on catastrophic forgetting.

The frequencies are **hyperparameters**. They are chosen — a designed chain of
update rates — and then the architecture is trained around that choice.

The claim I want to test is that they should not have to be. If a network's own
training dynamics already reveal which depths keep reorganising and which
settle early, then the frequency assignment is a *measurable property of the
model* rather than a design decision. And the J-lens is a way to measure it,
because it gives a comparable readout geometry at every depth of any model:
`lens_l(h) = unembed(J_l · h)`, with `J_l = E[∂h_final/∂h_l]`.

**The hypothesis:** plasticity is not uniform across depth. Early layers settle
on an input code and stop; late layers settle on an output code and stop; the
middle keeps reorganising. If so, a continual-learning system should let the
middle move and hold the edges still — and that assignment is free, because it
falls out of a measurement you can take before you start.

## What I found when I tested it

I have run the first version of this on the open campaign's three published
training runs (OLMo-3-32B, 11 checkpoints; SmolLM3-3B, 12; OLMo-7B, 4).

The campaign does not publish fitted lens tensors — they are 0.02 to 63 GB
each — so I could not compute the quantity I actually want, which is
cross-checkpoint CKA of `V_l`. What it does publish is the layer × layer CKA
matrix per checkpoint, so I measured a proxy: how much layer `l`'s relations to
every other layer moved between consecutive checkpoints.

**Raw, the hypothesis comes out backwards.** The middle drifts *less* than the
edges in all three runs — 0.71×, 0.62×, 0.89× — and a quadratic fit in relative
depth is positive in every case, i.e. a bowl rather than a hump.

**But there is a ceiling, and it is large.** Middle rows have the highest mean
CKA (0.92 vs 0.83 and 0.87 in OLMo-32B) and the smallest spread. A row already
near 1.0 against every other layer *cannot move* whatever training does to it.
The correlation between drift and row saturation is −0.36, −0.60 and −0.88
across the three runs. Divide through by the available headroom and two of the
three flip: 1.16×, 0.76×, 1.34×.

| run | raw middle/edges | headroom-corrected |
|---|---|---|
| OLMo-3-32B | 0.71× | 1.16× |
| SmolLM3-3B | 0.89× | 1.34× |
| OLMo-7B (early) | 0.62× | 0.76× |

So the two readings disagree and the honest conclusion is that **this proxy
does not settle the question** — not that the hypothesis is confirmed, and not
that it is refuted. Reporting either direction from this data would be picking
the correction that flatters the story.

That is where the project starts, and it is a better starting point than a
clean confirmation would have been: the question is well-posed, the obvious
measurement is confounded, and the confound is identified and quantified.

## The project

### Stage 1 — measure plasticity properly (weeks 1–3)

Fit lenses on **Pythia** checkpoints rather than reusing the campaign's
matrices. Pythia removes every confound that makes the current data
inconclusive: 154 checkpoints per size versus 4–12, eight sizes on *identical
data in identical order*, and one tokenizer throughout.

With the tensors in hand, measure the quantity the proxy was standing in for —
cross-checkpoint CKA of `V_l` — which has no saturation ceiling, since it
compares a layer to *itself* at an earlier time rather than to its neighbours.

Report the plasticity profile against relative depth, and check it is the same
shape across sizes. If it is, that is already a result: it says depth-relative
plasticity is a structural property rather than a per-model accident.

The instrument control from the existing work carries over unchanged and is
load-bearing here: the lens is fitted per checkpoint, so lens quality varies
along exactly the axis being measured. Fitting budget is held fixed and
hashed, lens fidelity is measured against the logit lens at every checkpoint,
and it is regressed out of the headline relationship.

**This stage can fail informatively.** If plasticity turns out to be flat in
depth, the rest of the project is dead and I would rather know in week 3 than
week 10. That is a feature of the ordering, not a risk of it.

### Stage 2 — does measured plasticity predict what forgets? (weeks 4–6)

A plasticity profile is only useful if it predicts something. The test:
fine-tune on a new task, then measure what the model lost on the old one, as a
function of which depths were allowed to move.

Freeze by *measured* plasticity — the top-k most plastic depths stay free,
everything else is held — and compare against three baselines: freeze nothing,
freeze a random depth-matched subset, and freeze by gradient norm, which is the
cheap thing you would try first and the one this has to beat.

If measured plasticity is not better than gradient norm, the J-lens is adding
nothing over a much simpler signal and I should say so.

### Stage 3 — set the nested frequencies from the measurement (weeks 7–10)

Take a Continuum Memory System, and instead of a hand-chosen frequency chain,
assign each level's update rate from the measured plasticity profile — fast
where the geometry keeps moving, slow where it settled.

Evaluate on a continual-learning sequence against the hand-set chain from the
paper and against a uniform-frequency ablation. The comparison that matters is
not "does it work" but "does *measuring* beat *choosing*", so the hand-set
baseline gets tuned properly rather than taken at its default.

### What would make me drop it

- Stage 1 shows a flat profile → no signal to use, stop.
- Stage 2 loses to gradient norm → the lens is not earning its cost, stop and
  write up the negative result, which is worth having.
- The profile is real but the frequencies derived from it don't beat hand-set
  ones → report as a negative result about the *transfer*, which still leaves
  stage 1 standing as a measurement contribution.

## Why the J-lens rather than something cheaper

The obvious objection is that weight-change norms or gradient magnitudes are
far cheaper ways to ask which layers are moving. Three reasons the lens is
worth its cost, and stage 2 is designed to check the first two:

**Weight change is not representation change.** A layer can have large weight
updates that leave its readout geometry alone, and small ones that reorganise
it. The lens measures the thing continual learning actually cares about — what
the layer *means* downstream — rather than how far its parameters travelled.

**It is comparable across depth and across models.** `V_l = P · J_l` is indexed
by vocabulary tokens, which mean the same thing at every layer and in every
model sharing a tokenizer. Gradient norms at layer 3 and layer 30 are not on a
common scale; CKA over `V_l` is.

**It connects to a structural claim already in evidence.** Every one of the 49
CKA matrices I segmented has exactly two blocks, across 32 to 66 layers, dense
and mixture-of-experts, quantized and not, four different optimizers. The
boundary depth varies a lot (ρ 0.27 to 0.98, median 0.52) while the *count*
does not. If a network's readout geometry really is two regimes with a movable
seam, then "which parts should keep learning" is a question about where that
seam sits — and the seam is measurable.

## What I already have

Working, tested, and public in [the repo](https://github.com/fbanki4/Jlens):

- lens fitting against any HuggingFace decoder, sharded and resumable
- checkpoint discovery from a repo's refs, so Pythia/OLMo/SmolLM3/Amber and
  families I have not heard of all work without a hardcoded table
- the campaign's exact data format, verified byte-identical against their
  files, so their 77 lenses and my runs go in one viewer
- the instrument control: fixed hashed fitting budgets, lens fidelity against
  the logit lens, partial correlation to regress it out
- the drift measurement above, including the ceiling control that is the reason
  I am not claiming a result from it
- 24 tests, including the estimator against a closed-form Jacobian and the
  broadcast ratio against its analytic value

Stage 1 is a GPU-week of work on top of this, not a rebuild.
