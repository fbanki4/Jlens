# Running the Pythia sweep without spending money

Short version: **you do not need a HuggingFace subscription, and you probably
do not need a GPU.**

## Do I have to pay HuggingFace?

No. Pythia is **ungated** — public weights, no login, no token, no licence
click-through. `from_pretrained("EleutherAI/pythia-70m", revision="step1000")`
works anonymously.

A free account is only needed for *gated* models (Llama, Gemma), and paid
tiers only for private storage, Inference Endpoints, or Spaces hardware. None
of that is involved here.

## Do I need a GPU?

Not for the experiment that matters first. The ignition stage is **forward
passes only** — no lens fitting, no backward passes — which is why the
protocol runs it first. Measured on four CPU cores:

| model | per trial | 40 trials | × 16 checkpoints |
|---|---|---|---|
| pythia-70m | 0.53 s | 0.4 min | ~6 min |
| pythia-160m | 0.87 s | 0.6 min | ~10 min |
| pythia-410m | 2.86 s | 1.9 min | ~30 min |

**About 45 minutes of compute on a laptop** for the whole fast-signal sweep.

Download is the real bottleneck, not arithmetic: 16 checkpoints of pythia-410m
in fp32 is ~26 GB. Pass `--purge` and each checkpoint is deleted after being
measured, so peak disk stays at one model.

```bash
pip install torch transformers
python experiments/pythia_sweep.py \
    --sizes 70m,160m,410m \
    --steps 0,1,2,4,8,16,32,64,128,256,512,1000,4000,16000,64000,143000 \
    --purge --out runs/pythia
```

## Stage 2, the lens stage

This one wants a GPU, and the reason is structural rather than incidental.
Fitting `J_l` costs `ceil(d_model / dim_batch)` backward passes **per prompt**,
each on a batch of `dim_batch` replicas of the sequence. Total backward work is
therefore about `n_prompts × d_model` sequence-backwards — independent of
`dim_batch`, which only trades memory for wall-clock.

At the defaults (16 prompts) that is ~8k sequence-backwards for pythia-70m and
~16k for pythia-410m, per checkpoint. On four CPU cores that is hours per
checkpoint; on a rented 3090 it is minutes. Run stage 1 on your laptop, then
move stage 2 to a GPU:

```bash
python experiments/pythia_sweep.py --stage lens \
    --sizes 70m,160m,410m --steps 0,16,512,4000,64000,143000 \
    --purge --out runs/pythia

python experiments/pythia_sweep.py --stage report --out runs/pythia
```

Fewer checkpoints than stage 1 on purpose — stage 2 exists to *agree or
disagree* with the lens-free curve at a handful of points, not to re-measure
it.

### Read the instrument column first

The sweep prints lens fidelity before it prints any result:

```
   size    step  lens r  vs logit  blockiness  PR ratio
    70m       0   0.200    +0.000       0.200     0.307  <- lens dead
    70m     512   0.230    +0.030       0.215     0.319
```

`vs logit` is the J-lens's margin over the plain logit lens. Where it is zero,
the fitted transport bought nothing, and every geometry number on that row is
describing the instrument rather than the model. Those rows are excluded from
the controlled fits and flagged in the JSON.

This matters because the lens gets better with scale on its own. The report
prints how strongly, and then regresses it out:

```
                 measure   raw r  partial r  shrinkage  left of y
              blockiness  +0.974     +0.000     +0.974      0.0%
```

That is the failure mode, caught: a raw correlation of +0.97 with compute that
survives none of the control. Nothing about the workspace, everything about the
lens warming up. A real effect looks like `+0.999 → +0.994` instead.

The comparison only means anything if every rung was measured the same way, so
the budget (prompt count, sequence length, probe layers, probe tokens) is
hashed into every row and the sweep **refuses** to append rows fitted under a
different budget. If you change `--lens-prompts` mid-sweep, write to a new
`--out`.

## The two GWT signatures

Separate from the scale sweep, and runnable on the toy model with no downloads:

```bash
python experiments/e_gwt_signatures.py
```

**Broadcast breadth** perturbs along a J-lens direction and measures how far the
disturbance actually travels, as a ratio against a random direction of identical
norm injected at the same place. 1.0 means "reaches exactly as far as noise of
the same size". The paper's version is weight-space alignment, which cannot
distinguish a direction a component reads from one it merely points at.

**The bottleneck** asks whether contents *compete* — hold `n` items, then `n+1`,
and see whether the first gets pushed out. Reported as an eviction slope plus a
selectivity, because a readout that dims uniformly as the prompt grows produces
the same slope without evicting anything.

## If you want to go bigger

Roughly in order of "free" to "cheap":

| option | cost | good for |
|---|---|---|
| your laptop, CPU | free | up to 410m, ignition stage |
| **Kaggle Notebooks** | free, ~30 GPU-hours/week (T4×2 or P100) | up to 2.8b; best free quota |
| Google Colab free | free, T4 when available | up to 1.4b in fp16 |
| Lightning AI Studio | free monthly credits | occasional bigger runs |
| Vast.ai / RunPod | ~$0.20–0.40/hr for a 3090/4090/A40 | the full 70m–2.8b grid |

Renting is genuinely cheap here: the full ignition grid up to 2.8b is a couple
of GPU-hours, i.e. **well under a dollar** on a spot 3090. Stage 2 is the
expensive one, for the reason above, and that is the stage to move to a rented
GPU — not this one.

## Colab, copy-paste

```python
!pip install -q transformers
!git clone -q https://github.com/fbanki4/Jlens && cd Jlens && git checkout -q claude/j-lens-jspace-learning-47fsd0
%cd Jlens
!python experiments/pythia_sweep.py --sizes 70m,160m --steps 0,1,2,4,8,16,32,64,128,256,512,1000,4000,16000,64000,143000 --purge
```

With a GPU runtime attached, add stage 2 and the report:

```python
!python experiments/pythia_sweep.py --stage lens --sizes 70m,160m --steps 0,16,512,4000,64000,143000 --purge
!python experiments/pythia_sweep.py --stage report
```

## What you are looking for

`sharpness` is normalised so that **proportional tracking reads 1.0** and a
perfect all-or-none switch reads `n_alphas - 1` (20 at the default). No
threshold to choose: you read the number.

For reference, the toy models in this repo peak at **2.73** across every depth
and every checkpoint — no ignition. If pythia-410m reads 3 and a larger model
reads 15, that is your curve.

Then check which axis the sharpness values collapse onto — compute (6ND),
parameters, or depth. The script prints all three correlations. That collapse
is the answer to "does the workspace come from scale or from architecture",
and it is a better answer than any single emergence threshold.
