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
of GPU-hours, i.e. **well under a dollar** on a spot 3090. The lens-fitting
stage is the expensive one (it needs `d_model / dim_batch` backward passes per
prompt), and that is the stage to move to a rented GPU — not this one.

## Colab, copy-paste

```python
!pip install -q transformers
!git clone -q https://github.com/fbanki4/Jlens && cd Jlens && git checkout -q claude/j-lens-jspace-learning-47fsd0
%cd Jlens
!python experiments/pythia_sweep.py --sizes 70m,160m --steps 0,1,2,4,8,16,32,64,128,256,512,1000,4000,16000,64000,143000 --purge
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
