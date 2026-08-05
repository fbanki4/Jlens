# How the open-model J-space visualisation was made

`eliebak.com/viz/jspace-open` shows J-lens geometry for a few dozen
open-weight models. **The site itself is blocked by this environment's egress
policy**, so this reconstruction comes from the two repositories behind it,
both of which are public and were cloned here:

- `github.com/anthropics/jacobian-lens` — the paper's reference implementation
  (Apache-2.0): fitting, applying, and the slice viewer.
- `github.com/eliebak/open-jlens-data` — the measurement campaign: `program.md`
  (the full protocol), `code/` (fitters + analysis), `runs/` (results),
  `FIT_INVENTORY.json` (77 fitted lenses indexed; the raw `.pt` tensors, 0.02
  to 63 GB each, are excluded).

If you want the site's own framing, you will need to open it from an
unrestricted network — everything below is derived from the code and data, not
from the page.

---

## The pipeline

Every experiment in the campaign reduces to the same four steps.

### 1. Fit `J_l`

The canonical invocation, which the protocol marks "do not change silently":

```bash
python fit_lens.py <hf_model_or_local_path> --out_dir <out> \
  --dataset Salesforce/wikitext --dataset_config wikitext-103-raw-v1 \
  --text_field text --max_chars 2000 --n_prompts 1000 --dim_batch 64 \
  --max_seq_len 128 --dtype bfloat16 --device_map cuda \
  --min_prompts 100 --stop_window 10 --stop_at_delta 1e-3
```

So: wikitext-103, 1000 prompts of ≤128 tokens, bf16, with convergence-based
early stopping after a 100-prompt floor. The CLI is Neuronpedia's
`fit_lens.py`, a thin wrapper over the reference `jlens.fitting.fit`.

Cost model from the protocol, anchored on a 31B model at 53.7 s/prompt on a
B200:

```
t_fit ≈ 6.7e-13 · params · d_model · n_prompts   seconds
```

31B at 1000 prompts is ~33 GPU-hours on one H200. Prompt-averaging shards
trivially: run disjoint slices, then `J = Σ jacobian_sum / Σ n_done`.

### 2. Build the probe

```
P = W_U[ids] · γ_final          4096 token ids, numpy RandomState(0), no replacement
```

Three notes, all load-bearing:

- **The final-norm gain must be folded in.** The protocol records that dropping
  `γ` "degenerates ~half the open models". Gemma stores its RMSNorm weight as
  an offset from 1, so its effective gain is `1 + γ`.
- **4096 sampled ids ≈ full vocabulary**, to within ±0.005 CKA.
- **The sampler is frozen** on `(vocab_size, n, seed)`, so two models sharing a
  tokenizer get literally the same rows and are directly comparable. Across
  *different* tokenizers you have to match by decoded string with a round-trip
  check in both directions — that is what makes a Llama×Qwen comparison mean
  anything.

### 3. Per-token vectors, then CKA

```
V_l = P · J_l                 rows = per-token J-lens/steering vectors
K_l = center(V_l) center(V_l)ᵀ
CKA(a,b) = <K_a, K_b>_F  with each K normalised
```

Comparing `V_l` rather than `J_l` is the key move: `J_l` lives in an arbitrary
internal basis, while `V_l`'s rows are indexed by tokens, which mean the same
thing in every model. Linear CKA is invariant to rotation and isotropic
scaling, so it compares *structure* and works across models of different
widths.

The relative-depth axis `ρ_l = l/(L−1)` is what lets a 26-layer model be
plotted against a 64-layer one; "matched depth" means equal `ρ`.

### 4. Segment the layer × layer matrix into blocks

Exact dynamic programming for the contiguous segmentation maximising
`Σ_b S_b/|b|` (contiguous kernel k-means, a.k.a. kernel temporal
segmentation), then pick the block count by

```
n_blocks = argmax_k [ blockiness(k) − 0.02·(k−1) ]
```

where `blockiness = mean_within − mean_between` over off-diagonal pairs. The
size normalisation in the objective is not cosmetic: the unnormalised
within-block sum is degenerate at fixed `k` and always prefers shaving
singletons off the ends.

Reported per model: `n_blocks`, `blockiness`, `layer_coupling` (mean
off-diagonal CKA, segmentation-free), and the boundary depths — the first
boundary labelled `sensory_end`, the last `motor_start`.

`jlens/cka.py` in this repo is an independent implementation of all four steps
and reproduces the segmentation rule exactly.

---

## What the campaign actually ran

`program.md` frames it as four axes the paper left open, later extended to
eleven experiments. Contents of `runs/`:

| Experiment | Question | Models |
|---|---|---|
| `E1_horizon` (+`_4k`, `_8k`) | how far forward in tokens does a nudge keep steering? | gemma-3-1b pt/it, qwen3-1.7b/4b, gemma-3-27b-it, olmo3-32b, qwen3.6-27b |
| `E1x2_horizon_training` | does that reach grow during training? | olmo checkpoints |
| `E2_training` | when does the geometry emerge and freeze? | OLMo-3-32B (11 branches), SmolLM3-3B (12), OLMo-7B early |
| `E3_transplant` | is the geometry causally interchangeable across models? | gemma-3-4b↔27b, llama-3.1-8b↔qwen3-8b |
| `E4_delphi` | how does capacity scale with compute? | Delphi ladder 3e18→1e23 FLOPs, isoFLOP bracket, seed replicates |
| `E5_datasets` | does the fitting corpus change the geometry? | ×{wikitext, code, math, finepdfs} |
| `E6_moe`, `E7_inkling` | does it hold for mixture-of-experts? | DeepSeek-V4-Flash, Kimi-K2.5, Inkling (~950B) |
| `E8`/`E9`/`E10_laguna` | quantization and version A/B | Laguna-XS bf16 vs NVFP4 |
| `E11_optimizers` | does the optimizer change the geometry? | 14 runs, 520M and 1.2B, one recipe |

The headline correlational result the program was built on: **matched-depth CKA
of ~0.76 between Llama and OLMo, with depth-correlation ~0.97**. Unrelated
model families lay the same structure out along depth, in the same order, at
the same relative depth. `E3` was the attempt to upgrade that from
correlational to causal by fitting a ridge map between two models' `V_l` and
injecting the mapped vector into the other model.

Two methodological cautions the protocol states about itself, worth carrying
forward: the original analysis pipeline was **lost and reimplemented** from
frozen definitions, so comparisons against the earlier 38-model numbers are
approximate; and the block-segmentation rule was **frozen by fiat** rather than
derived. `docs/03-open-problems.md` takes that second point seriously and tests
how much the pictures depend on it.

---

## Reproducing the pipeline here

`jlens/cka.py` gives you all of it against any `LensModel`:

```python
from jlens.cka import build_probe, cka_matrix, segment_blocks

P, ids = build_probe(model)                 # folds in the final-norm gain
C      = cka_matrix(lens, P)                # layer × layer
blocks = segment_blocks(C, lens.source_layers, model.n_layers)
print(blocks.n_blocks, blocks.blockiness, blocks.boundaries_rho)
```

and cross-model:

```python
from jlens.cka import cross_model_cka, shared_token_ids
shared = shared_token_ids(tok_a, tok_b)                 # match by string
P_a, _ = build_probe(model_a, shared["ids_a"])
P_b, _ = build_probe(model_b, shared["ids_b"])
result = cross_model_cka(lens_a, P_a, lens_b, P_b)
print(result["matched_mean"], result["depth_corr"])     # the 0.76 / 0.97 numbers
```

`experiments/e_shapes.py` runs the segmentation analysis over every CKA matrix
in a clone of `open-jlens-data`, which is how the findings in
`docs/03-open-problems.md` were produced.
