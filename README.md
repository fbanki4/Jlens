# Jlens

A from-scratch implementation of the **Jacobian lens**, plus a runnable
testbed for investigating four of the open questions it leaves behind.

The J-lens reads out what an internal activation is disposed to make a language
model *say*. It transports a residual-stream vector at any layer into the
final-layer basis using an average input-output Jacobian, then decodes it with
the model's own unembedding:

```
lens_l(h) = unembed( J_l @ h )        J_l = E[ ∂h_final / ∂h_l ]
```

Background: Gurnee et al., *Verbalizable Representations Form a Global
Workspace in Language Models*, Transformer Circuits Thread, 2026.

---

## What's here

```
jlens/
  fitting.py    averaged-Jacobian fitting; also the Δ-resolved variant
  lens.py       JacobianLens: transport, readout, save/load/merge
  jspace.py     sparse non-negative decomposition, occupancy, steer/ablate/swap
  cka.py        probe, V_l = P·J_l, linear CKA, block segmentation
  phrases.py    multi-token concepts without training anything
  ignition.py   the ambiguity sweep — a workspace measure that uses no lens
  broadcast.py  causal broadcast breadth, against matched-norm noise
  capacity.py   the bottleneck: does an N+1th item evict the first?
  scaling.py    smooth vs broken power law, with the breakpoint search priced in
  lenscontrol.py  keeps a per-model instrument out of a cross-model result
  checkpoints.py  discovers a pretraining run's checkpoints from its repo refs
  atlas.py      the open campaign's data layout, written and read
  atlas_vis.py  the atlas viewer — gallery, detail, training
  vis.py        self-contained layer × position slice viewer
  hf.py         HuggingFace adapter — run all of the above on a real open model
  toy/          a relational world + trainable small transformer (CPU, no downloads)

experiments/
  fit_toy_lens.py     fit a lens on a toy checkpoint
  e_multihop.py       does the lens surface an unverbalized intermediate, and steer?
  e_phrase.py         untrained vs trained multi-token concept vectors
  e_form.py           when does the J-space form, relative to the capability?
  e_shapes.py         what do the block "shapes" indicate? (real open-model data)
  e_gwt_signatures.py broadcast breadth and the bottleneck
  pythia_sweep.py     where does it emerge? lens-free, then lens, then controlled
  jspace_open.py      build + view a J-space atlas across checkpoints
  run_all.py          everything, end to end

docs/
  01-coding-a-j-lens.md                  how it works and why each choice matters
  02-how-the-open-model-viz-was-made.md  the open-model measurement pipeline
  03-open-problems.md                    the four investigations and their results
  04-running-it-yourself.md              the scale sweep, on a laptop and beyond
```

## On a real open model

```python
import transformers, jlens

hf  = transformers.AutoModelForCausalLM.from_pretrained(
          "Qwen/Qwen3-1.7B", dtype="bfloat16", device_map="cuda")
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = jlens.from_hf(hf, tok)

lens = jlens.fit(model, prompts, dim_batch=64, checkpoint_path="out/ckpt.pt")
lens.save("out/lens.pt")

print(lens.readout(model, "The capital of the country shaped like a boot is",
                   layer=12))
```

Cost is `params × d_model × n_prompts`: a 1–2B model with a few hundred
prompts is a single-GPU afternoon. Shard over prompts and combine with
`JacobianLens.merge()` for anything larger.

## Without a GPU

```bash
pip install -e .              # installs torch/numpy/scipy, makes `jlens` importable
python -m experiments.run_all
```

Trains a 1.4M-parameter transformer on a synthetic relational world, fits a
lens at every training checkpoint, and runs all four investigations. About
half an hour on four CPU cores.

Install the package rather than just its dependencies: without it, `python
experiments/<script>.py` puts `experiments/` on the import path instead of the
repo root and fails with `No module named 'jlens'`. `python -m experiments.…`
works either way, but only from the repo root.

## The J-space atlas

Recreates what `eliebak.com/viz/jspace-open` shows. To browse the open
campaign's 49 fitted lenses — no GPU, no model downloads:

```bash
git clone https://github.com/eliebak/open-jlens-data
python experiments/jspace_open.py --campaign open-jlens-data/runs --out runs/atlas
open runs/atlas/index.html
```

To measure your own checkpoints into the same page:

```bash
pip install -e ".[hf]"
python experiments/jspace_open.py --family pythia --sizes 70m --limit 8 --purge
```

The toy world exists so that **ground truth is available**. Its two-hop items
state some `person lives in city` and `city sits in country` facts in shuffled
order and then ask `<subject> hails from`; the city is a latent intermediate
that appears in neither the query nor the answer. Half the city names are two
words, so half the latents are genuinely multi-token concepts with no J-lens
vector of their own — the `blackmail`/`photosynthesis` problem, reproduced
where the answer is known.

## Results

See [`docs/03-open-problems.md`](docs/03-open-problems.md).

## Credits

The estimator, the `skip_first` convention, the checkpoint format and the
block-segmentation rule follow two public sources, both Apache-2.0:
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (the
paper's reference implementation) and
[`eliebak/open-jlens-data`](https://github.com/eliebak/open-jlens-data) (the
open-weight measurement campaign, whose CKA matrices `e_shapes.py` analyses).
The code here is written independently and is file-format compatible with the
reference.
