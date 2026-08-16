"""Build a J-space atlas from checkpointed pretraining runs.

A recreation of what `eliebak.com/viz/jspace-open` shows, pointed at models
with dense checkpoints rather than at released final weights.

The site's frontend was never published — `eliebak/open-jlens-data` is
explicitly "data and methods only" — so this is rebuilt from the schema and the
frozen analysis rules, both of which are public. Output matches the campaign's
layout exactly (``cka_<label>.npz`` + ``<label>_blocks.json``), so runs from
here and lenses from there can sit in one viewer.

Why checkpoints change the question
-----------------------------------
The campaign measured mostly finished models: one lens each, plus eleven
OLMo-3-32B branches and twelve SmolLM3 ones where a family happened to publish
them. That answers "do different models share a geometry" well, and "when does
the geometry appear" barely at all — a dozen scattered snapshots of one run.

Pythia is the other shape of data. 154 checkpoints per size, eight sizes, the
same data in the same order at every size, one tokenizer throughout. That turns
the training question into a grid, and the three confounds that would otherwise
sink it — different data, different tokenizer, different schedule — are gone by
construction rather than by argument.

Any family with published checkpoints works: the checkpoint list is discovered
from the repo's refs, not hardcoded (see :mod:`jlens.checkpoints`).

Cost, honestly
--------------
This fits a lens per checkpoint, which costs about ``n_prompts x d_model``
sequence-backward passes each. That is minutes per checkpoint on a GPU and
hours on CPU, so a wide sweep wants a GPU. Start with ``--sizes 70m --limit 8``
to see the shape of the curve before spending anything.

Usage::

    pip install torch transformers

    # a small real sweep
    python experiments/jspace_open.py --family pythia --sizes 70m,160m \\
        --limit 10 --purge --out runs/atlas

    # view a clone of the open campaign with the same tool, no GPU needed
    python experiments/jspace_open.py --campaign path/to/open-jlens-data/runs \\
        --out runs/atlas

    # both in one page
    python experiments/jspace_open.py --family pythia --sizes 70m \\
        --campaign path/to/open-jlens-data/runs --out runs/atlas
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def load_checkpoint(repo: str, revision: str, dtype: str, device: str):
    """Load one (repo, revision) pair. Pythia and OLMo are ungated: no account,
    no token, no licence click-through."""
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, dtype=getattr(torch, dtype)
    ).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(repo, revision=revision)
    return hf, tok


def purge_cache(repo: str, revision: str) -> None:
    """Drop one revision's snapshot, so peak disk stays at one checkpoint."""
    import shutil

    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception:
        return
    for r in cache.repos:
        if r.repo_id != repo:
            continue
        for rev in r.revisions:
            if revision in rev.refs:
                shutil.rmtree(rev.snapshot_path, ignore_errors=True)


def measure_checkpoint(repo, cp, family, size, budget, args) -> tuple:
    """Fit and measure one checkpoint. Returns ``(entry, C)``."""
    import jlens
    from jlens.atlas import AtlasEntry, measure
    from jlens.lenscontrol import lens_fidelity, matched_depth_layers

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() \
        else ("cpu" if args.device == "auto" else args.device)
    hf, tok = load_checkpoint(repo, cp.revision, args.dtype, device)
    model = jlens.from_hf(hf, tok)
    prompts = fitting_corpus(budget.n_prompts)
    layers, _ = matched_depth_layers(model.n_layers, budget.n_probe_layers)

    lens = jlens.fit(
        model, prompts, source_layers=layers, dim_batch=budget.dim_batch,
        max_seq_len=budget.max_seq_len, skip_first=budget.skip_first,
        checkpoint_path=None, log_every=0,
    )
    C, blocks = measure(model, lens, n_probe_tokens=budget.n_probe_tokens)
    fid = lens_fidelity(model, lens, prompts, max_seq_len=budget.max_seq_len)

    n_layers, d_model, params = family.arch(size)
    warnings = []
    if not (fid.advantage > 0):
        warnings.append(
            f"lens advantage over the logit lens is {fid.advantage:.3f}: the "
            f"geometry here describes the instrument, not the model"
        )
    entry = AtlasEntry(
        label=cp.label, family=family.name, size=size, repo=repo,
        n_layers=model.n_layers, d_model=model.d_model, params=params,
        step=cp.step, tokens=cp.tokens, compute_flops=cp.compute_flops(params),
        stage=cp.stage, variant=cp.variant,
        rho=[l / max(model.n_layers - 1, 1) for l in lens.source_layers],
        source_layers=list(lens.source_layers),
        blocks=blocks, fidelity=fid.summary(),
        budget_fingerprint=budget.fingerprint(), warnings=warnings,
    )
    del hf, model, lens
    if device == "cuda":
        torch.cuda.empty_cache()
    if args.purge:
        purge_cache(repo, cp.revision)
    return entry, C


#: Fitting corpus, hardcoded rather than downloaded. Every checkpoint must be
#: fitted on identical text: a corpus that varies across the sweep makes lens
#: quality vary across the sweep, which is the confound the whole exercise is
#: trying to avoid. The campaign used 1000 wikitext prompts; this is smaller and
#: fixed, and the budget fingerprint records which was used.
SENTENCES = [
    "The committee met on Tuesday to review the proposal in detail.",
    "Water boils at a lower temperature as altitude increases.",
    "She put the letter down and stared out of the kitchen window.",
    "Trade between the two ports declined sharply after the treaty.",
    "The compiler reported an error on the third line of the file.",
    "Most of the birds had already left before the first frost.",
    "He argued that the evidence supported a rather different reading.",
    "Rain fell steadily through the afternoon and into the evening.",
    "The bridge was built in the last century and rebuilt twice since.",
    "Prices rose again in the spring, though not by very much.",
    "A small crowd had gathered near the entrance to the station.",
    "The results were consistent across every group they tested.",
    "Nobody remembered who had suggested the idea in the first place.",
    "Copper conducts electricity better than iron but costs more.",
    "The novel was translated into eleven languages within a decade.",
    "Their argument turned on a single word in the second paragraph.",
]


def fitting_corpus(n_prompts: int, sentences_per_prompt: int = 6) -> list[str]:
    """Deterministic paragraphs — a pure function of ``n_prompts``."""
    return [
        " ".join(SENTENCES[(i * 5 + j) % len(SENTENCES)]
                 for j in range(sentences_per_prompt))
        for i in range(n_prompts)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default=None,
                    help="pythia, olmo2, smollm3, amber — or omit to only "
                         "render an existing atlas / campaign clone")
    ap.add_argument("--sizes", default="70m")
    ap.add_argument("--steps", default=None,
                    help="comma-separated steps to keep; default is the "
                         "log-spaced PYTHIA_STEPS set")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep at most this many checkpoints, evenly spaced, "
                         "endpoints always included")
    ap.add_argument("--campaign", default=None,
                    help="path to a clone of open-jlens-data/runs, to merge in")
    ap.add_argument("--out", default="runs/atlas")
    ap.add_argument("--title", default="J-space atlas")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--purge", action="store_true")

    b = ap.add_argument_group("fitting budget", "identical across every rung")
    b.add_argument("--n-prompts", type=int, default=16)
    b.add_argument("--max-seq-len", type=int, default=128)
    b.add_argument("--skip-first", type=int, default=16)
    b.add_argument("--probe-layers", type=int, default=5,
                   help="set from the SHALLOWEST model in the sweep")
    b.add_argument("--probe-tokens", type=int, default=4096,
                   help="the campaign used 4096, measured as within +/-0.005 "
                        "CKA of the full vocabulary")
    b.add_argument("--dim-batch", type=int, default=8)
    args = ap.parse_args()

    from jlens.atlas import Atlas, from_campaign, write_entry
    from jlens.atlas_vis import collect_matrices, write_page
    from jlens.checkpoints import FAMILIES, PYTHIA_STEPS, describe, discover
    from jlens.lenscontrol import FittingBudget

    os.makedirs(args.out, exist_ok=True)
    budget = FittingBudget(
        n_prompts=args.n_prompts, max_seq_len=args.max_seq_len,
        skip_first=args.skip_first, n_probe_layers=args.probe_layers,
        n_probe_tokens=args.probe_tokens, dim_batch=args.dim_batch,
    )

    atlas_path = os.path.join(args.out, "atlas.json")
    atlas = Atlas.load(atlas_path) if os.path.exists(atlas_path) else Atlas()
    atlas.title = args.title
    atlas.budget = budget.describe()
    done = {e.key for e in atlas.entries}

    if args.family:
        family = FAMILIES.get(args.family)
        if family is None:
            raise SystemExit(
                f"unknown family {args.family!r}; known: {sorted(FAMILIES)}"
            )
        keep = ([int(s) for s in args.steps.split(",")] if args.steps
                else (PYTHIA_STEPS if args.family == "pythia" else None))
        for size in args.sizes.split(","):
            repo = family.repo(size)
            cps = discover(repo, tokens_per_step=family.tokens_per_step,
                           keep_steps=keep, limit=args.limit)
            print(f"{repo}: {len(cps)} checkpoints "
                  f"({cps[0].label} .. {cps[-1].label})" if cps
                  else f"{repo}: no checkpoint refs found")
            meta_path = os.path.join(args.out, f"{family.name}-{size}.json")
            with open(meta_path, "w") as f:
                json.dump(describe(family, size, cps), f, indent=2)

            for cp in cps:
                key = f"{family.name}/{size}/{cp.label}"
                if key in done:
                    continue
                try:
                    entry, C = measure_checkpoint(
                        repo, cp, family, size, budget, args)
                except Exception as exc:
                    print(f"  {cp.label}: FAILED {type(exc).__name__}: {exc}")
                    continue
                sub = os.path.join(args.out, family.name, size)
                npz_path, _ = write_entry(
                    sub, cp.label, C, entry.source_layers, entry.n_layers,
                    entry.d_model, f"{repo} {cp.label}", entry.blocks)
                entry.npz_path = os.path.abspath(npz_path)
                atlas.add(entry)
                atlas.save(args.out)
                adv = entry.fidelity.get("advantage", float("nan"))
                flag = "  <- lens dead" if not (adv > 0) else ""
                print(f"  {cp.label:>18s}  blocks {entry.blocks['n_blocks']}  "
                      f"B={entry.blocks['blockiness']:.3f}  "
                      f"boundary rho={entry.blocks.get('sensory_end')}  "
                      f"lens {adv:+.3f}{flag}")

    if args.campaign:
        merged = from_campaign(args.campaign)
        added = 0
        for e in merged.entries:
            if e.key not in {x.key for x in atlas.entries}:
                atlas.add(e)
                added += 1
        print(f"merged {added} entries from {args.campaign}")

    if not atlas.entries:
        raise SystemExit(
            "atlas is empty — pass --family to measure checkpoints, or "
            "--campaign to load a clone of open-jlens-data/runs"
        )

    atlas.save(args.out)
    search = [args.out] + ([args.campaign] if args.campaign else [])
    page = write_page(atlas, os.path.join(args.out, "index.html"),
                      search_dirs=search)

    print(f"\n{len(atlas.entries)} entries -> {atlas_path}")
    for w in atlas.manifest().get("warnings", []):
        print(f"  !! {w}")
    print(f"open {page}")

    live = [e for e in atlas.entries if e.lens_is_live and e.blocks]
    nb = [e.blocks.get("n_blocks") for e in live if e.blocks.get("n_blocks")]
    if nb:
        vals, counts = np.unique(nb, return_counts=True)
        print("\nblock counts across the atlas: "
              + ", ".join(f"{v} blocks x{c}" for v, c in zip(vals, counts)))
    bounds = [e.blocks.get("sensory_end") for e in live
              if e.blocks.get("sensory_end") is not None]
    if bounds:
        print(f"first boundary depth: rho {np.min(bounds):.2f}-{np.max(bounds):.2f} "
              f"(median {np.median(bounds):.2f}) over {len(bounds)} models")


if __name__ == "__main__":
    main()
