"""The data behind a J-space atlas, in the format the open campaign used.

`eliebak.com/viz/jspace-open` renders J-lens geometry across open-weight
models. Its frontend is not published, but everything it displays is: the
`eliebak/open-jlens-data` campaign ships the arrays and the frozen analysis
rules, and the layout is simple enough to match exactly.

Per fitted lens, two files:

``cka_<label>.npz``      ``C`` (the layer x layer linear-CKA matrix),
                         ``source_layers``, ``rho``, ``n_layers``, ``d_model``,
                         ``model_name``
``<label>_blocks.json``  the frozen block metrics — ``n_blocks``, ``segments``,
                         ``boundaries_rho``, ``blockiness``, ``mean_within``,
                         ``mean_between``, ``layer_coupling``,
                         ``selection_curve``, plus the ``sensory_end`` /
                         ``motor_start`` aliases for the first and last
                         boundary

Matching the layout is not homage. It means a run produced here drops straight
into the campaign's own analysis scripts, and the campaign's 77 fitted lenses
drop straight into the viewer here — the comparison this whole exercise is for.

What this module adds
---------------------
The campaign measured *released* models: one lens per model, checkpoints where
a family happened to publish them. Its `E2_training` run had eleven OLMo-3-32B
branches and twelve SmolLM3 ones. Pythia has 154 per size at eight sizes, with
identical data order — so the same pipeline pointed at Pythia turns a handful
of scattered snapshots into a dense grid over (scale x training).

That is what :class:`Atlas` is for: it carries the training axis (step, tokens,
compute) alongside each entry, so the geometry can be plotted against the thing
that produced it rather than against an arbitrary checkpoint index.

**The instrument still has to be reported.** Every entry carries its
:class:`~jlens.lenscontrol.FittingBudget` fingerprint and its lens fidelity.
A CKA matrix from a checkpoint whose lens is no better than the logit lens is a
picture of the instrument, and the viewer marks it as such rather than letting
it sit in the grid looking like a result.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from jlens.checkpoints import _UNITS as _TOKEN_UNITS
from jlens.checkpoints import parse_ref
from jlens.cka import build_probe, cka_matrix, segment_blocks
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


def json_safe(obj):
    """Recursively replace non-finite floats with ``None``.

    ``json.dump`` writes bare ``NaN`` and ``Infinity``, which are not valid
    JSON and which ``JSON.parse`` rejects outright — so a single NaN anywhere
    in an atlas silently breaks the whole viewer rather than blanking one cell.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


@dataclass
class AtlasEntry:
    """One fitted lens, measured and placed on the training axes.

    Attributes:
        label: File stem and display name — the checkpoint's ref name. **Not
            unique across an atlas**: two families routinely publish the same
            ref name (OLMo-7B and OLMo-32B both have ``stage1-step1000``). Use
            :attr:`key` to identify an entry; ``label`` is for display and for
            the filename, where the enclosing directory disambiguates.
        npz_path: Where this entry's CKA matrix lives, when known. Recorded so
            the viewer can find the matrix by path instead of by globbing for
            the label, which is what made colliding labels share a thumbnail.
        blocks: The frozen block metrics, as written to ``<label>_blocks.json``.
        fidelity: Lens quality for this entry. ``advantage`` at or below zero
            means the fitted transport bought nothing here.
        budget_fingerprint: Which fitting budget produced it. Entries with
            different fingerprints were measured differently and must not be
            compared.
    """

    label: str
    family: str
    size: str
    repo: str
    n_layers: int
    d_model: int
    params: float | None = None
    step: int | None = None
    tokens: int | None = None
    compute_flops: float | None = None
    stage: str | None = None
    variant: str | None = None
    rho: list[float] = field(default_factory=list)
    source_layers: list[int] = field(default_factory=list)
    blocks: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    budget_fingerprint: str = ""
    npz_path: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique identity within an atlas: family, size and ref name."""
        return "/".join(p for p in (self.family, self.size, self.label) if p)

    @property
    def lens_is_live(self) -> bool:
        """Did the fitted transport beat the logit lens?

        The gate on whether this entry's geometry describes the model at all.
        Unknown fidelity counts as live — absence of the check is not evidence
        the lens was dead, and silently dropping unmeasured entries would be
        worse than showing them unmarked.
        """
        adv = self.fidelity.get("advantage")
        return adv is None or (np.isfinite(adv) and adv > 0)

    def summary(self) -> dict:
        return json_safe({
            **asdict(self), "key": self.key, "lens_is_live": self.lens_is_live,
        })


def write_entry(
    out_dir: str,
    label: str,
    C: np.ndarray,
    source_layers: Sequence[int],
    n_layers: int,
    d_model: int,
    model_name: str,
    blocks: dict,
) -> tuple[str, str]:
    """Write the two per-lens files, in the campaign's own layout.

    Returns:
        ``(npz_path, blocks_json_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    source_layers = np.asarray(list(source_layers), dtype=np.int64)
    rho = source_layers / max(n_layers - 1, 1)

    npz_path = os.path.join(out_dir, f"cka_{label}.npz")
    # The campaign's files also carry `ids_path`, an absolute path to the probe
    # ids on its own cluster. It is omitted here rather than faked: it would
    # point at nothing, and the probe is already pinned by (vocab_size, n, seed).
    np.savez(
        npz_path,
        C=np.asarray(C, dtype=np.float64),
        source_layers=source_layers,
        rho=rho.astype(np.float64),
        n_layers=np.int64(n_layers),
        d_model=np.int64(d_model),
        model_name=str(model_name),
    )
    json_path = os.path.join(out_dir, f"{label}_blocks.json")
    with open(json_path, "w") as f:
        json.dump(json_safe(blocks), f, indent=2)
    return npz_path, json_path


def read_entry(npz_path: str) -> dict:
    """Read a ``cka_*.npz`` written here or by the open campaign.

    Deliberately tolerant: campaign files carry an ``ids_path`` this repo does
    not write, and older ones may omit ``rho``, which is recomputed from the
    layer indices when missing.
    """
    d = np.load(npz_path, allow_pickle=True)
    n_layers = int(d["n_layers"])
    source_layers = d["source_layers"].astype(int)
    rho = (d["rho"] if "rho" in d.files
           else source_layers / max(n_layers - 1, 1)).astype(float)
    return {
        "C": d["C"].astype(float),
        "source_layers": source_layers.tolist(),
        "rho": rho.tolist(),
        "n_layers": n_layers,
        "d_model": int(d["d_model"]),
        "model_name": str(d["model_name"]),
    }


def measure(
    model: LensModel,
    lens: JacobianLens,
    *,
    probe_token_ids=None,
    n_probe_tokens: int = 4096,
    penalty_per_block: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """The CKA matrix and the frozen block metrics for one fitted lens.

    Args:
        n_probe_tokens: The campaign used 4096, which it measured as within
            +/-0.005 CKA of the full vocabulary. Lowering it is a speed knob
            that costs comparability, since a different row set is a different
            measurement.

    Returns:
        ``(C, blocks)`` where ``blocks`` is ready to serialise, including the
        ``sensory_end`` / ``motor_start`` aliases the campaign reports.
    """
    P, _ = build_probe(model, probe_token_ids, n_ids=n_probe_tokens)
    C = cka_matrix(lens, P)
    result = segment_blocks(
        C, lens.source_layers, n_layers=model.n_layers,
        penalty_per_block=penalty_per_block,
    )
    blocks = result.to_dict()
    bounds = blocks.get("boundaries_rho") or []
    # The campaign names the first and last boundary; with two blocks they are
    # the same number, which is why it reports both rather than a range.
    blocks["sensory_end"] = bounds[0] if bounds else None
    blocks["motor_start"] = bounds[-1] if bounds else None
    return C, blocks


@dataclass
class Atlas:
    """A collection of measured lenses, plus what is needed to compare them."""

    entries: list[AtlasEntry] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    title: str = "J-space atlas"

    def add(self, entry: AtlasEntry) -> None:
        self.entries.append(entry)

    def manifest(self) -> dict:
        """The single JSON a viewer loads."""
        prints = {e.budget_fingerprint for e in self.entries if e.budget_fingerprint}
        warnings = []
        if len(prints) > 1:
            warnings.append(
                f"entries carry {len(prints)} different budget fingerprints "
                f"({sorted(prints)}); they were measured under different "
                f"settings and must not be compared"
            )
        dead = [e.label for e in self.entries if not e.lens_is_live]
        if dead:
            warnings.append(
                f"{len(dead)} of {len(self.entries)} entries have a lens no "
                f"better than the logit lens; their geometry describes the "
                f"instrument, not the model"
            )
        keys = [e.key for e in self.entries]
        clashes = sorted({k for k in keys if keys.count(k) > 1})
        if clashes:
            warnings.append(
                f"{len(clashes)} duplicate entry key(s) ({clashes[:3]}); "
                f"entries will overwrite each other in the viewer"
            )
        return json_safe({
            "title": self.title,
            "budget": self.budget,
            "n_entries": len(self.entries),
            "warnings": warnings,
            "entries": [e.summary() for e in self.entries],
        })

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "atlas.json")
        with open(path, "w") as f:
            json.dump(self.manifest(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Atlas":
        with open(path) as f:
            data = json.load(f)
        known = set(AtlasEntry.__dataclass_fields__)
        entries = [
            AtlasEntry(**{k: v for k, v in e.items() if k in known})
            for e in data.get("entries", [])
        ]
        return cls(entries=entries, budget=data.get("budget", {}),
                   title=data.get("title", "J-space atlas"))


def from_campaign(runs_dir: str, *, title: str = "open-jlens-data") -> Atlas:
    """Build an :class:`Atlas` from a clone of ``eliebak/open-jlens-data``.

    Lets the campaign's 77 measured lenses be viewed with the same tool as a
    Pythia sweep run here, which is the comparison worth making: their models
    are big and their checkpoints sparse, this repo's are small and dense.

    Entries get no fidelity block, because the campaign did not measure lens
    fidelity — so they are treated as live and their `budget_fingerprint` is
    left empty. That is honest rather than convenient: an unmeasured instrument
    is unknown, not good.
    """
    import glob
    import re

    #: Path components that describe where a file sits, not what it is of.
    _PLUMBING = {"results", "permodel", "fits", "analysis"}

    def group_of(npz_path: str) -> str:
        """Experiment/model path, with the storage plumbing stripped out."""
        rel = os.path.relpath(os.path.dirname(npz_path), runs_dir)
        parts = [p for p in rel.split(os.sep) if p not in _PLUMBING and p != "."]
        return "/".join(parts) or "root"

    atlas = Atlas(title=title)
    for npz_path in sorted(glob.glob(os.path.join(runs_dir, "**", "cka_*.npz"),
                                     recursive=True)):
        try:
            rec = read_entry(npz_path)
        except Exception:
            continue  # crossx_*/shared_ids_* files share the glob, not the schema
        C = rec.get("C")
        if C is None or np.ndim(C) != 2 or C.shape[0] != C.shape[1]:
            continue
        stem = os.path.basename(npz_path)[len("cka_"):-len(".npz")]
        blocks_path = os.path.join(os.path.dirname(npz_path), f"{stem}_blocks.json")
        if os.path.exists(blocks_path):
            with open(blocks_path) as f:
                blocks = json.load(f)
        else:
            # The campaign segmented some matrices and not others. The rule is
            # frozen and deterministic, so recomputing it here yields exactly
            # what they would have written — and an atlas with a hole in it is
            # worse than one filled by the documented rule.
            blocks = segment_blocks(
                C, rec["source_layers"], n_layers=rec["n_layers"]
            ).to_dict()
            blocks["recomputed"] = True
        bounds = blocks.get("boundaries_rho") or []
        blocks.setdefault("sensory_end", bounds[0] if bounds else None)
        blocks.setdefault("motor_start", bounds[-1] if bounds else None)

        # Recover the training axes. The ref name gives the step; the campaign
        # writes its own token accounting into model_name, e.g.
        # "olmo32b stage1-step11000 (92.3B, n=250)" — and its accounting beats
        # anything reconstructed from a step count, since these runs have
        # multi-stage curricula whose tokens-per-step is not constant.
        cp = parse_ref(stem)
        step = cp.step if cp else None
        if step is None:
            m = re.search(r"step(\d+)", stem)
            step = int(m.group(1)) if m else None
        tokens = cp.tokens if cp else None
        if tokens is None:
            m = re.search(r"\(([\d.]+)([KMBT])[,)]", rec["model_name"])
            if m:
                tokens = int(float(m.group(1)) * _TOKEN_UNITS[m.group(2)])

        atlas.add(AtlasEntry(
            label=stem,
            family=group_of(npz_path),
            size="", repo=rec["model_name"],
            n_layers=rec["n_layers"], d_model=rec["d_model"],
            step=step, tokens=tokens,
            stage=cp.stage if cp else None,
            variant=cp.variant if cp else None,
            rho=rec["rho"], source_layers=rec["source_layers"],
            blocks=blocks, npz_path=os.path.abspath(npz_path),
        ))
    return atlas
