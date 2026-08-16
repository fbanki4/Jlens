"""Finding the checkpoints of a pretraining run.

Every open pretraining release publishes its intermediate checkpoints as git
refs on the Hub, and every one of them names those refs differently. Pythia
uses ``step143000``. OLMo-2 uses ``stage1-step140000-tokens587B``. LLM360's
Amber uses ``ckpt_356``. A registry that hardcodes those lists is wrong the
day a new release lands, and silently wrong the day a family renames a branch.

So this module **discovers** them: list the refs the repo actually has, parse
them with a table of patterns, and return what is there. The registry below
carries only the things a ref name cannot tell you — architecture, and how
many tokens a step is worth — and even that is optional, because a run with
unknown token accounting can still be plotted against its step count.

Why token accounting is worth the trouble
-----------------------------------------
The interesting axis is almost never step. Step is a unit of optimizer, not of
learning: pythia-14m and pythia-12b at step 1000 have seen the same *tokens*
but bought wildly different amounts of compute with them. Plotting a scale
sweep against step compares training schedules; plotting against tokens or
against ``6ND`` compares the thing you meant. :meth:`Checkpoint.compute_flops`
is what the collapse test in :mod:`jlens.scaling` consumes.

Pythia is the family this repo defaults to, for three reasons that each remove
a confound: identical data in identical order across all eight sizes, 154
checkpoints per size with ten of them log-spaced over steps 1-512 (which is
where a workspace would be absent, so the resolution sits where the transition
would be), and one tokenizer throughout — so a cross-size claim about lens
geometry is not partly a claim about tokenization.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

#: Token-count suffixes used in checkpoint branch names.
_UNITS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

#: Ref-name patterns, tried in order. The first that matches a ref wins, so
#: the more specific ones (those carrying a token count) come first.
#:
#: Adding a family usually means adding one line here — the rest of the module
#: is family-agnostic.
REF_PATTERNS: tuple[str, ...] = (
    # OLMo-2, SmolLM3: stage1-step140000-tokens587B, and the stage-2 variants
    # that carry an extra descriptor: stage2-ingredient1-step11931-tokens50B.
    # Those descriptors are different training branches at the same step, so
    # they are kept in `variant` rather than discarded.
    r"^(?P<stage>stage\d+)(?P<variant>(?:-[A-Za-z0-9.]+)*?)"
    r"-step(?P<step>\d+)(?:-tokens(?P<tokens>[\d.]+)(?P<unit>[KMBT]))?$",
    # OLMo-1: step1000-tokens4B
    r"^step(?P<step>\d+)-tokens(?P<tokens>[\d.]+)(?P<unit>[KMBT])$",
    # Pythia, Qwen intermediate releases: step143000
    r"^step(?P<step>\d+)$",
    # LLM360 Amber / CrystalCoder: ckpt_356
    r"^ckpt[-_](?P<step>\d+)$",
    # Megatron-style exports: iter_0143000
    r"^iter[-_](?P<step>\d+)$",
    # TinyLlama and friends: 2T, 480B — no step, only a token count
    r"^(?P<tokens>[\d.]+)(?P<unit>[KMBT])$",
)


@dataclass(frozen=True, order=True)
class Checkpoint:
    """One point on a training run.

    Attributes:
        step: Optimizer step, when the ref name carries one.
        revision: The git ref to pass to ``from_pretrained(revision=...)``.
        label: Short display name, and the stem of the files written for it.
        tokens: Tokens seen. Taken from the ref name when it says, otherwise
            derived from ``step`` and the family's ``tokens_per_step``.
        stage: Curriculum stage, for runs that have them (OLMo-2, SmolLM3).
        variant: Extra branch descriptor, where a family publishes several
            branches at the same step (OLMo-2's stage-2 ingredients). Two
            checkpoints sharing a step are different runs, not duplicates.
    """

    sort_key: tuple = field(repr=False)
    revision: str
    label: str
    step: int | None = None
    tokens: int | None = None
    stage: str | None = None
    variant: str | None = None

    def compute_flops(self, params: float | None) -> float | None:
        """The ``6ND`` compute proxy, or ``None`` if either factor is unknown.

        Non-embedding parameters times tokens times six. Approximate by
        construction, and that is fine — it is used to place points on a log
        axis spanning many orders of magnitude, where the constant does not
        change which axis the curves collapse onto.
        """
        if params is None or self.tokens is None:
            return None
        return 6.0 * params * self.tokens

    def summary(self) -> dict:
        return {
            "revision": self.revision, "label": self.label, "step": self.step,
            "tokens": self.tokens, "stage": self.stage,
        }


def parse_ref(name: str, *, tokens_per_step: float | None = None) -> Checkpoint | None:
    """Parse one ref name into a :class:`Checkpoint`, or ``None`` if it is not
    a checkpoint ref at all (``main``, ``refs/pr/3``, a tag, ...).

    Args:
        tokens_per_step: Used only when the ref name does not carry a token
            count itself. A name that states its tokens always wins — it is the
            release's own accounting, and a family-level constant is an
            assumption about a schedule that may have changed mid-run.
    """
    ref = name.rsplit("/", 1)[-1].strip()
    for pattern in REF_PATTERNS:
        m = re.match(pattern, ref)
        if not m:
            continue
        g = m.groupdict()
        step = int(g["step"]) if g.get("step") is not None else None
        tokens = None
        if g.get("tokens") is not None:
            tokens = int(float(g["tokens"]) * _UNITS[g["unit"].upper()])
        elif step is not None and tokens_per_step:
            tokens = int(step * tokens_per_step)
        # Sort by tokens when there is no step, so token-labelled runs
        # (TinyLlama's "2T") still order correctly. The ref name is the final
        # tiebreak, so families publishing several branches at one step get a
        # deterministic order instead of an arbitrary one.
        variant = (g.get("variant") or "").strip("-") or None
        key = ((0, step, ref) if step is not None else (1, tokens or 0, ref))
        return Checkpoint(
            sort_key=key, revision=ref, label=ref, step=step, tokens=tokens,
            stage=g.get("stage"), variant=variant,
        )
    return None


@dataclass(frozen=True)
class Family:
    """A pretraining family, and the parts of it a ref name cannot tell you.

    Attributes:
        repo_template: ``str.format``-ready, with ``{size}``.
        sizes: ``{size: (n_layers, d_model, non_embedding_params)}``.
        tokens_per_step: Fallback token accounting for ref names that omit it.
        note: Anything a user should know before trusting the numbers.
    """

    name: str
    repo_template: str
    sizes: dict[str, tuple[int, int, float]]
    tokens_per_step: float | None = None
    note: str = ""

    def repo(self, size: str) -> str:
        if size not in self.sizes:
            raise KeyError(
                f"{self.name} has no size {size!r}; known: {sorted(self.sizes)}"
            )
        return self.repo_template.format(size=size)

    def arch(self, size: str) -> tuple[int, int, float]:
        return self.sizes[size]


#: Pythia. Steps are 0, 1, 2, 4, ... 512 (log-spaced), then every 1000 to
#: 143000 — 154 per size. Batch is 1024 sequences of 2048 tokens, so a step is
#: 2^21 tokens exactly, and the same data in the same order is used at every
#: size.
PYTHIA = Family(
    name="pythia",
    repo_template="EleutherAI/pythia-{size}",
    sizes={
        "14m": (6, 128, 1.4e7), "31m": (6, 256, 3.1e7), "70m": (6, 512, 7.0e7),
        "160m": (12, 768, 1.6e8), "410m": (24, 1024, 4.1e8),
        "1b": (16, 2048, 1.0e9), "1.4b": (24, 2048, 1.4e9),
        "2.8b": (32, 2560, 2.8e9), "6.9b": (32, 4096, 6.9e9),
        "12b": (36, 5120, 1.18e10),
    },
    tokens_per_step=2_097_152,
    note="ungated; identical data order across sizes; one tokenizer",
)

#: Log-spaced early steps plus every 1000. Handy as a default when you want the
#: shape of a curve rather than all 154 points.
PYTHIA_STEPS: tuple[int, ...] = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000,
)

#: Other families with public intermediate checkpoints. Architecture is filled
#: in where this repo has used it; the rest is left to discovery, which is the
#: point — an entry here is a convenience, not a requirement.
OLMO2 = Family(
    name="olmo2",
    repo_template="allenai/OLMo-2-1124-{size}",
    sizes={"7B": (32, 4096, 7.3e9), "13B": (40, 5120, 1.3e10)},
    note="stage1/stage2 curriculum; refs carry their own token counts",
)

SMOLLM3 = Family(
    name="smollm3",
    repo_template="HuggingFaceTB/SmolLM3-{size}",
    sizes={"3B": (36, 2048, 3.1e9)},
    tokens_per_step=2.37e6,
    note="tokens_per_step from open-jlens-data's E2 meta.json",
)

AMBER = Family(
    name="amber",
    repo_template="LLM360/{size}",
    sizes={"Amber": (32, 4096, 6.7e9)},
    note="360 checkpoints named ckpt_NNN; token accounting not in the ref name",
)

FAMILIES: dict[str, Family] = {
    f.name: f for f in (PYTHIA, OLMO2, SMOLLM3, AMBER)
}


def list_refs(repo: str, *, token: str | None = None) -> list[str]:
    """Every branch name on a Hub repo.

    Separated out so :func:`discover` can be tested without network access —
    and so a caller behind a blocked egress can substitute their own lister.
    """
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(repo, token=token)
    return [b.name for b in refs.branches]


def discover(
    repo: str,
    *,
    tokens_per_step: float | None = None,
    lister=list_refs,
    keep_steps: Sequence[int] | None = None,
    limit: int | None = None,
) -> list[Checkpoint]:
    """Every checkpoint a repo actually publishes, in training order.

    Discovery rather than a hardcoded list, because checkpoint naming is a
    per-release decision and a stale table fails silently — it yields a short
    sweep that looks complete.

    Args:
        keep_steps: Keep only these steps. Steps that the repo does not have
            are dropped rather than raising, since checkpoint availability
            varies across sizes within a family.
        limit: Keep at most this many, evenly spaced across the run. Applied
            after ``keep_steps``. Endpoints are always kept — the start and end
            of training are the two points you cannot do without.

    Returns:
        Sorted checkpoints. Empty if the repo publishes no parseable refs,
        which usually means it ships only ``main``.
    """
    found = [
        cp for cp in (
            parse_ref(name, tokens_per_step=tokens_per_step)
            for name in lister(repo)
        ) if cp is not None
    ]
    found.sort()
    if keep_steps is not None:
        wanted = set(keep_steps)
        found = [cp for cp in found if cp.step in wanted]
    if limit is not None and 0 < limit < len(found):
        idx = sorted({round(i * (len(found) - 1) / (limit - 1))
                      for i in range(limit)}) if limit > 1 else [0]
        found = [found[i] for i in idx]
    return found


def describe(family: str | Family, size: str, checkpoints: Sequence[Checkpoint]) -> dict:
    """Metadata block for an atlas entry: what was measured, and on what axes."""
    fam = FAMILIES[family] if isinstance(family, str) else family
    n_layers, d_model, params = fam.arch(size)
    return {
        "family": fam.name,
        "size": size,
        "repo": fam.repo(size),
        "n_layers": n_layers,
        "d_model": d_model,
        "params": params,
        "note": fam.note,
        "n_checkpoints": len(checkpoints),
        "checkpoints": [
            {**cp.summary(), "compute_flops": cp.compute_flops(params)}
            for cp in checkpoints
        ],
    }
