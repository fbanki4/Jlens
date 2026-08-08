"""The bottleneck: does an N+1th concept evict an earlier one?

The capacity claim in global workspace theory is not "the workspace is small".
It is that the workspace is a *bottleneck* — contents compete, and adding one
more thing costs you something already in there. Those are different claims and
only the second is interesting, because any measurement has a floor and will
look "limited" if you squint.

So the test is not "how many concepts can I read at once". It is: hold ``n``
concepts, then hold ``n + 1``, and check whether the *earlier* ones get worse.
A system with no bottleneck degrades gracefully or not at all. A system with a
bottleneck shows the first item's readability falling as the list grows, and
the fall is what capacity means.

Everything here is continuous. There is no "is it in the top k" threshold,
because thresholded scores manufacture sharp capacity limits out of smooth
decay — the same failure mode that manufactures emergent abilities.

Two curves come out:

``retention``   readability of the *first* item as list length grows.
                Flat = no bottleneck. Falling = eviction.
``occupancy``   readability of *all* items at each length, which tells you
                whether the loss is uniform (everything dims) or selective
                (early items evicted, late ones fine — the primacy/recency
                signature).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@dataclass
class CapacityCurve:
    """Readability of list items as a function of list length.

    Attributes:
        lengths: List lengths tested.
        first_item_rank: Median lens rank of the first item, per length.
        all_item_rank: Median lens rank over all items, per length.
        by_position: ``{length: [median rank of item 0, item 1, ...]}``.
        control_rank: Median rank of words that were never in the list —
            the floor that "readable" is measured against.
    """

    lengths: list[int]
    first_item_rank: np.ndarray
    all_item_rank: np.ndarray
    by_position: dict[int, list[float]]
    control_rank: np.ndarray

    @property
    def eviction_slope(self) -> float:
        """Rate at which the first item's log-rank grows with list length.

        Positive means the first item is pushed out as the list grows, i.e. a
        bottleneck. Zero means adding items costs the earlier ones nothing.
        Log-rank because rank is scale-free and heavy-tailed.
        """
        x = np.asarray(self.lengths, float)
        y = np.log(np.maximum(self.first_item_rank, 1.0))
        ok = np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        return float(np.polyfit(x[ok], y[ok], 1)[0])

    @property
    def selectivity(self) -> float:
        """How much more the first item suffers than the average item.

        Separates a genuine bottleneck (early items evicted) from the whole
        readout simply dimming as the prompt gets longer. Near zero means
        uniform decay and no eviction.
        """
        x = np.asarray(self.lengths, float)
        ya = np.log(np.maximum(self.all_item_rank, 1.0))
        ok = np.isfinite(ya)
        if ok.sum() < 3:
            return float("nan")
        return self.eviction_slope - float(np.polyfit(x[ok], ya[ok], 1)[0])

    def summary(self) -> dict:
        return {
            "lengths": self.lengths,
            "first_item_rank": self.first_item_rank.tolist(),
            "all_item_rank": self.all_item_rank.tolist(),
            "control_rank": self.control_rank.tolist(),
            "eviction_slope": self.eviction_slope,
            "selectivity": self.selectivity,
            "by_position": {str(k): v for k, v in self.by_position.items()},
        }


def _first_token_id(tokenizer, word: str) -> int | None:
    """First token id of ``word``, without a BOS in front of it.

    HuggingFace tokenizers prepend a special token by default, which would
    make every item's "first token" the same id and silently flatten the whole
    capacity curve.
    """
    for kwargs in ({"add_special_tokens": False}, {}):
        try:
            ids = tokenizer.encode(word, **kwargs)
            if ids:
                return int(ids[0])
        except TypeError:
            continue
        except Exception:
            return None
    return None


@torch.no_grad()
def _ranks_at(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    targets: Sequence[str],
    band: Sequence[int],
    *,
    max_seq_len: int = 512,
) -> dict[str, int]:
    """Best (lowest) lens rank of each target's first token over ``band``."""
    lens_logits, _, _ = lens.apply(
        model, prompt, layers=list(band), positions=[-1], max_seq_len=max_seq_len
    )
    first_id = {t: _first_token_id(model.tokenizer, t) for t in targets}
    best = {t: 10**9 for t in targets}
    for layer in band:
        order = lens_logits[layer][0].argsort(descending=True).tolist()
        rank = {tid: i + 1 for i, tid in enumerate(order)}
        for t in targets:
            tid = first_id[t]
            if tid is not None and tid in rank:
                best[t] = min(best[t], rank[tid])
    return best


@torch.no_grad()
def eviction_curve(
    model: LensModel,
    lens: JacobianLens,
    item_pool: Sequence[str],
    *,
    lengths: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16),
    n_trials: int = 20,
    template: str = "Remember these words: {items}. The words were:",
    separator: str = ", ",
    band: Sequence[int] | None = None,
    seed: int = 0,
    max_seq_len: int = 512,
) -> CapacityCurve:
    """Measure eviction: does a longer list push out the earliest item?

    Args:
        item_pool: Words to sample lists from. Should be mutually unrelated —
            related items can be held by a single shared direction, which
            inflates apparent capacity.
        lengths: List lengths to test.
        template: Must contain ``{items}``. The readout is taken at the final
            position, where the model is poised to start reproducing the list.
        band: Layers to read over. Defaults to all fitted layers.

    Returns:
        A :class:`CapacityCurve`.
    """
    import random

    rng = random.Random(seed)
    band = list(band if band is not None else lens.source_layers)
    lengths = list(lengths)

    first_ranks: dict[int, list[float]] = {n: [] for n in lengths}
    all_ranks: dict[int, list[float]] = {n: [] for n in lengths}
    pos_ranks: dict[int, list[list[float]]] = {n: [] for n in lengths}
    ctrl_ranks: dict[int, list[float]] = {n: [] for n in lengths}

    for _ in range(n_trials):
        # One shuffled pool per trial, so the same item appears at the same
        # position across lengths: the only thing changing is list length.
        order = rng.sample(list(item_pool), min(max(lengths), len(item_pool)))
        controls = [w for w in item_pool if w not in order]
        for n in lengths:
            if n > len(order):
                continue
            items = order[:n]
            prompt = template.format(items=separator.join(items))
            probes = list(items) + rng.sample(controls, min(5, len(controls)))
            r = _ranks_at(model, lens, prompt, probes, band, max_seq_len=max_seq_len)
            item_r = [float(r[w]) for w in items]
            first_ranks[n].append(item_r[0])
            all_ranks[n].extend(item_r)
            pos_ranks[n].append(item_r)
            ctrl_ranks[n].extend(
                float(r[w]) for w in probes[n:] if w in r
            )

    def med(d, n):
        v = [x for x in d[n] if np.isfinite(x) and x < 10**9]
        return float(np.median(v)) if v else float("nan")

    by_position = {}
    for n in lengths:
        if pos_ranks[n]:
            arr = np.array(pos_ranks[n], dtype=float)
            by_position[n] = np.median(arr, axis=0).tolist()

    return CapacityCurve(
        lengths=lengths,
        first_item_rank=np.array([med(first_ranks, n) for n in lengths]),
        all_item_rank=np.array([med(all_ranks, n) for n in lengths]),
        by_position=by_position,
        control_rank=np.array([med(ctrl_ranks, n) for n in lengths]),
    )
