"""Geometry of the J-lens across layers: probes, CKA, and block segmentation.

This is the machinery behind the layer x layer pictures — the "J-space shapes".
The pipeline is short:

1. **Probe.** ``P = W_U[ids] * gamma_final``: a sample of unembedding rows with
   the final-norm gain folded in. Sampling a few thousand token ids is within
   ~0.005 CKA of using the whole vocabulary, and makes everything tractable.
2. **Per-token vectors.** ``V_l = P @ J_l``. Row ``t`` is the direction at
   layer ``l`` associated with token ``t``. This is the object whose geometry
   we compare — not ``J_l`` itself, because ``J_l`` lives in an arbitrary
   internal basis while ``V_l`` is indexed by something meaningful.
3. **Similarity.** Linear CKA between ``V_a`` and ``V_b`` asks: do these two
   layers organise the *same tokens* in the same relative arrangement? It is
   invariant to rotation and isotropic scaling, so it compares structure rather
   than coordinates, and it works across models of different widths.
4. **Segmentation.** The layer x layer CKA matrix is blocky. Cut it into
   contiguous blocks by exact dynamic programming and report where the cuts land.

What the resulting blocks *mean* is not settled — see ``docs/shapes.md``. This
module's job is to compute them deterministically so the question can be asked.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel

N_PROBE_IDS = 4096
PROBE_SEED = 0
#: An extra block must buy at least this much blockiness to be selected.
PENALTY_PER_BLOCK = 0.02
K_MAX_CAP = 10


# ----------------------------------------------------------------------------
# Probe
# ----------------------------------------------------------------------------


def sample_probe_ids(vocab_size: int, n: int = N_PROBE_IDS, seed: int = PROBE_SEED):
    """Deterministic uniform sample of token ids, without replacement.

    Frozen on ``(vocab_size, n, seed)`` so two models sharing a tokenizer get
    the *same* rows and their ``V_l`` matrices are directly comparable.
    """
    rs = np.random.RandomState(seed)
    n = min(n, vocab_size)
    return np.sort(rs.choice(vocab_size, size=n, replace=False)).astype(np.int64)


@torch.no_grad()
def build_probe(
    model: LensModel, token_ids: np.ndarray | torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``P`` by pushing the identity through the model's own unembedding.

    Rather than hunting for ``lm_head.weight`` and ``model.norm.weight`` under
    whatever names an architecture happens to use, feed ``unembed`` the
    identity matrix. For an RMSNorm model this returns exactly
    ``sqrt(d) * W_U @ diag(gamma)`` — the probe the analysis wants, up to one
    global scalar that CKA and rank-based readouts are both blind to. (For a
    centering LayerNorm it is an approximation, since the mean-subtraction is
    not diagonal; pass explicit weights if that matters.)

    Folding in the final-norm gain is not cosmetic: dropping it degenerates the
    geometry on a large fraction of open models, because ``gamma`` is often
    wildly anisotropic.

    Returns:
        ``(P [n_ids, d_model], token_ids)``.
    """
    device = next(model.layers[0].parameters()).device
    d = model.d_model
    W = model.unembed(torch.eye(d, device=device)).float().cpu().T  # [vocab, d]
    if token_ids is None:
        token_ids = sample_probe_ids(W.shape[0])
    ids = torch.as_tensor(np.asarray(token_ids), dtype=torch.long)
    return W[ids].contiguous(), ids


def lens_vectors(P: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """``V_l = P @ J_l`` — the per-token J-lens vectors at one layer."""
    return (P.float() @ J.float()).contiguous()


def all_lens_vectors(lens: JacobianLens, P: torch.Tensor) -> dict[int, torch.Tensor]:
    return {l: lens_vectors(P, J) for l, J in lens.jacobians.items()}


# ----------------------------------------------------------------------------
# CKA
# ----------------------------------------------------------------------------


def _center(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA over matched rows.

    ``CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)``, equal to
    gram-space CKA on ``K = Xc Xc^T``. Rows are the samples (probe tokens), so
    ``X`` and ``Y`` may have different widths — that is what lets this compare
    a 2048-wide model to a 5120-wide one.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row counts differ: {X.shape[0]} vs {Y.shape[0]}")
    Xc, Yc = _center(X.double()), _center(Y.double())
    num = (Xc.T @ Yc).square().sum()
    den = (Xc.T @ Xc).norm() * (Yc.T @ Yc).norm()
    return float(num / den) if den > 0 else float("nan")


def _normalized_gram(X: torch.Tensor) -> torch.Tensor:
    Xc = _center(X.float())
    K = Xc @ Xc.T
    return K / torch.linalg.norm(K).clamp_min(1e-30)


def cka_matrix(lens: JacobianLens, P: torch.Tensor) -> np.ndarray:
    """Layer x layer linear-CKA matrix for one model.

    Streams one layer at a time into a normalised gram matrix, so peak memory
    is ``n_layers * n_ids^2 * 4`` bytes rather than the ``V`` matrices.
    """
    grams = [_normalized_gram(lens_vectors(P, J)) for _, J in sorted(lens.jacobians.items())]
    L = len(grams)
    C = np.eye(L)
    for a in range(L):
        Ka = grams[a].double()
        for b in range(a + 1, L):
            C[a, b] = C[b, a] = float((Ka * grams[b].double()).sum())
    return C


def relative_depth(lens: JacobianLens, n_layers: int | None = None) -> np.ndarray:
    """``rho_l = l / (L - 1)`` for each fitted layer — the axis that makes
    models of different depths comparable."""
    L = n_layers if n_layers is not None else max(lens.source_layers) + 2
    return np.array([l / (L - 1) for l in lens.source_layers], dtype=np.float64)


def matched_depth_pairs(rho_a: np.ndarray, rho_b: np.ndarray) -> list[tuple[int, int]]:
    """Pair layers of two models at equal relative depth (nearest-rho, driven
    by whichever model has fewer fitted layers)."""
    if len(rho_a) <= len(rho_b):
        return [(i, int(np.argmin(np.abs(rho_b - r)))) for i, r in enumerate(rho_a)]
    return [(int(np.argmin(np.abs(rho_a - r))), j) for j, r in enumerate(rho_b)]


def cross_model_cka(
    lens_a: JacobianLens,
    P_a: torch.Tensor,
    lens_b: JacobianLens,
    P_b: torch.Tensor,
    *,
    n_layers_a: int | None = None,
    n_layers_b: int | None = None,
) -> dict:
    """Compare two models' J-lens geometry.

    Rows of ``P_a`` and ``P_b`` must correspond to the same tokens (same
    tokenizer, or matched by string — see :func:`shared_token_ids`).

    Returns a dict with the matched-depth curve, its mean, the full cross
    matrix, and ``depth_corr``: the correlation between a layer's depth in A
    and the depth of its best match in B. A high ``depth_corr`` says the two
    models lay the same structure out along depth in the same order.
    """
    rho_a = relative_depth(lens_a, n_layers_a)
    rho_b = relative_depth(lens_b, n_layers_b)
    grams_a = [_normalized_gram(lens_vectors(P_a, J)) for _, J in sorted(lens_a.jacobians.items())]
    grams_b = [_normalized_gram(lens_vectors(P_b, J)) for _, J in sorted(lens_b.jacobians.items())]

    pairs = matched_depth_pairs(rho_a, rho_b)
    matched = np.array(
        [float((grams_a[i].double() * grams_b[j].double()).sum()) for i, j in pairs]
    )
    cross = np.zeros((len(grams_a), len(grams_b)))
    for a, Ka in enumerate(grams_a):
        Kad = Ka.double()
        for b, Kb in enumerate(grams_b):
            cross[a, b] = float((Kad * Kb.double()).sum())
    best_rho_b = rho_b[np.argmax(cross, axis=1)]
    return {
        "matched_rho": np.array(
            [rho_a[i] if len(rho_a) <= len(rho_b) else rho_b[j] for i, j in pairs]
        ),
        "matched_cka": matched,
        "matched_mean": float(matched.mean()),
        "cross": cross,
        "depth_corr": float(np.corrcoef(rho_a, best_rho_b)[0, 1]),
    }


def shared_token_ids(tok_a, tok_b, n: int = N_PROBE_IDS, seed: int = PROBE_SEED) -> dict:
    """Token ids that mean the same *string* in two different tokenizers.

    Round-trips each candidate through both tokenizers and keeps only ids that
    survive as a single token in each, so the two probes' rows align one-to-one
    by string. Without this, cross-family comparison is meaningless.
    """
    rs = np.random.RandomState(seed)
    ids_a, ids_b, strings = [], [], []
    for id_a in rs.permutation(len(tok_a)):
        id_a = int(id_a)
        try:
            s = tok_a.decode([id_a])
        except Exception:
            continue
        if not s or "�" in s:
            continue
        if tok_a.encode(s, add_special_tokens=False) != [id_a]:
            continue
        enc_b = tok_b.encode(s, add_special_tokens=False)
        if len(enc_b) != 1 or tok_b.decode([enc_b[0]]) != s:
            continue
        ids_a.append(id_a)
        ids_b.append(int(enc_b[0]))
        strings.append(s)
        if len(ids_a) >= n:
            break
    return {
        "ids_a": np.array(ids_a, dtype=np.int64),
        "ids_b": np.array(ids_b, dtype=np.int64),
        "strings": strings,
    }


# ----------------------------------------------------------------------------
# Block segmentation — "the shapes"
# ----------------------------------------------------------------------------


def _within_sum_table(C: np.ndarray) -> np.ndarray:
    L = C.shape[0]
    P = np.zeros((L + 1, L + 1))
    P[1:, 1:] = np.cumsum(np.cumsum(C, axis=0), axis=1)
    S = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            S[i, j] = P[j + 1, j + 1] - P[i, j + 1] - P[j + 1, i] + P[i, i]
    return S


def best_segmentation(C: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Contiguous segmentation into ``k`` blocks maximising ``sum_b S_b/|b|``.

    Exact dynamic programming over split points. This is contiguous kernel
    k-means (the kernel-temporal-segmentation objective) with ``C`` as the
    kernel: it minimises total within-block scatter.

    The size normalisation is load-bearing. The unnormalised within-block *sum*
    is degenerate at fixed ``k`` — it always prefers shaving singletons off the
    ends, because that cuts the fewest pairs.
    """
    L = C.shape[0]
    if not 1 <= k <= L:
        raise ValueError(f"k={k} out of range for L={L}")
    S = _within_sum_table(C)
    length = (np.arange(L)[None, :] - np.arange(L)[:, None] + 1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(length > 0, S / np.maximum(length, 1), -np.inf)
    best = np.full((k + 1, L), -np.inf)
    prev = np.zeros((k + 1, L), dtype=int)
    best[1, :] = G[0, :]
    for m in range(2, k + 1):
        for j in range(m - 1, L):
            cand = best[m - 1, m - 2 : j] + G[m - 1 : j + 1, j]
            i_rel = int(np.argmax(cand))  # first max -> earlier split on ties
            best[m, j] = cand[i_rel]
            prev[m, j] = i_rel + (m - 1)
    bounds, j = [], L - 1
    for m in range(k, 0, -1):
        i = prev[m, j] if m > 1 else 0
        bounds.append((int(i), int(j)))
        j = i - 1
    return bounds[::-1]


def _block_labels(segments: list[tuple[int, int]], L: int) -> np.ndarray:
    lab = np.zeros(L, dtype=int)
    for b, (i, j) in enumerate(segments):
        lab[i : j + 1] = b
    return lab


def blockiness_of(C: np.ndarray, segments: list[tuple[int, int]]):
    """``(blockiness, mean_within, mean_between)`` over off-diagonal pairs."""
    L = C.shape[0]
    lab = _block_labels(segments, L)
    iu, ju = np.triu_indices(L, k=1)
    same = lab[iu] == lab[ju]
    vals = C[iu, ju]
    mw = float(vals[same].mean()) if same.any() else float("nan")
    mb = float(vals[~same].mean()) if (~same).any() else 0.0
    if not (~same).any():
        return 0.0, mw, float("nan")
    if not same.any():
        return 0.0, float("nan"), mb
    return mw - mb, mw, mb


@dataclass
class BlockResult:
    n_blocks: int
    segments: list[tuple[int, int]]
    block_layers: list[tuple[int, int]]
    boundaries_rho: list[float]
    blockiness: float
    mean_within: float
    mean_between: float
    layer_coupling: float
    selection_curve: dict = field(default_factory=dict)

    @property
    def sensory_end(self) -> float:
        """Relative depth of the first block boundary."""
        return self.boundaries_rho[0] if self.boundaries_rho else float("nan")

    @property
    def motor_start(self) -> float:
        """Relative depth of the last block boundary. Equal to
        :attr:`sensory_end` when only two blocks were selected — in which case
        the segmentation has *not* separately identified a motor band, and
        reading a workspace width off it would be wrong."""
        return self.boundaries_rho[-1] if self.boundaries_rho else float("nan")

    def to_dict(self) -> dict:
        d = {
            "n_blocks": self.n_blocks,
            "segments": self.segments,
            "block_layers": self.block_layers,
            "boundaries_rho": self.boundaries_rho,
            "sensory_end": self.sensory_end,
            "motor_start": self.motor_start,
            "blockiness": self.blockiness,
            "mean_within": self.mean_within,
            "mean_between": self.mean_between,
            "layer_coupling": self.layer_coupling,
            "selection_curve": self.selection_curve,
        }
        return d


def segment_blocks(
    C: np.ndarray,
    source_layers,
    n_layers: int,
    *,
    k: int | None = None,
    penalty_per_block: float = PENALTY_PER_BLOCK,
) -> BlockResult:
    """Segment a CKA matrix and report the frozen block metrics.

    Model selection: pick ``k`` maximising ``blockiness(k) - penalty*(k-1)``.
    A fixed per-block cost rather than an eyeballed elbow, so the answer is
    deterministic and comparable across models and checkpoints. Note that the
    selected ``k`` is sensitive to this penalty — sweep it before believing a
    claim about *how many* regimes a model has.
    """
    C = np.asarray(C, dtype=np.float64)
    L = C.shape[0]
    source_layers = list(source_layers)
    k_max = min(K_MAX_CAP, max(1, int(np.ceil(L / 2))))
    per_k = {}
    for kk in range(1, k_max + 1):
        seg = best_segmentation(C, kk)
        B, mw, mb = blockiness_of(C, seg)
        per_k[kk] = {
            "segments": seg, "blockiness": B, "mean_within": mw, "mean_between": mb,
            "penalized": B - penalty_per_block * (kk - 1),
        }
    if k is None:
        k = min(per_k, key=lambda kk: (-per_k[kk]["penalized"], kk))
    seg = per_k[k]["segments"]
    iu, ju = np.triu_indices(L, k=1)
    return BlockResult(
        n_blocks=int(k),
        segments=seg,
        block_layers=[(source_layers[i], source_layers[j]) for i, j in seg],
        boundaries_rho=[source_layers[i] / (n_layers - 1) for i, _ in seg[1:]],
        blockiness=per_k[k]["blockiness"],
        mean_within=per_k[k]["mean_within"],
        mean_between=per_k[k]["mean_between"],
        layer_coupling=float(C[iu, ju].mean()),
        selection_curve={
            kk: {"blockiness": v["blockiness"], "penalized": v["penalized"]}
            for kk, v in per_k.items()
        },
    )


def save_blocks(result: BlockResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


# ----------------------------------------------------------------------------
# Per-layer geometry summaries
# ----------------------------------------------------------------------------


def effective_dimension(V: torch.Tensor, threshold: float = 0.95) -> float:
    """Fraction of dimensions needed to hold ``threshold`` of ``V``'s variance.

    Low early (the lens vectors collapse into a small subspace), rising sharply
    where the workspace begins.
    """
    s = torch.linalg.svdvals(_center(V.float())).pow(2)
    cumulative = s.cumsum(0) / s.sum().clamp_min(1e-30)
    return float((cumulative < threshold).sum() + 1) / len(s)


def readout_kurtosis(V: torch.Tensor, h: torch.Tensor) -> float:
    """Excess kurtosis of a lens readout distribution.

    Near zero when the readout is diffuse Gaussian noise; high when it is
    sharply peaked on a few tokens. Used as the "is there anything in here at
    all" signal at the start of the workspace band.
    """
    scores = (V.float() @ h.float()).double()
    z = (scores - scores.mean()) / scores.std().clamp_min(1e-12)
    return float(z.pow(4).mean() - 3.0)


def geometry_summary(lens: JacobianLens, P: torch.Tensor, n_layers: int) -> dict:
    """Per-layer geometry: effective dimension, vector norm scale, and the
    participation ratio of the dictionary."""
    from jlens.jspace import participation_ratio

    rows = []
    for layer in lens.source_layers:
        V = lens_vectors(P, lens.jacobians[layer])
        rows.append(
            {
                "layer": layer,
                "rho": layer / (n_layers - 1),
                "effective_dim_95": effective_dimension(V, 0.95),
                "effective_dim_99": effective_dimension(V, 0.99),
                "participation_ratio": participation_ratio(V),
                "mean_norm": float(V.norm(dim=1).mean()),
                "jacobian_fro": float(lens.jacobians[layer].norm()),
            }
        )
    return {"layers": rows, "n_layers": n_layers}
