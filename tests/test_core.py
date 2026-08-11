"""Correctness tests for the pieces where a bug would be silent.

Run with ``python -m tests.test_core`` (no pytest required).

The J-lens fails quietly when it is wrong: a subtly mis-estimated ``J_l`` still
produces a plausible-looking ranked word list. So the tests here pin the
properties that would otherwise only show up as "the readouts seem a bit off":

* on a model whose Jacobian is known in closed form, does the estimator
  recover it exactly?
* does the Δ-bucketed fit really sum back to the plain fit?
* does the sparse decomposition recover a planted combination?
* does the block segmenter find planted blocks?
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch import nn

from jlens.atlas import Atlas, AtlasEntry, json_safe, read_entry, write_entry
from jlens.broadcast import broadcast_profile
from jlens.capacity import CapacityCurve, _first_token_id
from jlens.checkpoints import PYTHIA, discover, parse_ref
from jlens.cka import best_segmentation, blockiness_of, linear_cka, segment_blocks
from jlens.fitting import (
    delta_jacobian_for_prompt,
    fit,
    fit_delta,
    jacobian_for_prompt,
    valid_position_mask,
)
from jlens.jspace import (
    gradient_pursuit,
    nn_omp,
    participation_ratio,
    swap_coordinates,
)
from jlens.lens import JacobianLens
from jlens.lenscontrol import (
    FittingBudget,
    capacity_controlled_geometry,
    lens_fidelity,
    matched_depth_layers,
)
from jlens.scaling import (
    collapse_test,
    compare_power_laws,
    fit_broken_power_law,
    fit_power_law,
    partial_correlation,
)


# ----------------------------------------------------------------------------
# A model with a Jacobian we can write down
# ----------------------------------------------------------------------------


class LinearBlock(nn.Module):
    """``x -> x + W x``: purely position-local, so ``dh_out/dh_in = I + W``."""

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(d, d, generator=g) * (0.3 / math.sqrt(d)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x @ self.W.T


class LinearStack:
    """A LensModel whose layer-to-layer Jacobian is exactly a product of
    matrices, with no cross-position mixing at all."""

    def __init__(self, d: int = 8, n_layers: int = 4, vocab: int = 11) -> None:
        self.d_model, self.n_layers = d, n_layers
        self.layers = nn.ModuleList(LinearBlock(d, seed=i) for i in range(n_layers))
        g = torch.Generator().manual_seed(99)
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, d, generator=g))
            self.head.weight.copy_(torch.randn(vocab, d, generator=g))
        for p in list(self.layers.parameters()) + list(self.embed.parameters()) + list(
            self.head.parameters()
        ):
            p.requires_grad_(False)
        self.vocab = vocab
        self.tokenizer = self

    def decode(self, ids) -> str:
        return " ".join(f"t{int(i)}" for i in ids)

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = [int(w[1:]) for w in text.split()][:max_length]
        return torch.tensor(ids, dtype=torch.long)[None]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.layers:
            x = block(x)
        return x

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.head(residual)

    def true_jacobian(self, layer: int) -> torch.Tensor:
        """Exact ``dh_final / dh_layer`` for one position."""
        d = self.d_model
        J = torch.eye(d)
        for i in range(layer + 1, self.n_layers):
            J = (torch.eye(d) + self.layers[i].W) @ J
        return J


def _prompt(n: int) -> str:
    return " ".join(f"t{i % 11}" for i in range(n))


def test_estimator_matches_closed_form() -> None:
    """On a position-local linear stack, the estimator must recover ``J_l``
    exactly — but with a known multiplier.

    The default estimator sums over target positions ``p' >= p``. With no
    cross-position mixing, only ``p' == p`` contributes, so the sum collapses
    to the single-position Jacobian and the mean over sources leaves it
    unchanged.
    """
    model = LinearStack()
    layers = [0, 1, 2]
    J, seq_len, n_valid = jacobian_for_prompt(
        model, _prompt(24), layers, dim_batch=4, max_seq_len=24, skip_first=2
    )
    for layer in layers:
        expected = model.true_jacobian(layer)
        err = (J[layer] - expected).abs().max().item()
        assert err < 1e-4, f"layer {layer}: max abs error {err}"
    print(f"  ok  estimator matches closed form (seq={seq_len}, valid={n_valid})")


def test_delta_buckets_sum_to_plain_fit() -> None:
    """Summing the Δ-resolved buckets must reproduce the collapsed estimator.

    This is the property that makes the horizon experiment a *refinement* of
    the standard lens rather than a different measurement.
    """
    model = LinearStack(d=6, n_layers=3)
    prompts = [_prompt(20), _prompt(22)]
    kw = dict(dim_batch=3, max_seq_len=22, skip_first=2)
    plain = fit(model, prompts, source_layers=[0, 1], log_every=0,
                checkpoint_path=None, **kw)
    bucketed = fit_delta(
        model, prompts, source_layers=[0, 1], delta_bins=(0, 1, 2, 4, 8),
        log_every=0, **kw
    ).collapsed()
    for layer in (0, 1):
        err = (plain.jacobians[layer] - bucketed.jacobians[layer]).abs().max().item()
        scale = plain.jacobians[layer].abs().max().item()
        assert err / scale < 1e-4, f"layer {layer}: relative error {err / scale}"
    print("  ok  delta buckets sum to the plain fit")


def test_delta_localises_a_position_local_model() -> None:
    """A position-local model has zero forward reach: all of its Jacobian mass
    must land in the Δ=0 bucket."""
    model = LinearStack(d=6, n_layers=3)
    sums, counts, _, _ = delta_jacobian_for_prompt(
        model, _prompt(18), [0, 1], delta_bins=(0, 1, 2, 4),
        dim_batch=3, max_seq_len=18, skip_first=2,
    )
    mass = sums[0].flatten(1).norm(dim=1)
    assert mass[0] > 0, "no mass at Delta=0"
    assert mass[1:].max() < 1e-5 * mass[0], f"leaked mass to Delta>0: {mass}"
    print(f"  ok  delta attribution localises (mass by bucket: "
          f"{[round(float(m), 4) for m in mass]})")


def test_merge_equals_single_fit() -> None:
    """Fitting on disjoint slices and merging must equal one fit on the union —
    otherwise sharding across machines silently changes the answer."""
    model = LinearStack(d=6, n_layers=3)
    prompts = [_prompt(20), _prompt(21), _prompt(22), _prompt(23)]
    kw = dict(source_layers=[0, 1], dim_batch=3, max_seq_len=24, skip_first=2,
              log_every=0, checkpoint_path=None)
    whole = fit(model, prompts, **kw)
    merged = JacobianLens.merge(
        [fit(model, prompts[:1], **kw), fit(model, prompts[1:], **kw)]
    )
    assert merged.n_prompts == whole.n_prompts == 4
    for layer in (0, 1):
        err = (whole.jacobians[layer] - merged.jacobians[layer]).abs().max().item()
        assert err < 1e-5, f"layer {layer}: {err}"
    print("  ok  sharded fit + merge == single fit")


def test_gradient_pursuit_recovers_planted() -> None:
    """Plant a sparse non-negative combination and recover it."""
    g = torch.Generator().manual_seed(3)
    V = torch.randn(200, 32, generator=g)
    V = V / V.norm(dim=1, keepdim=True)
    idx = [7, 42, 130]
    coefs = torch.tensor([2.0, 1.3, 0.7])
    x = (V[idx] * coefs[:, None]).sum(0)
    # Both must identify the right atoms. They differ in how well they fit the
    # coefficients: nn_omp re-solves the whole support each step, gradient
    # pursuit takes one line-search step, so its residual stays a little higher.
    # That is the documented trade, and the test pins it rather than hiding it.
    errors = {}
    for name, fn, tol in (
        ("gradient_pursuit", gradient_pursuit, 0.10),
        ("nn_omp", nn_omp, 0.02),
    ):
        dec = fn(x, V, 5)
        found = {int(i) for i in dec.indices[:3]}
        assert set(idx) <= found, f"{name} picked {found}, wanted {idx}"
        err = float(dec.reconstruction_error[3])
        assert err < tol, f"{name} residual {err:.3f} after 3 atoms (tol {tol})"
        errors[name] = err
    assert errors["nn_omp"] <= errors["gradient_pursuit"] + 1e-6
    print(f"  ok  sparse decomposition recovers a planted combination "
          f"(residual after 3 atoms: gp {errors['gradient_pursuit']:.3f}, "
          f"omp {errors['nn_omp']:.3f})")


def test_swap_preserves_orthogonal_component() -> None:
    """The swap must leave everything orthogonal to the two vectors alone."""
    g = torch.Generator().manual_seed(5)
    v_s, v_t = torch.randn(16, generator=g), torch.randn(16, generator=g)
    h = torch.randn(16, generator=g)
    out = swap_coordinates(h, v_s, v_t)
    V = torch.stack([v_s, v_t], dim=1)
    Q, _ = torch.linalg.qr(V)
    perp = lambda z: z - Q @ (Q.T @ z)  # noqa: E731
    assert (perp(out) - perp(h)).abs().max() < 1e-5
    # And it must actually move the coordinates.
    c_in = torch.linalg.pinv(V) @ h
    c_out = torch.linalg.pinv(V) @ out
    assert (c_out - torch.stack([c_in[1], c_in[0]])).abs().max() < 1e-4
    print("  ok  swap exchanges coordinates and preserves the complement")


def test_segmentation_finds_planted_blocks() -> None:
    """Build a matrix with known blocks and check the segmenter finds them."""
    sizes = [6, 9, 5]
    L = sum(sizes)
    C = np.full((L, L), 0.3)
    start = 0
    bounds = []
    for s in sizes:
        C[start : start + s, start : start + s] = 0.9
        bounds.append(start)
        start += s
    np.fill_diagonal(C, 1.0)
    seg = best_segmentation(C, 3)
    assert [i for i, _ in seg] == bounds, f"{seg} vs {bounds}"
    B, mw, mb = blockiness_of(C, seg)
    assert B > 0.5, B
    chosen = segment_blocks(C, list(range(L)), L + 1)
    assert chosen.n_blocks == 3, chosen.n_blocks
    print(f"  ok  segmenter recovers planted blocks (blockiness {B:.2f})")


def test_cka_basic_properties() -> None:
    """CKA must be 1 against a rotation/scaling of itself, and lower against
    an unrelated matrix — that invariance is what makes it usable across models
    of different widths."""
    g = torch.Generator().manual_seed(7)
    X = torch.randn(64, 12, generator=g)
    Q, _ = torch.linalg.qr(torch.randn(12, 12, generator=g))
    assert abs(linear_cka(X, X) - 1.0) < 1e-6
    assert abs(linear_cka(X, 3.7 * X @ Q) - 1.0) < 1e-6
    assert linear_cka(X, torch.randn(64, 12, generator=g)) < 0.5
    print("  ok  CKA is rotation/scale invariant and discriminative")


def test_participation_ratio_bounds() -> None:
    """PR is 1 for a rank-1 dictionary and ~d for an isotropic one."""
    g = torch.Generator().manual_seed(11)
    u = torch.randn(1, 24, generator=g)
    rank1 = torch.randn(400, 1, generator=g) @ u
    assert participation_ratio(rank1) < 1.05
    iso = torch.randn(4000, 24, generator=g)
    assert 20 < participation_ratio(iso) <= 24
    print("  ok  participation ratio behaves at both extremes")


def test_valid_position_mask() -> None:
    m = valid_position_mask(20, skip_first=4)
    assert m.sum() == 15 and not m[:4].any() and not m[-1]
    for bad in (3, 4):
        try:
            valid_position_mask(bad, skip_first=4)
        except ValueError:
            pass
        else:
            raise AssertionError(f"seq_len={bad} should have raised")
    print("  ok  position mask excludes sinks and the final position")


def test_broadcast_ratio_matches_closed_form() -> None:
    """The causal broadcast ratio must equal ``||Mu|| / mean_i ||Mr_i||``.

    On the position-local linear stack a perturbation ``δ`` injected at the last
    position of layer ``l`` arrives at downstream block ``k`` as exactly
    ``M δ``, and nothing else in the sequence moves. So the reach — the ratio at
    the deepest block — has a closed form, and because the control directions
    come from a seeded generator in a known order they can be replayed exactly.

    This pins the part that is easy to get quietly wrong: whether the
    normalisation really cancels, leaving a ratio that means "how much further
    than matched noise" rather than "how big was my perturbation".
    """
    model = LinearStack(d=8, n_layers=4)
    prompt = _prompt(12)
    M = model.true_jacobian(0)  # source layer 0 -> final block
    u = torch.randn(8, generator=torch.Generator().manual_seed(3))

    n_controls = 8
    prof = broadcast_profile(
        model, u, 0, [prompt], n_controls=n_controls, max_seq_len=12,
        generator=torch.Generator().manual_seed(7),
    )

    replay = torch.Generator().manual_seed(7)
    num = (M @ (u / u.norm())).norm()
    den = np.mean([
        float((M @ (r / r.norm())).norm())
        for r in (torch.randn(8, generator=replay) for _ in range(n_controls))
    ])
    expected = float(num) / den
    assert abs(prof.reach - expected) < 1e-4, (prof.reach, expected)

    # A direction the network amplifies must out-reach one it attenuates.
    _, _, Vh = torch.linalg.svd(M)
    kw = dict(prompts=[prompt], n_controls=4, max_seq_len=12)
    top = broadcast_profile(model, Vh[0], 0,
                            generator=torch.Generator().manual_seed(1), **kw)
    bot = broadcast_profile(model, Vh[-1], 0,
                            generator=torch.Generator().manual_seed(1), **kw)
    assert top.reach > 1.0 > bot.reach, (top.reach, bot.reach)
    print(f"  ok  broadcast ratio matches closed form (reach {prof.reach:.4f}, "
          f"top/bottom singular {top.reach:.2f}/{bot.reach:.2f})")


def test_eviction_separates_decay_from_eviction() -> None:
    """A uniform dimming must not be reported as a bottleneck.

    Two curves with the *same* eviction slope: in the first only the earliest
    item decays, in the second every item decays together. Selectivity is the
    only thing that tells them apart, and it is the difference between "the
    workspace evicts" and "the readout gets noisier with longer prompts".
    """
    lengths = [1, 2, 3, 4, 6, 8]
    n = np.array(lengths, float)
    first = np.exp(0.3 * n)

    def curve(all_item):
        return CapacityCurve(lengths, first, all_item, {}, np.full(len(n), 1e3))

    evicting = curve(np.full(len(n), np.e))          # average flat
    dimming = curve(np.exp(0.3 * n))                 # average falls identically

    assert abs(evicting.eviction_slope - 0.3) < 1e-9
    assert abs(evicting.selectivity - 0.3) < 1e-9
    assert abs(dimming.eviction_slope - 0.3) < 1e-9
    assert abs(dimming.selectivity) < 1e-9, dimming.selectivity

    flat = CapacityCurve(lengths, np.full(len(n), 5.0), np.full(len(n), 5.0),
                         {}, np.full(len(n), 1e3))
    assert abs(flat.eviction_slope) < 1e-9
    print("  ok  eviction slope separates selective eviction from uniform decay")


def test_first_token_id_skips_bos() -> None:
    """Every item's first token being BOS would flatten the capacity curve."""

    class BosTokenizer:
        def encode(self, text, add_special_tokens=True):
            ids = [ord(c) for c in text]
            return [0] + ids if add_special_tokens else ids

    assert _first_token_id(BosTokenizer(), "ab") == ord("a")

    class NoKwargTokenizer:
        def encode(self, text):
            return [ord(c) for c in text]

    assert _first_token_id(NoKwargTokenizer(), "ab") == ord("a")
    assert _first_token_id(NoKwargTokenizer(), "") is None
    print("  ok  first-token lookup avoids the BOS that would flatten the curve")


def test_power_law_comparison_on_planted_data() -> None:
    """Smooth data must not yield a break; planted breaks must be located.

    The bootstrap is the point of the exercise: the naive nested F-test does not
    price the breakpoint search, so it calls breaks on smooth data. Both
    branches are checked, because a test that only confirms breaks are found
    would pass on a procedure that finds them everywhere.
    """
    rng = np.random.default_rng(0)
    x = np.logspace(2, 6, 40)

    smooth_y = 10 ** (1.0 + 0.5 * np.log10(x) + rng.normal(0, 0.03, len(x)))
    fit = fit_power_law(x, smooth_y)
    assert abs(fit.slope - 0.5) < 0.05, fit.slope
    res = compare_power_laws(x, smooth_y, n_boot=200, seed=1)
    assert res["verdict"] == "smooth power law not rejected", res["verdict"]

    lx = np.log10(x)
    b = 4.0
    broken_y = 10 ** (1.0 + np.where(lx < b, 0.05 * (lx - b), 0.9 * (lx - b))
                      + rng.normal(0, 0.03, len(x)))
    bf = fit_broken_power_law(x, broken_y)
    assert abs(np.log10(bf.breakpoint) - b) < 0.35, bf.breakpoint
    res = compare_power_laws(x, broken_y, n_boot=200, seed=1)
    assert res["verdict"] == "break justified", res
    lo, hi = res["breakpoint_ci95"]
    assert lo <= 10**b <= hi, (lo, 10**b, hi)
    print(f"  ok  break test: smooth data clean, planted break found at "
          f"{bf.breakpoint:.3g} (CI {lo:.3g}-{hi:.3g})")


def test_collapse_picks_the_generating_axis() -> None:
    """The collapse test must name the axis the data was actually made from."""
    rng = np.random.default_rng(2)
    params = np.repeat([1e7, 1e8, 1e9], 8).astype(float)
    tokens = np.tile(np.logspace(8, 11, 8), 3)
    compute = 6 * params * tokens
    y = 10 ** (-3 + 0.2 * np.log10(compute) + rng.normal(0, 0.02, len(compute)))
    out = collapse_test(y, {"compute": compute, "params": params, "tokens": tokens})
    assert out["best"] == "compute", out
    assert out["compute"]["r2"] > 0.99, out["compute"]
    print(f"  ok  collapse test picks the generating axis "
          f"(compute r2 {out['compute']['r2']:.3f})")


def test_matched_depth_layers_are_shape_matched() -> None:
    """Models of different depths must yield the same number of probe layers.

    Otherwise every layer-resolved statistic is computed on a differently sized
    matrix, and a claim about depth is partly a claim about matrix size.
    """
    for n_layers in (6, 12, 24, 36):
        layers, rho = matched_depth_layers(n_layers, 5)
        assert len(layers) == 5, (n_layers, layers)
        assert len(set(layers)) == len(layers)
        assert layers == sorted(layers)
        assert layers[0] == 0 and layers[-1] <= n_layers - 2
        assert 0.0 <= rho[0] < rho[-1] <= 1.0

    # Too shallow to honour the request: report fewer rather than inventing.
    short, _ = matched_depth_layers(3, 5)
    assert len(short) < 5, short
    print("  ok  probe layers are shape-matched across depths 6..36")


def test_budget_fingerprint_catches_a_changed_budget() -> None:
    """Two runs under different budgets must not hash the same."""
    base = FittingBudget()
    assert FittingBudget().fingerprint() == base.fingerprint()
    for changed in (
        FittingBudget(n_prompts=base.n_prompts + 1),
        FittingBudget(max_seq_len=base.max_seq_len * 2),
        FittingBudget(skip_first=base.skip_first + 1),
        FittingBudget(n_probe_layers=base.n_probe_layers + 1),
        FittingBudget(n_probe_tokens=base.n_probe_tokens + 1),
    ):
        assert changed.fingerprint() != base.fingerprint(), changed
    # Samples per entry, not prompts, is what fixes estimation noise, so
    # sequence length has to move it.
    assert FittingBudget(max_seq_len=256).samples_per_entry > base.samples_per_entry
    print("  ok  budget fingerprint changes with every budget field")


def test_capacity_control_shrinks_the_width_confound() -> None:
    """The same latent structure in a wider ambient space must not read as more.

    A rank-``k`` dictionary embedded in 32 and in 96 dimensions holds the same
    amount; raw participation ratio still reports more for the wider one, which
    is exactly how a scale sweep talks itself into "bigger models have bigger
    workspaces". The controlled statistic has to shrink that gap.
    """
    def build(d: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        B = torch.randn(8, d, generator=g)
        W = torch.randn(200, 8, generator=g)
        return W @ B + 0.05 * torch.randn(200, d, generator=g)

    narrow = capacity_controlled_geometry(build(32), d_ref=24, seed=1)
    wide = capacity_controlled_geometry(build(96), d_ref=24, seed=1)

    def gap(a, b):
        return abs(a - b) / max(abs(a), abs(b))

    raw_gap = gap(narrow["participation_ratio"], wide["participation_ratio"])
    ctl_gap = gap(narrow["projected_pr_ratio"], wide["projected_pr_ratio"])
    assert ctl_gap < raw_gap, (raw_gap, ctl_gap)
    assert narrow["d_ref"] == wide["d_ref"] == 24
    print(f"  ok  width confound shrinks under control ({raw_gap:.1%} -> "
          f"{ctl_gap:.1%} across a 3x width change)")


def test_lens_fidelity_beats_the_logit_lens_where_it_should() -> None:
    """The instrument check must be able to tell a live lens from a dead one.

    On the linear stack the true Jacobian is known, so a lens holding it must
    reconstruct the final logits *exactly* (r = 1) and beat the logit lens. A
    lens holding zeros is the degenerate case an early checkpoint produces, and
    must report no advantage — that is the signal that suppresses geometry
    claims from a rung where the transport bought nothing.
    """
    model = LinearStack(d=8, n_layers=4)
    prompts = [_prompt(20), _prompt(22)]
    layers = [0, 1, 2]

    exact = JacobianLens(
        {l: model.true_jacobian(l) for l in layers}, n_prompts=1, d_model=8
    )
    fid = lens_fidelity(model, exact, prompts, max_seq_len=22)
    assert np.allclose(fid.jlens_r, 1.0, atol=1e-5), fid.jlens_r
    assert fid.advantage > 0.0, fid.advantage

    dead = JacobianLens(
        {l: torch.zeros(8, 8) for l in layers}, n_prompts=1, d_model=8
    )
    dead_fid = lens_fidelity(model, dead, prompts, max_seq_len=22)
    assert not (dead_fid.advantage > 0), dead_fid.advantage
    print(f"  ok  lens fidelity separates an exact lens (advantage "
          f"{fid.advantage:+.3f}) from a dead one ({dead_fid.advantage:+.3f})")


def test_partial_correlation_reports_full_explanation_as_zero() -> None:
    """A measure that is entirely the covariate must read 0, not NaN.

    This is the whole point of the instrument control, and it is the case where
    the residual collapses — so the naive guard returns NaN and hides the
    finding behind a missing value. The genuinely undefined case (covariate
    collinear with x) still has to return NaN.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(18, 24, 20)
    z = 0.3 * (x - 18) / 6 + rng.normal(0, 0.02, 20)

    pure = partial_correlation(2.0 * z, x, {"z": z})
    assert pure["partial"] == 0.0 and abs(pure["raw"]) > 0.9, pure
    assert "note" in pure

    real = partial_correlation(2.0 * z + 0.5 * x + rng.normal(0, 0.01, 20), x, {"z": z})
    assert abs(real["partial"]) > 0.9, real

    collinear = partial_correlation(rng.normal(0, 1, 20), x, {"x_again": x})
    assert np.isnan(collinear["partial"]) and "note" in collinear, collinear
    print("  ok  partial correlation: full explanation reads 0, collinear reads NaN")


def test_checkpoint_refs_parse_across_families() -> None:
    """Checkpoint discovery must handle each family's naming, and reject refs
    that are not checkpoints at all.

    A stale hardcoded list fails silently — it yields a short sweep that looks
    complete — so the parsing is what has to be right.
    """
    cases = {
        "step143000": (143000, None, None),
        "stage1-step140000-tokens587B": (140000, 587_000_000_000, "stage1"),
        "stage2-ingredient1-step11931-tokens50B": (11931, 50_000_000_000, "stage2"),
        "step1000-tokens4B": (1000, 4_000_000_000, None),
        "ckpt_356": (356, None, None),
        "iter_0143000": (143000, None, None),
    }
    for ref, (step, tokens, stage) in cases.items():
        cp = parse_ref(ref)
        assert cp is not None, ref
        assert cp.step == step and cp.stage == stage, (ref, cp)
        if tokens is not None:
            assert cp.tokens == tokens, (ref, cp.tokens)

    for junk in ("main", "refs/pr/3", "v1.0", "latest"):
        assert parse_ref(junk) is None, junk

    # A token count in the ref name is the release's own accounting and must
    # beat a family-level constant, which assumes a schedule that may have
    # changed mid-run.
    cp = parse_ref("stage1-step140000-tokens587B", tokens_per_step=999)
    assert cp.tokens == 587_000_000_000, cp.tokens

    # Same step, different branch: distinct entries, deterministic order.
    a = parse_ref("stage2-ingredient1-step11931-tokens50B")
    b = parse_ref("stage2-ingredient3-step11931-tokens50B")
    assert a.variant != b.variant and a.sort_key < b.sort_key
    print("  ok  checkpoint refs parse across families, junk refs rejected")


def test_discovery_orders_and_subsets() -> None:
    """Discovery must sort by training order and keep the endpoints."""
    refs = ["main"] + [f"step{s}" for s in
                       [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                       + list(range(1000, 144000, 1000))]
    cps = discover("EleutherAI/pythia-410m", tokens_per_step=PYTHIA.tokens_per_step,
                   lister=lambda r: refs)
    assert len(cps) == 154, len(cps)  # what Pythia actually publishes
    assert [c.step for c in cps] == sorted(c.step for c in cps)

    lim = discover("x", lister=lambda r: refs, limit=8)
    assert len(lim) == 8 and lim[0].step == 0 and lim[-1].step == 143000, lim

    sub = discover("x", lister=lambda r: refs, keep_steps=[0, 512, 999999])
    assert [c.step for c in sub] == [0, 512], sub  # absent steps dropped, not raised

    _, _, params = PYTHIA.arch("410m")
    assert cps[-1].compute_flops(params) > 0
    assert cps[0].compute_flops(None) is None  # unknown params -> unknown compute
    print(f"  ok  discovery orders 154 Pythia checkpoints, subsets keep endpoints")


def test_atlas_roundtrip_and_key_collisions() -> None:
    """The atlas must round-trip through disk, and colliding labels must not
    silently merge.

    Two families routinely publish the same ref name — OLMo-7B and OLMo-32B
    both have ``stage1-step1000`` — and keying on the label alone gives them one
    shared heatmap in the viewer.
    """
    import tempfile

    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 9))
    C = np.clip((X @ X.T) / np.abs(X @ X.T).max(), 0, 1)
    np.fill_diagonal(C, 1.0)

    with tempfile.TemporaryDirectory() as d:
        npz, js = write_entry(d, "step1000", C, list(range(9)), 12, 96,
                              "toy/12L step1000", {"n_blocks": 2})
        back = read_entry(npz)
        assert np.allclose(back["C"], C)
        assert back["n_layers"] == 12 and back["d_model"] == 96
        assert np.allclose(back["rho"], np.arange(9) / 11)
        assert os.path.basename(npz) == "cka_step1000.npz"
        assert os.path.basename(js) == "step1000_blocks.json"

        a = AtlasEntry(label="stage1-step1000", family="olmo7b", size="7B",
                       repo="r", n_layers=32, d_model=4096)
        b = AtlasEntry(label="stage1-step1000", family="olmo32b", size="32B",
                       repo="r", n_layers=64, d_model=5120)
        assert a.key != b.key, (a.key, b.key)

        atlas = Atlas(entries=[a, b], title="t")
        assert not atlas.manifest()["warnings"]
        atlas.add(AtlasEntry(label="stage1-step1000", family="olmo7b",
                             size="7B", repo="r", n_layers=32, d_model=4096))
        assert any("duplicate" in w for w in atlas.manifest()["warnings"])

        path = Atlas(entries=[a, b], title="t").save(d)
        loaded = Atlas.load(path)
        assert [e.key for e in loaded.entries] == [a.key, b.key]

    # A bare NaN is invalid JSON and blanks the whole viewer, not one cell.
    assert json_safe({"a": float("nan"), "b": [float("inf"), 1.5]}) == {
        "a": None, "b": [None, 1.5]}
    print("  ok  atlas round-trips; colliding labels stay distinct")


def test_dead_lens_entries_are_marked() -> None:
    """An entry whose lens lost to the logit lens must be flagged, and an
    unmeasured one must not be silently treated as dead."""
    live = AtlasEntry(label="a", family="f", size="s", repo="r",
                      n_layers=6, d_model=64, fidelity={"advantage": 0.2})
    dead = AtlasEntry(label="b", family="f", size="s", repo="r",
                      n_layers=6, d_model=64, fidelity={"advantage": -0.1})
    zero = AtlasEntry(label="c", family="f", size="s", repo="r",
                      n_layers=6, d_model=64, fidelity={"advantage": 0.0})
    unknown = AtlasEntry(label="d", family="f", size="s", repo="r",
                         n_layers=6, d_model=64)
    assert live.lens_is_live and not dead.lens_is_live and not zero.lens_is_live
    assert unknown.lens_is_live, "absence of a measurement is not evidence of a dead lens"

    warnings = Atlas(entries=[live, dead, zero, unknown]).manifest()["warnings"]
    assert any("2 of 4" in w for w in warnings), warnings
    print("  ok  dead-lens entries are flagged, unmeasured ones are not")


def main() -> None:
    torch.set_num_threads(2)
    tests = [
        test_valid_position_mask,
        test_estimator_matches_closed_form,
        test_delta_buckets_sum_to_plain_fit,
        test_delta_localises_a_position_local_model,
        test_merge_equals_single_fit,
        test_gradient_pursuit_recovers_planted,
        test_swap_preserves_orthogonal_component,
        test_segmentation_finds_planted_blocks,
        test_cka_basic_properties,
        test_participation_ratio_bounds,
        test_broadcast_ratio_matches_closed_form,
        test_eviction_separates_decay_from_eviction,
        test_first_token_id_skips_bos,
        test_power_law_comparison_on_planted_data,
        test_collapse_picks_the_generating_axis,
        test_matched_depth_layers_are_shape_matched,
        test_budget_fingerprint_catches_a_changed_budget,
        test_capacity_control_shrinks_the_width_confound,
        test_lens_fidelity_beats_the_logit_lens_where_it_should,
        test_partial_correlation_reports_full_explanation_as_zero,
        test_checkpoint_refs_parse_across_families,
        test_discovery_orders_and_subsets,
        test_atlas_roundtrip_and_key_collisions,
        test_dead_lens_entries_are_marked,
    ]
    print(f"running {len(tests)} tests\n")
    for t in tests:
        t()
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
