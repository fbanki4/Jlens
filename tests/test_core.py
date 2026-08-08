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

import numpy as np
import torch
from torch import nn

from jlens.broadcast import broadcast_profile
from jlens.capacity import CapacityCurve, _first_token_id
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
from jlens.scaling import (
    collapse_test,
    compare_power_laws,
    fit_broken_power_law,
    fit_power_law,
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
    ]
    print(f"running {len(tests)} tests\n")
    for t in tests:
        t()
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
