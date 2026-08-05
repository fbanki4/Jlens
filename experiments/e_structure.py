"""Is the J-space really just a bag of concepts?

The paper's own caveat: the lens reads the workspace as a flat set of
independently active concept vectors. A readout containing *spider*, *legs* and
*eight* says those concepts are present, not how they are bound — and if the
model imposes extra structure (roles, relations, bindings), a bag-of-features
readout cannot see it.

The composed relations make this testable, because they let two differences be
separated cleanly:

**The latent axis.** Two subjects whose composed *answer* is the same country
but whose intermediate city differs. ``d_latent = h(A) - h(B)`` therefore
differs in the unverbalized intermediate while the output is held fixed.

**The role axis.** The *same* subject city under two different composed
relations — ``paris utters`` (city -> country -> language) and ``paris stands
within`` (city -> country -> continent). Same subject, same latent country,
different relation being applied. ``d_role`` is what the workspace carries
about *which operation is in flight*, with the concept held fixed.

If the J-space is a bag of concepts, the J-lens dictionary should account for
the latent axis well and the role axis poorly. If it accounts for both, the
workspace carries more than concept identity and the bag reading is too
modest.

Also replicates the capacity measurements: occupancy and fraction of variance
explained, by layer, against a random-dictionary control.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from jlens.cka import build_probe
from jlens.hooks import ActivationRecorder
from jlens.jspace import (
    fraction_variance_explained,
    gradient_pursuit,
    occupancy,
    random_control_dictionary,
)
from jlens.lens import JacobianLens
from jlens.toy.model import ToyLensModel, load_toy
from jlens.toy.world import (
    COMPOSED_RELATIONS,
    build_world,
    composed_prompt,
    make_split,
)


@torch.no_grad()
def query_acts(model: ToyLensModel, prompt: str, layers: list[int]):
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
        return {l: rec.activations[l][0, -1].detach().float() for l in layers}


def explained_by(x: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of ``x``'s energy inside the span of ``basis`` rows."""
    Q, _ = torch.linalg.qr(basis.T.float())
    return float((x.float() @ Q).pow(2).sum() / x.float().pow(2).sum().clamp_min(1e-12))


def solves(model: ToyLensModel, prompt: str, answer: str) -> bool:
    with torch.no_grad():
        pred = int(model.unembed(model.forward(model.encode(prompt)))[0, -1].argmax())
    return pred == model.tokenizer.token_id(answer.split()[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/toy/final.pt")
    ap.add_argument("--lens", default="runs/lens/final_lens.pt")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--out", default="runs/structure")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    net, tok = load_toy(args.model)
    model = ToyLensModel(net, tok)
    lens = JacobianLens.load(args.lens)
    world, split = build_world(), make_split(build_world())
    n_layers, layers = model.n_layers, lens.source_layers
    band = [l for l in layers if 0.3 <= l / (n_layers - 1) <= 0.8]

    P_full, _ = build_probe(model, token_ids=torch.arange(len(tok)))
    V = {l: (P_full @ lens.jacobians[l]).float() for l in layers}

    def vecs(layer: int, words: list[str]) -> torch.Tensor:
        return torch.stack(
            [V[layer][tok.encode(w)].sum(0) for w in words if tok.encode(w)]
        )

    # ---- build the two kinds of pair ------------------------------------
    composed = world.composed_map("hails_from")
    latents = world.latent_map("hails_from")
    by_answer = defaultdict(list)
    for s, ans in composed.items():
        if solves(model, composed_prompt("hails_from", s), ans):
            by_answer[ans].append(s)

    latent_pairs = []
    for ans, subjects in by_answer.items():
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                a, b = subjects[i], subjects[j]
                if latents[a] != latents[b]:
                    latent_pairs.append((a, b))
    print(f"latent-axis pairs (same answer, different intermediate city): "
          f"{len(latent_pairs)}")

    role_pairs = []
    for city in world.entities["city"]:
        pa = composed_prompt("tongue", city)
        pb = composed_prompt("lies_within", city)
        ans_a = world.composed_map("tongue").get(city)
        ans_b = world.composed_map("lies_within").get(city)
        if ans_a and ans_b and solves(model, pa, ans_a) and solves(model, pb, ans_b):
            role_pairs.append((city, pa, pb, ans_a, ans_b))
    print(f"role-axis pairs (same subject and latent, different relation): "
          f"{len(role_pairs)}")
    if len(latent_pairs) < 5 or len(role_pairs) < 5:
        raise SystemExit("not enough usable pairs; train the model longer")

    # ---- how much of each difference does the dictionary explain? --------
    stats = {
        axis: {l: defaultdict(list) for l in band}
        for axis in ("latent", "role")
    }
    for a, b in latent_pairs:
        ha = query_acts(model, composed_prompt("hails_from", a), band)
        hb = query_acts(model, composed_prompt("hails_from", b), band)
        for l in band:
            d = ha[l] - hb[l]
            dec = gradient_pursuit(d, V[l], args.k)
            ctrl = gradient_pursuit(d, random_control_dictionary(V[l]), args.k)
            e = d.pow(2).sum().clamp_min(1e-12)
            stats["latent"][l]["jspace"].append(
                1.0 - float(dec.residual.pow(2).sum() / e)
            )
            stats["latent"][l]["random"].append(
                1.0 - float(ctrl.residual.pow(2).sum() / e)
            )
            # the pure-bag model: just the two intermediates' own vectors
            stats["latent"][l]["specific"].append(
                explained_by(d, vecs(l, [latents[a], latents[b]]))
            )

    for city, pa, pb, _, _ in role_pairs:
        ha, hb = query_acts(model, pa, band), query_acts(model, pb, band)
        for l in band:
            d = ha[l] - hb[l]
            dec = gradient_pursuit(d, V[l], args.k)
            ctrl = gradient_pursuit(d, random_control_dictionary(V[l]), args.k)
            e = d.pow(2).sum().clamp_min(1e-12)
            stats["role"][l]["jspace"].append(1.0 - float(dec.residual.pow(2).sum() / e))
            stats["role"][l]["random"].append(1.0 - float(ctrl.residual.pow(2).sum() / e))
            # the "concepts named in the two prompts" model: the relation verbs
            # and the two answers, which is everything a bag-of-concepts
            # readout could point at to distinguish these two prompts
            stats["role"][l]["specific"].append(
                explained_by(
                    d,
                    vecs(l, ["utters", "stands within",
                             world.composed_map("tongue")[city],
                             world.composed_map("lies_within")[city]]),
                )
            )

    print()
    print("=" * 78)
    print("HOW MUCH OF EACH DIFFERENCE IS IN THE J-SPACE?")
    print("=" * 78)
    print(f"  {'':11s} {'LATENT axis (same answer, diff. city)':>38s}   "
          f"{'ROLE axis':>24s}")
    print(f"  {'layer  rho':11s} {'k-atom':>9s} {'random':>9s} {'2 vecs':>9s}   "
          f"{'k-atom':>7s} {'random':>7s} {'named':>7s}")
    for l in band:
        s_l, s_r = stats["latent"][l], stats["role"][l]
        print(f"  {l:5d} {l / (n_layers - 1):5.2f} "
              f"{np.mean(s_l['jspace']):9.3f} {np.mean(s_l['random']):9.3f} "
              f"{np.mean(s_l['specific']):9.3f}   "
              f"{np.mean(s_r['jspace']):7.3f} {np.mean(s_r['random']):7.3f} "
              f"{np.mean(s_r['specific']):7.3f}")

    def band_mean(axis: str, key: str) -> float:
        return float(np.mean([np.mean(stats[axis][l][key]) for l in band]))

    print()
    print(f"  band averages")
    print(f"    latent axis: {args.k}-atom J-lens {band_mean('latent', 'jspace'):.1%} "
          f"(random {band_mean('latent', 'random'):.1%}), "
          f"the two cities' own vectors alone {band_mean('latent', 'specific'):.1%}")
    print(f"    role   axis: {args.k}-atom J-lens {band_mean('role', 'jspace'):.1%} "
          f"(random {band_mean('role', 'random'):.1%}), "
          f"the named tokens alone {band_mean('role', 'specific'):.1%}")

    # ---- capacity ---------------------------------------------------------
    print()
    print("=" * 78)
    print("CAPACITY: occupancy and variance share by layer")
    print("=" * 78)
    probe_subjects = [s for subs in by_answer.values() for s in subs][:24]
    acts = {l: [] for l in layers}
    for s in probe_subjects:
        h = query_acts(model, composed_prompt("hails_from", s), layers)
        for l in layers:
            acts[l].append(h[l])
    print(f"  {'layer':>5s} {'rho':>5s} {'occupancy':>10s} {'FVE':>7s} "
          f"{'FVE rand':>9s} {'excess':>7s}")
    capacity = []
    gen = torch.Generator().manual_seed(0)
    for l in layers:
        occ = [
            occupancy(h, V[l], k_max=40, n_controls=1, generator=gen)[0]
            for h in acts[l][:10]
        ]
        k = max(1, int(np.median(occ)))
        fve, fve_r = fraction_variance_explained(torch.stack(acts[l]), V[l], k)
        capacity.append({"layer": l, "rho": l / (n_layers - 1),
                         "occupancy": float(np.median(occ)),
                         "fve": fve, "fve_random": fve_r})
        print(f"  {l:5d} {l / (n_layers - 1):5.2f} {np.median(occ):10.1f} "
              f"{fve:7.3f} {fve_r:9.3f} {fve - fve_r:7.3f}")

    with open(os.path.join(args.out, "structure.json"), "w") as f:
        json.dump(
            {
                "n_latent_pairs": len(latent_pairs),
                "n_role_pairs": len(role_pairs),
                "band": band,
                "axes": {
                    axis: {
                        str(l): {k: float(np.mean(v)) for k, v in stats[axis][l].items()}
                        for l in band
                    }
                    for axis in stats
                },
                "capacity": capacity,
            },
            f, indent=2,
        )
    print(f"\nwrote {args.out}/structure.json")


if __name__ == "__main__":
    main()
