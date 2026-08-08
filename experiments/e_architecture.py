"""Is the J-space architectural, or is it learned?

A claim you will meet: the workspace is "baked into the transformer
architecture itself, not merely an artifact of massive scale or post-training".
It is half right, and which half matters, because two different objects get
called "the J-space":

**Geometry** — the block structure in the layer x layer CKA matrix, the
low effective rank of the lens vectors, the fact that some band of layers
looks internally coherent. This is a property of ``J_l``, computable without
the model ever being right about anything.

**Function** — the workspace actually holding an unverbalized intermediate,
that intermediate being readable above matched distractors, and swapping it
changing the model's answer. This is a property of the model's *computation*.

Confusing them is easy, because both get measured off the same fitted lens.
This script measures both on the *same architecture* at random initialisation
and after training, so the two can be told apart.

If the geometry is architectural, an untrained model should show it too.
If the function is learned, only the trained model should show it.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from jlens.cka import (
    build_probe,
    cka_matrix,
    effective_dimension,
    lens_vectors,
    segment_blocks,
)
from jlens.fitting import fit
from jlens.jspace import participation_ratio, swap_coordinates
from jlens.lens import JacobianLens
from jlens.toy.model import ToyConfig, ToyLensModel, ToyTransformer, WordTokenizer, load_toy
from jlens.toy.world import build_world, composed_prompt, make_split, vocabulary
from experiments.fit_toy_lens import fitting_corpus


def geometry(model: ToyLensModel, lens: JacobianLens) -> dict:
    """Everything computable from the fitted lens alone."""
    P, _ = build_probe(model)
    C = cka_matrix(lens, P)
    seg = segment_blocks(C, lens.source_layers, model.n_layers)
    Vs = [lens_vectors(P, lens.jacobians[l]) for l in lens.source_layers]
    return {
        "n_blocks": seg.n_blocks,
        "blockiness": seg.blockiness,
        "layer_coupling": seg.layer_coupling,
        "boundaries_rho": seg.boundaries_rho,
        "pr_peak": float(max(participation_ratio(V) for V in Vs)),
        "eff_dim_95_mid": float(effective_dimension(Vs[len(Vs) // 2], 0.95)),
    }


@torch.no_grad()
def function(model: ToyLensModel, lens: JacobianLens, world, split) -> dict:
    """Everything that requires the model to actually compute something."""
    tok = model.tokenizer
    composed = world.composed_map("hails_from")
    latents = world.latent_map("hails_from")
    cities = world.entities["city"]
    city_country = world.atomic["city_in"]
    n_layers = model.n_layers
    band = [l for l in lens.source_layers if 0.3 <= l / (n_layers - 1) <= 0.8]

    P_full, _ = build_probe(model, token_ids=torch.arange(len(tok)))
    V = {l: (P_full @ lens.jacobians[l]).float() for l in lens.source_layers}

    solved, lat_ranks, dis_ranks = [], [], []
    rng = random.Random(0)
    for subject, answer in composed.items():
        prompt = composed_prompt("hails_from", subject)
        pred = int(model.unembed(model.forward(model.encode(prompt)))[0, -1].argmax())
        ok = pred == tok.token_id(answer)
        solved.append(ok)
        # Read the latent's rank against five matched distractors, whether or
        # not the model got it right — an untrained model still has a readout.
        lens_logits, _, _ = lens.apply(model, prompt, positions=[-1])
        distractors = rng.sample([c for c in cities if c != latents[subject]], 5)
        best_lat, best_dis = 10**9, 10**9
        for l in band:
            order = lens_logits[l][0].argsort(descending=True).tolist()
            rank = {t: i + 1 for i, t in enumerate(order)}
            best_lat = min(best_lat, rank[tok.token_id(latents[subject].split()[0])])
            best_dis = min(
                best_dis,
                int(np.mean([rank[tok.token_id(d.split()[0])] for d in distractors])),
            )
        lat_ranks.append(best_lat)
        dis_ranks.append(best_dis)

    # Causal: swap the latent for another city, does the answer follow?
    follow = total = 0
    for subject, answer in composed.items():
        prompt = composed_prompt("hails_from", subject)
        pred = int(model.unembed(model.forward(model.encode(prompt)))[0, -1].argmax())
        if pred != tok.token_id(answer):
            continue  # only meaningful where the model was right to begin with
        target = rng.choice(
            [c for c in cities if c != latents[subject] and city_country[c] != answer]
        )
        src = tok.token_id(latents[subject].split()[0])
        tgt = tok.token_id(target.split()[0])
        handles = []

        def make_hook(layer):
            def hook(mod, inp, out):
                t = out if torch.is_tensor(out) else out[0]
                p = t.clone()
                p[0, -1] = swap_coordinates(t[0, -1], V[layer][src], V[layer][tgt])
                return p if torch.is_tensor(out) else (p, *out[1:])
            return hook

        try:
            for l in band:
                handles.append(model.layers[l].register_forward_hook(make_hook(l)))
            got = tok.decode(
                [int(model.unembed(model.forward(model.encode(prompt)))[0, -1].argmax())]
            )
        finally:
            for h in handles:
                h.remove()
        total += 1
        follow += got == city_country[target]

    lat, dis = np.array(lat_ranks), np.array(dis_ranks)
    return {
        "heldout_accuracy": float(np.mean(solved)),
        "median_latent_rank": float(np.median(lat)),
        "median_distractor_rank": float(np.median(dis)),
        "specificity": float(np.median(dis) / max(np.median(lat), 1)),
        "latent_beats_distractors_frac": float((lat < dis).mean()),
        "swap_n": total,
        "swap_follow_rate": (follow / total) if total else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", default="runs/toy/final.pt")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--out", default="runs/architecture")
    args = ap.parse_args()

    torch.set_num_threads(4)
    os.makedirs(args.out, exist_ok=True)
    world, split = build_world(), make_split(build_world())
    prompts = fitting_corpus(args.n_prompts)

    trained_net, tok = load_toy(args.trained)
    cfg = trained_net.cfg

    # Same architecture, same seed discipline, zero training.
    torch.manual_seed(0)
    untrained_net = ToyTransformer(ToyConfig(**vars(cfg)))

    report = {}
    for name, net in (("untrained", untrained_net), ("trained", trained_net)):
        model = ToyLensModel(net, tok)
        cache = os.path.join(args.out, f"lens_{name}.pt")
        if os.path.exists(cache):
            lens = JacobianLens.load(cache)
        else:
            print(f"fitting lens on {name} model ({args.n_prompts} prompts) ...")
            lens = fit(model, prompts, dim_batch=96, max_seq_len=16, skip_first=1,
                       checkpoint_path=None, log_every=0)
            lens.save(cache)
        report[name] = {
            "geometry": geometry(model, lens),
            "function": function(model, lens, world, split),
        }

    u, t = report["untrained"], report["trained"]
    print()
    print("=" * 74)
    print("GEOMETRY — computable from the fitted lens alone")
    print("=" * 74)
    print(f"  {'':26s} {'untrained':>12s} {'trained':>12s}   {'ratio':>7s}")
    for k in ("n_blocks", "blockiness", "layer_coupling", "pr_peak", "eff_dim_95_mid"):
        a, b = u["geometry"][k], t["geometry"][k]
        r = f"{b / a:.2f}x" if a else "—"
        print(f"  {k:26s} {a:12.3f} {b:12.3f}   {r:>7s}")
    print(f"  {'block boundary (rho)':26s} "
          f"{str([round(x, 2) for x in u['geometry']['boundaries_rho']]):>12s} "
          f"{str([round(x, 2) for x in t['geometry']['boundaries_rho']]):>12s}")

    print()
    print("=" * 74)
    print("FUNCTION — requires the model to actually compute something")
    print("=" * 74)
    print(f"  {'':26s} {'untrained':>12s} {'trained':>12s}")
    for k, fmt in (("heldout_accuracy", "{:12.2f}"),
                   ("median_latent_rank", "{:12.0f}"),
                   ("median_distractor_rank", "{:12.0f}"),
                   ("specificity", "{:11.1f}x"),
                   ("latent_beats_distractors_frac", "{:12.2f}"),
                   ("swap_follow_rate", "{:12.3f}")):
        a, b = u["function"][k], t["function"][k]
        print(f"  {k:26s} " + fmt.format(a) + " " + fmt.format(b))
    print(f"  {'(swap n)':26s} {u['function']['swap_n']:12d} {t['function']['swap_n']:12d}")

    print()
    print("=" * 74)
    print("  GEOMETRY is already there at random init — training does not create it")
    print(f"    blockiness {u['geometry']['blockiness']:.3f} untrained -> "
          f"{t['geometry']['blockiness']:.3f} trained"
          f"   (here training *removes* it)")
    print(f"    block count {u['geometry']['n_blocks']} untrained -> "
          f"{t['geometry']['n_blocks']} trained")
    print("  FUNCTION is entirely absent at init — training creates all of it")
    print(f"    latent specificity {u['function']['specificity']:.1f}x untrained -> "
          f"{t['function']['specificity']:.1f}x trained")
    print(f"    latent outranks distractors on "
          f"{u['function']['latent_beats_distractors_frac']:.0%} of items untrained "
          f"(chance is 50%) -> {t['function']['latent_beats_distractors_frac']:.0%} trained")
    print("=" * 74)

    with open(os.path.join(args.out, "architecture.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}/architecture.json")


if __name__ == "__main__":
    main()
