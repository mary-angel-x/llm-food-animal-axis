"""
explore_layers.py — EXPLORATORY, not preregistered.

Same LOPO pipeline as analyze.py, but with the layer fixed instead of chosen
per fold: d_M, d_I, exact permutation p and the placebo 95th percentile at
every layer. Written after the preregistered selection rule picked layer 0 in
most Qwen folds (tiny but consistent differences maximise the scale-free d_z).
28 layers are tested, so p values here are uncorrected and descriptive.

Usage:
    python explore_layers.py --model qwen-7b
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import analyze as A
from models import MODELS
from stimuli import PAIRS, PLACEBO_PAIRS

LOGS = Path(__file__).parent / "logs"


def fixed_layer(G, layer):
    return G[..., layer:layer + 1]  # one-layer Gram: the argmax has a single choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--placebo", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(A.SEED)

    states, index = A.load_states(args.model)
    bank, slot = A.build_bank(states, index, [(p["food"], p["nonfood"]) for p in PAIRS])
    G = A.gram(bank)
    bank_p, slot_p = A.build_bank(states, index, [(p["a"], p["b"]) for p in PLACEBO_PAIRS])
    Gp = A.gram(bank_p)
    nos = list(range(len(PAIRS)))
    K = len(nos)
    bits = [np.array(b, bool) for b in itertools.product([0, 1], repeat=K)]

    rows = []
    print(f"{'L':>3} {'d_M':>6} {'p_M':>7} {'plc95':>6} {'d_I':>6} {'p_I':>7} {'plc95':>6}  mean s_M  mean s_N")
    for L in range(G.shape[-1]):
        g, gp = fixed_layer(G, L), fixed_layer(Gp, L)
        sM, sN, _ = A.run_lopo(g, slot, nos)
        obs = A.stats(sM, sN)
        flips = np.array([A.stats(*A.run_lopo(g, slot, nos, sign=np.where(b, -1, 1))[:2])["d_M"] for b in bits])
        swaps = np.array([A.stats(*A.run_lopo(g, slot, nos, swap=b)[:2])["d_I"] for b in bits])
        plc = {"d_M": [], "d_I": []}
        for _ in range(args.placebo):
            pn = rng.choice(len(PLACEBO_PAIRS), K, replace=False)
            st = A.stats(*A.run_lopo(gp, slot_p, list(pn), sign=rng.choice([-1, 1], K))[:2])
            plc["d_M"].append(st["d_M"])
            plc["d_I"].append(st["d_I"])
        row = {"layer": L, **obs,
               "p_M": float((flips >= obs["d_M"] - 1e-12).mean()),
               "p_I": float((swaps >= obs["d_I"] - 1e-12).mean()),
               "placebo95_M": float(np.percentile(plc["d_M"], 95)),
               "placebo95_I": float(np.percentile(plc["d_I"], 95)),
               "mean_s_M": float(sM.mean()), "mean_s_N": float(sN.mean()), "s_M": sM.tolist(), "s_N": sN.tolist()}
        rows.append(row)
        print(f"{L:>3} {row['d_M']:6.2f} {row['p_M']:7.4f} {row['placebo95_M']:6.2f} "
              f"{row['d_I']:6.2f} {row['p_I']:7.4f} {row['placebo95_I']:6.2f}  {row['mean_s_M']:8.3f}  {row['mean_s_N']:8.3f}",
              flush=True)
    (LOGS / f"{args.model}_explore_layers.json").write_text(
        json.dumps({"model": MODELS[args.model]["name"], "exploratory": True, "rows": rows}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
