"""
analyze.py — stage 3: H1, H1-int and B0 for one model (preregistration §5, §6).

Every pair is reduced to difference vectors (x − y, per layer):
    fit_M / fit_N   mean over fit templates (0–3), moral / neutral context
    ev_M  / ev_N    mean over eval templates (4–7)
    fit_I           fit_M − fit_N (v_int, secondary)
Direction, projections and layer choice are linear in these vectors, so the
whole leave-one-pair-out pipeline runs from their Gram matrix. A permutation
(label flip or context swap inside a pair) only re-signs / re-indexes that
matrix, which makes the exact 2^13 permutation tests cheap.

Output: logs/<model>_h1.json (all statistics) and logs/<model>_directions.pt
(per-fold unit directions and layers, used by the ablation stage).

Usage:
    python analyze.py --model qwen-7b
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from models import MODELS
from stimuli import (PAIRS, AMBIGUOUS, PLACEBO_PAIRS, FIT_TEMPLATES, EVAL_TEMPLATES,
                     TOKEN_MISMATCH_PAIRS, MEAT_HOMONYM, SCALE_MAX)

LOGS = Path(__file__).parent / "logs"
STIMULI = Path(__file__).parent / "stimuli.json"
SEED = 0
N_PLACEBO = 1000
N_BOOT = 10_000
KINDS = ["fit_M", "fit_N", "ev_M", "ev_N", "fit_I"]


# ---------------------------------------------------------------- data

def load_states(key):
    d = torch.load(LOGS / f"{key}_states.pt")
    states = d["states"].float().numpy()  # [n_prompts, n_layers, d]
    index = {(r["animal"], r["context"], r["template"]): i for i, r in enumerate(d["rows"])}
    return states, index


def mean_state(states, index, animal, ctx, templates):
    return states[[index[(animal, ctx, t)] for t in templates]].mean(0)  # [n_layers, d]


def pair_vectors(states, index, x, y):
    """The five difference vectors of a pair, each [n_layers, d]."""
    diff = lambda ctx, tpl: (mean_state(states, index, x, ctx, tpl)
                             - mean_state(states, index, y, ctx, tpl))
    v = {"fit_M": diff("moral", FIT_TEMPLATES), "fit_N": diff("neutral", FIT_TEMPLATES),
         "ev_M": diff("moral", EVAL_TEMPLATES), "ev_N": diff("neutral", EVAL_TEMPLATES)}
    v["fit_I"] = v["fit_M"] - v["fit_N"]
    return v


def build_bank(states, index, pairs):
    """Stack vectors of all pairs; slot[(pair_no, kind)] → row in the bank."""
    rows, slot = [], {}
    for p, (x, y) in enumerate(pairs):
        vec = pair_vectors(states, index, x, y)
        for kind in KINDS:
            slot[(p, kind)] = len(rows)
            rows.append(vec[kind])
    bank = np.stack(rows).astype(np.float64)  # [V, n_layers, d]
    return bank, slot


def gram(bank):
    return np.einsum("ild,jld->ijl", bank, bank)  # [V, V, n_layers]


# ---------------------------------------------------------------- LOPO

def d_z(x, axis=0):
    return x.mean(axis) / x.std(axis, ddof=1)


def lopo(G, fit, evm, evn, sign):
    """
    Leave-one-pair-out from the Gram matrix (§5.3).
    fit / evm / evn: bank rows per pair (K,), sign: ±1 per pair (label flip).
    Returns s_M, s_N (K,) and the layer chosen in each fold.
    """
    K = len(fit)
    ss = (sign[:, None] * sign[None, :])[..., None]
    B = G[np.ix_(fit, fit)] * ss            # <fit_i, fit_i'>
    AM = G[np.ix_(evm, fit)] * ss           # <evM_j, fit_i>
    AN = G[np.ix_(evn, fit)] * ss
    diag = np.arange(K)
    # v_k ∝ Σ_{i≠k} fit_i
    norm = np.sqrt(B.sum((0, 1))[None] - 2 * B.sum(1) + B[diag, diag])   # [K(k), L]
    PM = (AM.sum(1)[:, None] - AM) / norm[None]   # [j, k, L] projection of pair j on v_k
    PN = (AN.sum(1)[:, None] - AN) / norm[None]

    layers = np.empty(K, dtype=int)
    for k in range(K):
        train = PM[diag != k, k]                  # [K-1, L]
        layers[k] = int(np.argmax(d_z(train)))
    return PM[diag, diag, layers], PN[diag, diag, layers], layers


def run_lopo(G, slot, pair_nos, sign=None, swap=None, fit_kind="fit_M"):
    """swap[k]=True exchanges moral and neutral context for pair k (interaction null)."""
    K = len(pair_nos)
    sign = np.ones(K) if sign is None else np.asarray(sign, float)
    swap = np.zeros(K, bool) if swap is None else np.asarray(swap, bool)
    fit_alt = {"fit_M": "fit_N", "fit_N": "fit_M", "fit_I": "fit_I"}[fit_kind]
    fit = [slot[(p, fit_alt if sw else fit_kind)] for p, sw in zip(pair_nos, swap)]
    evm = [slot[(p, "ev_N" if sw else "ev_M")] for p, sw in zip(pair_nos, swap)]
    evn = [slot[(p, "ev_M" if sw else "ev_N")] for p, sw in zip(pair_nos, swap)]
    if fit_kind == "fit_I" and swap.any():  # swapping contexts negates fit_M − fit_N
        G = G.copy()
        for p, sw in zip(pair_nos, swap):
            if sw:
                r = slot[(p, "fit_I")]
                G[r, :] *= -1
                G[:, r] *= -1
    sM, sN, layers = lopo(G, np.array(fit), np.array(evm), np.array(evn), sign)
    return sM, sN, layers


def stats(sM, sN):
    return {"d_M": float(d_z(sM)), "d_I": float(d_z(sM - sN))}


# ---------------------------------------------------------------- tests

def exact_permutations(G, slot, pair_nos, observed, fit_kind="fit_M"):
    """Exact one-sided p for d_M (label flips) and d_I (context swaps), 2^K each."""
    K = len(pair_nos)
    flips, swaps = [], []
    for bits in itertools.product([0, 1], repeat=K):
        b = np.array(bits, bool)
        flips.append(stats(*run_lopo(G, slot, pair_nos, sign=np.where(b, -1, 1), fit_kind=fit_kind)[:2])["d_M"])
        swaps.append(stats(*run_lopo(G, slot, pair_nos, swap=b, fit_kind=fit_kind)[:2])["d_I"])
    flips, swaps = np.array(flips), np.array(swaps)
    return {"p_M": float((flips >= observed["d_M"] - 1e-12).mean()),
            "p_I": float((swaps >= observed["d_I"] - 1e-12).mean()),
            "n_perm": len(flips)}


def placebo(Gp, slot_p, n_pairs, rng):
    """§5.4: n_pairs random placebo pairs, random orientation, same pipeline."""
    pool = len(PLACEBO_PAIRS)
    out = {"d_M": [], "d_I": []}
    for _ in range(N_PLACEBO):
        nos = rng.choice(pool, n_pairs, replace=False)
        sign = rng.choice([-1, 1], n_pairs)
        st = stats(*run_lopo(Gp, slot_p, list(nos), sign=sign)[:2])
        out["d_M"].append(st["d_M"])
        out["d_I"].append(st["d_I"])
    return {k: np.array(v) for k, v in out.items()}


def bootstrap_d(x, rng):
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    return np.percentile(d_z(x[idx], axis=1), [2.5, 97.5]).tolist()


def covariate_intercept(sM, dzipf, dtok, rng):
    """§5.5: OLS s_M ~ 1 + ΔZipf + Δtok; bootstrap CI of the intercept."""
    X = np.column_stack([np.ones_like(sM), dzipf, dtok])
    fit = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
    beta = fit(X, sM)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(sM), len(sM))
        if np.linalg.matrix_rank(X[i]) == X.shape[1]:
            boot.append(fit(X[i], sM[i])[0])
    return {"beta": beta.tolist(), "intercept_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "n_boot_valid": len(boot)}


# ---------------------------------------------------------------- ambiguous, B0

def ambiguous_positions(states, index, layer):
    """§5.6: t = (proj − μ_nonfood) / (μ_food − μ_nonfood) with the full v."""
    v = np.mean([mean_state(states, index, p["food"], "moral", FIT_TEMPLATES)[layer]
                 - mean_state(states, index, p["nonfood"], "moral", FIT_TEMPLATES)[layer]
                 for p in PAIRS], axis=0)
    v /= np.linalg.norm(v)
    proj = lambda a, ctx: float(mean_state(states, index, a, ctx, EVAL_TEMPLATES)[layer] @ v)
    out = {}
    for ctx in ("moral", "neutral"):
        mf = np.mean([proj(p["food"], ctx) for p in PAIRS])
        mn = np.mean([proj(p["nonfood"], ctx) for p in PAIRS])
        out[ctx] = {a["animal"]: (proj(a["animal"], ctx) - mn) / (mf - mn) for a in AMBIGUOUS}
    return out


def behavior(key):
    """§6 B0: per-animal readout averaged over paraphrases; gap = nonfood − food."""
    rows = json.loads((LOGS / f"{key}_readout.json").read_text(encoding="utf-8"))["rows"]
    agg = {}
    for r in rows:
        a = agg.setdefault(r["animal"], {"E": [], "mass": [], "yes_no": []})
        for k in ("E", "mass", "yes_no"):
            if k in r:
                a[k].append(r[k])
    per = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in agg.items()}
    out = {"per_animal": per, "mass_mean": float(np.mean([per[a]["mass"] for a in per]))}
    for metric in ("E", "yes_no"):
        g = np.array([per[p["nonfood"]][metric] - per[p["food"]][metric] for p in PAIRS])
        null = np.array([(g * np.array(s)).mean() for s in itertools.product([-1, 1], repeat=len(g))])
        out[metric] = {"gap": g.tolist(), "mean_gap": float(g.mean()), "d_z": float(d_z(g)),
                       "p": float((null >= g.mean() - 1e-12).mean())}
    out["B0_pass"] = bool(out["mass_mean"] >= 0.5 and out["E"]["mean_gap"] > 0 and out["E"]["p"] < 0.05)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    states, index = load_states(args.model)
    n_layers = states.shape[1]
    main_pairs = [(p["food"], p["nonfood"]) for p in PAIRS]
    bank, slot = build_bank(states, index, main_pairs)
    G = gram(bank)
    nos = list(range(len(PAIRS)))

    sM, sN, layers = run_lopo(G, slot, nos)
    obs = stats(sM, sN)
    perm = exact_permutations(G, slot, nos, obs)

    bank_p, slot_p = build_bank(states, index, [(p["a"], p["b"]) for p in PLACEBO_PAIRS])
    Gp = gram(bank_p)
    plc = placebo(Gp, slot_p, len(PAIRS), rng)
    p95 = {k: float(np.percentile(v, 95)) for k, v in plc.items()}

    stim = json.loads(STIMULI.read_text(encoding="utf-8"))["checks"]["animals"]
    dzipf = np.array([stim[f]["zipf"] - stim[n]["zipf"] for f, n in main_pairs])
    dtok = np.array([stim[f]["tokens"][args.model] - stim[n]["tokens"][args.model] for f, n in main_pairs])
    cov = covariate_intercept(sM, dzipf, dtok, rng)

    keep = [i for i, p in enumerate(PAIRS) if p["id"] not in TOKEN_MISMATCH_PAIRS]
    sM_m, sN_m, _ = run_lopo(G, slot, keep)
    homonym = np.array([p["food"] in MEAT_HOMONYM for p in PAIRS])
    sM_i, sN_i, _ = run_lopo(G, slot, nos, fit_kind="fit_I")

    beh = behavior(args.model)
    median_layer = int(np.median(layers))

    h1 = obs["d_M"] > p95["d_M"] and perm["p_M"] < 0.01 and cov["intercept_ci"][0] > 0
    h1_int = obs["d_I"] > p95["d_I"] and perm["p_I"] < 0.01
    result = {
        "model": MODELS[args.model]["name"], "n_layers": n_layers,
        "pairs": [p["id"] for p in PAIRS], "layers": layers.tolist(), "median_layer": median_layer,
        "s_M": sM.tolist(), "s_N": sN.tolist(),
        **obs, "ci_d_M": bootstrap_d(sM, rng), "ci_d_I": bootstrap_d(sM - sN, rng),
        "perm": perm, "placebo_p95": p95,
        "placebo_mean": {k: float(v.mean()) for k, v in plc.items()},
        "covariates": cov,
        "sensitivity": {
            "no_token_mismatch": {**stats(sM_m, sN_m), "n": len(keep)},
            "meat_homonym": {"d_M": float(d_z(sM[homonym])), "n": int(homonym.sum())},
            "no_meat_homonym": {"d_M": float(d_z(sM[~homonym])), "n": int((~homonym).sum())},
            "v_int": stats(sM_i, sN_i),
        },
        "ambiguous_t": ambiguous_positions(states, index, median_layer),
        "behavior": beh,
        "verdict": {"H1": bool(h1), "H1_int": bool(h1_int), "B0": beh["B0_pass"]},
    }
    (LOGS / f"{args.model}_h1.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    # per-fold directions for the ablation stage (v_{-k} at L_k, unit norm)
    fit_all = np.stack([bank[slot[(p, "fit_M")]] for p in nos])          # [K, L, d]
    dirs = []
    for k in nos:
        v = fit_all[[i for i in nos if i != k]].mean(0)[layers[k]]
        dirs.append(v / np.linalg.norm(v))
    torch.save({"model": MODELS[args.model]["name"], "pairs": [p["id"] for p in PAIRS],
                "layers": layers.tolist(), "v": torch.tensor(np.stack(dirs), dtype=torch.float32)},
               LOGS / f"{args.model}_directions.pt")

    print(f"{args.model}: layers per fold {layers.tolist()} (of {n_layers})")
    print(f"  d_M = {obs['d_M']:.2f} CI {result['ci_d_M'][0]:.2f}..{result['ci_d_M'][1]:.2f}"
          f"  placebo95 {p95['d_M']:.2f}  p {perm['p_M']:.4f}")
    print(f"  d_I = {obs['d_I']:.2f} CI {result['ci_d_I'][0]:.2f}..{result['ci_d_I'][1]:.2f}"
          f"  placebo95 {p95['d_I']:.2f}  p {perm['p_I']:.4f}")
    print(f"  covariate intercept CI {cov['intercept_ci'][0]:.3f}..{cov['intercept_ci'][1]:.3f}")
    print(f"  sensitivity {result['sensitivity']}")
    print(f"  ambiguous t {result['ambiguous_t']}")
    print(f"  B0: mass {beh['mass_mean']:.3f}  E gap {beh['E']['mean_gap']:+.3f} (p {beh['E']['p']:.4f})"
          f"  Yes-No gap {beh['yes_no']['mean_gap']:+.3f} (p {beh['yes_no']['p']:.4f})")
    print(f"  verdict {result['verdict']}")


if __name__ == "__main__":
    main()
