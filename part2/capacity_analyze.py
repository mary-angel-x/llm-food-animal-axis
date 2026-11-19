"""
capacity_analyze.py — decision for the 2026-11-18 amendment (part2/PREREGISTRATION.md): does the food effect
survive controlling for reasoning capacity?

    python -m part2.capacity_analyze          (from the moral_AI folder)

M0: accept ~ 1 + food + suffer ; M1: accept ~ 1 + food + suffer + reason, over 60 entities.
95% bootstrap CI for the food coefficient in M1 (2000 resamples of entities, fixed regression).
Gate V: digit mass >= 0.5 in every category for suffer, accept and reason.
Llama-8B (prefix) is reported as exploratory and does not enter the decision.
"""
import json

import numpy as np

from .entities import CATEGORIES, all_entities
from .run import LOGS

FOOD_CATS = {"farm", "seafood_inv"}
MODELS = ["qwen-7b", "gemma-2b"]           # decision models (no prefix)
RNG = np.random.default_rng(0)


def load_means(model, path):
    """Per entity: mean E for suffer, accept, reason; and gate V (mean digit mass per category)."""
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    sums = {k: {} for k in ("suffer", "accept", "reason")}
    masses = {k: {c: [] for c in CATEGORIES} for k in ("suffer", "accept", "reason")}
    cat_of = {e: c for c, es in CATEGORIES.items() for e in es}
    for r in rows:
        k = r["kind"]
        if k not in sums or r["entity"] not in cat_of:   # self-descriptions are not in the 60 entities
            continue
        sums[k].setdefault(r["entity"], []).append(r["E"])
        masses[k][cat_of[r["entity"]]].append(r["mass"])
    means = {k: {e: float(np.mean(v)) for e, v in sums[k].items()} for k in sums}
    gate = {k: min(float(np.mean(v)) for v in masses[k].values() if v) for k in masses if any(masses[k].values())}
    return means, gate


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    ents = [e for c in CATEGORIES for e in CATEGORIES[c]]     # the 60 entities; self-descriptions excluded
    verdicts = []
    food = np.array([1.0 if any(e in CATEGORIES[c] for c in FOOD_CATS) else 0.0 for e in ents])
    for model in MODELS:
        base = LOGS / f"{model}_readout.json"
        cap = LOGS / f"{model}_capacity.json"
        if not cap.exists():
            print(f"{model}: capacity file missing, skipped")
            continue
        m_base, g_base = load_means(model, base)
        m_cap, g_cap = load_means(model, cap)
        gate_reason = g_cap["reason"]
        passed = min(g_base["suffer"], g_base["accept"], gate_reason) >= 0.5
        suffer = np.array([m_base["suffer"][e] for e in ents])
        accept = np.array([m_base["accept"][e] for e in ents])
        reason = np.array([m_cap["reason"][e] for e in ents])

        X0 = np.column_stack([np.ones(len(ents)), food, suffer])
        X1 = np.column_stack([np.ones(len(ents)), food, suffer, reason])
        b1 = ols(accept, X1)
        boots = []
        n = len(ents)
        for _ in range(2000):
            idx = RNG.integers(0, n, n)
            boots.append(ols(accept[idx], X1[idx])[1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"== {model}  (gate V: suffer {g_base['suffer']:.2f}, accept {g_base['accept']:.2f}, reason {gate_reason:.2f} "
              f"-> {'PASS' if passed else 'FAIL, not read'})")
        print(f"   M1 food = {b1[1]:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]; reason = {b1[3]:+.3f}; "
              f"M0 food = {ols(accept, X0)[1]:+.3f}")
        print(f"   corr(reason, suffer) over 60 entities = {np.corrcoef(reason, suffer)[0, 1]:+.2f}")
        if not passed:
            verdict = "not read (gate V failed)"
        elif lo > 0:
            verdict = "food survives"            # coefficient > 0 and CI excludes 0
        else:
            verdict = "capacity explains"        # coefficient <= 0 or CI includes 0
        print(f"   verdict: {verdict}")
        verdicts.append(verdict)
    decided = [v for v in verdicts if v in ("food survives", "capacity explains")]
    print("== decision (>=2 models read, same verdict):")
    for label in ("food survives", "capacity explains"):
        if decided.count(label) >= 2 and len(decided) == decided.count(label):
            print(f"   {label}")
            return
    print("   not decided")


if __name__ == "__main__":
    main()
