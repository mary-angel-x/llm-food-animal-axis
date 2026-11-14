"""
analyze.py — Part 2 gates and predictions for one model (part2/PREREGISTRATION.md).

    python -m part2.analyze --model qwen-7b     -> part2/logs/<model>_part2.md and .json
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from models import MODELS
from stimuli import EVAL_TEMPLATES, PAIRS

from .entities import CATEGORIES, MAIN, NON_VERTEBRATE, PART1_ANIMALS, SEA_OTHER_INV, SELF, all_entities, category_of

ROOT = Path(__file__).resolve().parent
LOGS, PART1_LOGS = ROOT / "logs", ROOT.parent / "logs"
N_BOOT = 2000


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x))
    out[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        tied = x == value
        out[tied] = out[tied].mean()
    return out


def spearman(x, y) -> float:
    return float(np.corrcoef(ranks(np.asarray(x, float)), ranks(np.asarray(y, float)))[0, 1])


def exact_permutation(a, b) -> tuple:
    """(mean(a) - mean(b), one-sided p for 'a > b', two-sided p) over every split of the pooled values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, observed = np.concatenate([a, b]), a.mean() - b.mean()
    diffs = []
    for chosen in combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(chosen)] = True
        diffs.append(pooled[mask].mean() - pooled[~mask].mean())
    diffs = np.array(diffs)
    return float(observed), float((diffs >= observed - 1e-12).mean()), float((np.abs(diffs) >= abs(observed) - 1e-12).mean())


def behavior(key: str, suffix: str = '') -> tuple:
    """({kind: {entity: E}}, {kind: {entity: mass}}), mean over the three paraphrases."""
    rows = json.loads((LOGS / f"{key}_readout{suffix}.json").read_text(encoding="utf-8"))["rows"]
    score = {kind: {} for kind in ("suffer", "accept")}
    mass = {kind: {} for kind in ("suffer", "accept")}
    for entity in all_entities():
        for kind in score:
            mine = [r for r in rows if r["entity"] == entity and r["kind"] == kind]
            score[kind][entity] = float(np.mean([r["E"] for r in mine]))
            mass[kind][entity] = float(np.mean([r["mass"] for r in mine]))
    return score, mass


def part1_accept(key: str, suffix: str = '') -> dict:
    rows = json.loads((PART1_LOGS / f"{key}_readout{suffix}.json").read_text(encoding="utf-8"))["rows"]
    return {a: float(np.mean([r["E"] for r in rows if r["animal"] == a and r["kind"] == "accept"])) for a in PART1_ANIMALS}


def positions(key: str) -> dict:
    """{context: {entity: t}} on the frozen Part 1 axis; 0 = non-food anchor, 1 = food anchor."""
    saved = torch.load(PART1_LOGS / f"{key}_directions_v03.pt")
    layer = saved["layers"][0]
    v = saved["v"].float().mean(0)
    v = v / v.norm()
    part1 = torch.load(PART1_LOGS / f"{key}_states.pt")
    part2 = torch.load(LOGS / f"{key}_states.pt")
    assert part2["layer"] == layer
    food, nonfood = [p["food"] for p in PAIRS], [p["nonfood"] for p in PAIRS]
    out = {}
    for ctx in ("moral", "neutral"):
        def anchor(names: list) -> float:
            rows = [i for i, r in enumerate(part1["rows"])
                    if r["animal"] in names and r["context"] == ctx and r["template"] in EVAL_TEMPLATES]
            return float((part1["states"][rows, layer].float() @ v).mean())
        low, high = anchor(nonfood), anchor(food)
        assert high > low, "direction sign: food must project above non-food"
        out[ctx] = {}
        for entity in all_entities():
            rows = [i for i, r in enumerate(part2["rows"]) if r["entity"] == entity and r["context"] == ctx]
            s = float((part2["states"][rows].float() @ v).mean())
            out[ctx][entity] = (s - low) / (high - low)
    return out


def residuals(score: dict) -> dict:
    """Leave-category-out residuals of accept ~ suffer: {category: [six residuals]}; self uses all ten categories."""
    def fit(names: list) -> tuple:
        x, y = np.array([score["suffer"][n] for n in names]), np.array([score["accept"][n] for n in names])
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept
    out = {}
    for category in MAIN + ["self"]:
        held = SELF if category == "self" else CATEGORIES[category]
        slope, intercept = fit([n for c in MAIN if c != category for n in CATEGORIES[c]])
        out[category] = [score["accept"][n] - (slope * score["suffer"][n] + intercept) for n in held]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--answer-prefix", action="store_true", help="read the prefixed readout (Part 1 §14)")
    args = parser.parse_args()
    key, suffix = args.model, "_prefix" if args.answer_prefix else ""
    rng = np.random.default_rng(0)
    score, mass = behavior(key, suffix)
    t = positions(key)
    category = category_of()
    cat_mean = lambda values, c: float(np.mean([values[n] for n in (SELF if c == "self" else CATEGORIES[c])]))
    of = lambda values, names: [values[n] for n in names]
    result = {"model": key}

    lines = [f"# Part 2: {MODELS[key]['short']}" + (' (readout with the "Rating: " prefix)' if suffix else ""), "",
             "| category | suffer | accept | digit mass (suffer / accept) | t moral | t neutral |", "|---|---|---|---|---|---|"]
    for c in MAIN + ["self"]:
        lines.append(f"| {c} | {cat_mean(score['suffer'], c):.2f} | {cat_mean(score['accept'], c):.2f} | "
                     f"{cat_mean(mass['suffer'], c):.2f} / {cat_mean(mass['accept'], c):.2f} | "
                     f"{cat_mean(t['moral'], c):+.2f} | {cat_mean(t['neutral'], c):+.2f} |")

    gate_v = all(cat_mean(mass[kind], c) >= 0.5 for kind in mass for c in MAIN)
    old = part1_accept(key, suffix)
    rho_b = spearman(of(score["accept"], PART1_ANIMALS), of(old, PART1_ANIMALS))
    gate_b = rho_b >= 0.7
    lines += ["", "## Gates", "", f"- **V** digit mass >= 0.5 in every category: **{'PASS' if gate_v else 'FAIL'}**",
              f"- **B** new accept vs Part 1 accept on 12 animals: rho = {rho_b:+.2f} (threshold 0.7): "
              f"**{'PASS' if gate_b else 'FAIL'}**"]
    accept_means = [cat_mean(score["accept"], c) for c in MAIN]
    spread = max(accept_means) - min(accept_means)
    gate_r = spread >= 1.0
    lines += [f"- **R** spread of category means of accept = {spread:.2f} (threshold 1.0): "
              f"**{'PASS' if gate_r else 'FAIL: P1, P3, P4, P6 not testable'}**", "", "## Predictions", ""]
    result.update({"gate_R": gate_r, "accept_spread": spread})

    sea, other = CATEGORIES["seafood_inv"], CATEGORIES["other_inv"]
    d1, p1, _ = exact_permutation(of(score["accept"], sea), of(score["accept"], other))
    d1s = np.mean(of(score["accept"], sea)) - np.mean(of(score["accept"], SEA_OTHER_INV))
    d1f, p1f, _ = exact_permutation(of(score["suffer"], other), of(score["suffer"], sea))
    lines.append(f"- **P1** accept(seafood_inv) - accept(other_inv) = {d1:+.2f}, p = {p1:.4f}: "
                 f"**{'yes' if d1 > 0 and p1 < 0.05 else 'no'}**; sea-only {d1s:+.2f}; "
                 f"suffer(other) - suffer(seafood) = {d1f:+.2f} (p = {p1f:.4f})")
    d2, p2, _ = exact_permutation(of(t["moral"], sea), of(t["moral"], other))
    d2s = np.mean(of(t["moral"], sea)) - np.mean(of(t["moral"], SEA_OTHER_INV))
    d2n, p2n, _ = exact_permutation(of(t["neutral"], sea), of(t["neutral"], other))
    lines.append(f"- **P2** t(seafood_inv) - t(other_inv), moral = {d2:+.2f}, p = {p2:.4f}: "
                 f"**{'yes' if d2 > 0 and p2 < 0.05 else 'no'}**; sea-only {d2s:+.2f}; neutral {d2n:+.2f} (p = {p2n:.4f})")

    main_names = [n for c in MAIN for n in CATEGORIES[c]]
    rho3 = spearman(of(score["suffer"], main_names), of(score["accept"], main_names))
    lines.append(f"- **P3** rho(suffer, accept) over 60 entities = {rho3:+.2f} (threshold < -0.5): "
                 f"**{'yes' if rho3 < -0.5 else 'no'}**")

    res = residuals(score)
    means = {c: float(np.mean(res[c])) for c in MAIN}
    largest = max(means, key=lambda c: abs(means[c]))
    ai = np.array(res["ai"])
    boot = [ai[rng.integers(0, len(ai), len(ai))].mean() for _ in range(N_BOOT)]
    low, high = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    edge = largest == "ai" and (low > 0 or high < 0)
    lines.append(f"- **P4** mean residual of AI = {means['ai']:+.2f}, CI [{low:+.2f}, {high:+.2f}]; largest |residual| is "
                 f"`{largest}` ({means[largest]:+.2f}): **{'AI is an edge' if edge else 'no'}**")
    lines.append("  - residuals by category: " + ", ".join(f"{c} {means[c]:+.2f}" for c in MAIN)
                 + f"; self {np.mean(res['self']):+.2f}")

    d5, p5, _ = exact_permutation(of(score["suffer"], CATEGORIES["ai"]), of(score["suffer"], CATEGORIES["machine"]))
    d5a, _, p5a = exact_permutation(of(score["accept"], CATEGORIES["ai"]), of(score["accept"], CATEGORIES["machine"]))
    lines.append(f"- **P5** suffer(ai) - suffer(machine) = {d5:+.2f}, p = {p5:.4f}: "
                 f"**{'yes' if d5 > 0 and p5 < 0.05 else 'no'}**; accept(ai) - accept(machine) = {d5a:+.2f} "
                 f"(two-sided p = {p5a:.4f})")

    nonvert = [n for c in NON_VERTEBRATE for n in CATEGORIES[c]]
    rho6 = spearman(of(t["moral"], nonvert), of(score["accept"], nonvert))
    reading = "general axis" if rho6 > 0.5 else "cuisine-specific" if abs(rho6) < 0.3 else "unclear"
    lines.append(f"- **P6** rho(t moral, accept) over 36 non-vertebrates = {rho6:+.2f}: **{reading}**")

    lines += ["", "## Self (descriptive)", "", "| entity | suffer | accept | residual | t moral |", "|---|---|---|---|---|"]
    for name, r in zip(SELF, res["self"]):
        lines.append(f"| {name} | {score['suffer'][name]:.2f} | {score['accept'][name]:.2f} | {r:+.2f} | {t['moral'][name]:+.2f} |")
    lines += ["", "## Every entity", "", "| entity | category | suffer | accept | t moral |", "|---|---|---|---|---|"]
    for name in all_entities():
        lines.append(f"| {name} | {category[name]} | {score['suffer'][name]:.2f} | {score['accept'][name]:.2f} | "
                     f"{t['moral'][name]:+.2f} |")

    result.update({"gate_V": gate_v, "gate_B": gate_b, "rho_B": rho_b, "P1": [d1, p1], "P2": [d2, p2], "P3": rho3,
                   "P4": {"ai": means["ai"], "ci": [low, high], "largest": largest, "all": means}, "P5": [d5, p5],
                   "P6": rho6, "score": score, "mass": mass, "t": t})
    (LOGS / f"{key}_part2{suffix}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (LOGS / f"{key}_part2{suffix}.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
