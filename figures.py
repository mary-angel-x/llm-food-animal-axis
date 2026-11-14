"""
figures.py — README figures from the saved logs (no model needed).

    python figures.py     -> figures/fig1_food_axis.png, figures/fig2_moral_circle.png, figures/fig3_seafood.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part2.entities import CATEGORIES, MAIN
from stimuli import PAIRS

ROOT = Path(__file__).resolve().parent
LOGS, LOGS2, OUT = ROOT / "logs", ROOT / "part2" / "logs", ROOT / "figures"
MODELS = {"qwen-7b": "Qwen2.5-7B", "llama-8b": "Llama-3.1-8B", "gemma-2b": "Gemma-2-2B"}
ROLE = {"qwen-7b": "discovery", "llama-8b": "confirmation", "gemma-2b": "confirmation"}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig1() -> None:
    """A: the axis in three families. B: sentient but edible (Qwen, 13 pairs). C: ablation against controls."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1, 1.15, 1.1]})

    for row, key in enumerate(MODELS):
        h1 = load(LOGS / f"{key}_v03.json")["H1"]
        low, high = h1["ci_d_M"]
        a.plot([low, high], [row, row], color=BLUE, linewidth=2, solid_capstyle="round")
        a.plot(h1["d_M"], row, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        a.plot([h1["placebo_p95"]] * 2, [row - 0.22, row + 0.22], color=INK2, linewidth=2)
        a.text(h1["d_M"], row - 0.2, f"{h1['d_M']:.2f}", ha="center", va="bottom", color=INK)
    a.text(0.62, -0.42, "placebo pairs,\n95th percentile", color=INK2, fontsize=8.5, va="center", ha="left")
    a.set_yticks(range(len(MODELS)), [f"{name}\n({ROLE[key]})" for key, name in MODELS.items()])
    a.set_ylim(len(MODELS) - 0.4, -0.75)
    a.set_xlim(0, 5)
    a.grid(axis="y", visible=False)
    a.set_xlabel("held-out-pair effect size d (95% CI)")
    a.set_title("A  A food-status axis in three model families")

    per = load(LOGS / "qwen-7b_v03.json")["behavior"]["per_animal"]
    xs = np.array([per[p["food"]]["E"] - per[p["nonfood"]]["E"] for p in PAIRS])
    ys = np.array([per[p["food"]]["accept"] - per[p["nonfood"]]["accept"] for p in PAIRS])
    b.axhline(0, color=AXIS, linewidth=1)
    b.axvline(0, color=AXIS, linewidth=1)
    b.plot(xs, ys, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
    nudge = {"duck": (6, -12, "left"), "hen": (6, -11, "left"), "salmon": (-6, -12, "right"), "cow": (-6, 5, "right")}
    for p, x, y in zip(PAIRS, xs, ys):
        dx, dy, ha = nudge.get(p["food"], (6, 4, "left"))
        b.annotate(f"{p['food']} / {p['nonfood']}", (x, y), xytext=(dx, dy), textcoords="offset points",
                   fontsize=7.5, color=INK2, ha=ha)
    b.set_xlabel("rated capacity to suffer, farmed − non-farmed (1–9 scale)")
    b.set_ylabel("acceptability of harm, farmed − non-farmed")
    b.set_title("B  Sentient but edible (Qwen, 13 pairs)")
    b.set_ylim(-0.12, 1.52)
    b.text(0.98, 0.7, f"means: suffering {xs.mean():+.2f},\nharm {ys.mean():+.2f}", transform=b.transAxes,
           ha="right", fontsize=8.5, color=INK2)

    rows = [("food axis ablated", "C", BLUE), ("placebo directions, 95th pct", "placebo_p95", ORANGE),
            ("random directions, 95th pct", "random_p95", AQUA)]
    files = {"gemma-2b": "gemma-2b_h2_all_pos.json", "qwen-7b": "qwen-7b_h2_all_pos_b6.json"}
    for group, (key, name) in enumerate(files.items()):
        h2 = load(LOGS / name)["H2b"]
        for i, (label, field, color) in enumerate(rows):
            y = group + (i - 1) * 0.26
            c.barh(y, h2[field], height=0.22, color=color, label=label if group == 0 else None)
            c.text(h2[field] + 0.004, y, f"{h2[field]:.3f}", va="center", fontsize=8.5, color=INK)
        note = "confirmation; 99% of gap, floor effect" if key == "gemma-2b" else "exploratory; 20% of gap"
        c.text(0.0, group - 0.47, note, fontsize=8, color=INK2, va="bottom")
    c.set_yticks(range(len(files)), [MODELS[k] for k in files])
    c.set_ylim(1.5, -0.75)
    c.set_xlim(0, 0.37)
    c.grid(axis="y", visible=False)
    c.set_xlabel("closure of the acceptability gap (scale points)")
    c.set_title("C  Ablating the axis narrows the gap")
    c.set_xticks([0, 0.05, 0.10, 0.15])
    c.legend(loc="lower right", fontsize=8.5, handlelength=1.2)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "fig1_food_axis.png", dpi=200)
    plt.close(fig)


def fig2() -> None:
    """Part 2, Qwen: rated suffering against acceptability of harm for 60 entities, category means labelled."""
    score = load(LOGS2 / "qwen-7b_part2.json")["score"]
    highlight = {"human": ORANGE, "ai": BLUE, "farm": AQUA}
    names = {"human": "humans", "companion": "companion animals", "farm": "farmed animals", "wild": "wild animals",
             "seafood_inv": "seafood invertebrates", "other_inv": "other invertebrates", "plant": "plants",
             "object": "objects", "machine": "machines", "ai": "AI"}
    offset = {"human": (0, -14), "companion": (0, 10), "farm": (8, 10), "wild": (-52, -3), "seafood_inv": (10, 4),
              "other_inv": (0, 10), "plant": (-4, 10), "object": (-16, -16), "machine": (24, -15), "ai": (0, 11)}
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for category in MAIN:
        x = np.array([score["suffer"][n] for n in CATEGORIES[category]])
        y = np.array([score["accept"][n] for n in CATEGORIES[category]])
        color = highlight.get(category, MUTED)
        ax.plot(x, y, "o", color=color, markersize=6, alpha=0.85 if category in highlight else 0.45,
                markeredgecolor=SURFACE, markeredgewidth=1, label=names[category] if category in highlight else None)
        ax.plot(x.mean(), y.mean(), "D", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(names[category], (x.mean(), y.mean()), xytext=offset[category], textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight="bold" if category in highlight else "normal")
    ax.plot([], [], "o", color=MUTED, markersize=6, alpha=0.6, label="other categories")
    ax.plot([], [], "D", color=MUTED, markersize=8, label="category mean")
    ax.set_xlabel("rated capacity to suffer (1–9)")
    ax.set_ylabel("acceptability of harming or destroying it (1–9)")
    ax.set_xlim(0.5, 9)
    ax.set_ylim(0.8, 4.1)
    ax.set_title("Suffering does not set the boundary of harm (Qwen2.5-7B, 60 entities)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_moral_circle.png", dpi=200)
    plt.close(fig)


def fig3() -> None:
    """Part 2, P2: position on the Part 1 food axis, seafood invertebrates against other invertebrates."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, (key, name) in zip(axes, MODELS.items()):
        result = load(LOGS2 / f"{key}_part2.json")
        t = result["t"]["moral"]
        for row, (category, color, label) in enumerate([("seafood_inv", BLUE, "seafood"), ("other_inv", ORANGE, "other")]):
            values = [t[n] for n in CATEGORIES[category]]
            ax.plot(values, [row] * 6, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.plot([np.mean(values)] * 2, [row - 0.28, row + 0.28], color=color, linewidth=2)
            ordered = sorted(zip(values, CATEGORIES[category]))
            for (v, n), dy in zip(ordered, [10, -17, 21, -28, 10, -17]):
                ax.annotate(n, (v, row), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.axvline(1, color=AXIS, linewidth=1)
        d, p = result["P2"]
        ax.set_title(f"{name}: {d:+.2f}, p = {p:.4f}", fontsize=10)
        ax.set_yticks([0, 1], ["seafood\ninvertebrates", "other\ninvertebrates"])
        ax.set_ylim(1.6, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("position on the food axis\n(0 = non-farmed vertebrates, 1 = farmed)")
    fig.suptitle("The food axis, built on vertebrates only, separates shrimp from jellyfish", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_seafood.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig1()
    fig2()
    fig3()
