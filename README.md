# Does Moral Weight Track Sentience or Cuisine?

*A linear "food status" direction in LLMs drives harm permissibility, not sentience attribution.*

LLMs judge harm to farmed animals as more acceptable than harm to comparable non-farmed animals (Jotautaitė et al., 2025, arXiv:2508.11534). We ask whether this asymmetry is carried by a linear "food status" direction in the residual stream.

## Findings

1. **A food-status axis exists.** Farmed vs non-farmed animals (pig/dog, turkey/eagle, trout/goldfish, …) separate along a single direction in middle-to-late layers, in all three model families (held-out-pair effect size d = 1.55–2.58, exact permutation p ≤ 0.0005). It survives controls for word frequency and token length; ambiguous animals (rabbit, guinea pig) fall in between.
2. **Moral framing amplifies it.** The farmed − non-farmed difference is larger and more consistent in harm/suffering prompts than in neutral ones.
3. **Sentient but edible.** Models do *not* rate farmed animals as less able to suffer (Qwen even rates them higher), yet rate harming them as *more permissible* (Qwen +0.70, Gemma +0.13 on a 1–9 scale, p ≤ 0.0002).
4. **The axis is causal (Gemma-2-2B).** Ablating it closes 99% of the permissibility gap; random and placebo directions close ≤ 2%.

**Caveats.** Gemma's ratings sit near the scale floor, so the ablation cannot yet separate "farmed becomes like non-farmed" from "model becomes stricter overall"; the Qwen ablation that resolves this is in progress. Llama's digit readout failed the preregistered validity check, so H2 was not tested on it. Qwen was the discovery model; its neural results are exploratory.

Full numbers: [`RESULTS.md`](RESULTS.md) (in Russian).

Design, metrics and decision thresholds are fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) (in Russian) before any model run; every later change is logged in its §11.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1 stimuli | `build_stimuli.py` | `stimuli.json` — tokenization, frequency, template-length checks |
| 2 extraction | `extract.py --model M` | `logs/M_states.pt`, `logs/M_readout.json` |
| 3 H1 / H1-int / B0 | `analyze.py --model M` | `logs/M_h1.json`, `logs/M_directions.pt` |
| 4 H2 | `ablate.py --model M [--variant ...]`, then `--summary` | `logs/M_ablate_*.json`, `logs/M_h2_*.json` |

`test_analyze.py` checks the stage 3 pipeline on synthetic data (Gram-matrix LOPO = brute force; planted signal is found; signal present in both contexts gives no interaction).

Models: `qwen-7b` (discovery), `llama-8b` and `gemma-2b` (confirmation). Run from the project venv:

```
..\venv\Scripts\python.exe extract.py --model qwen-7b
..\venv\Scripts\python.exe analyze.py --model qwen-7b
..\venv\Scripts\python.exe ablate.py --model qwen-7b
..\venv\Scripts\python.exe ablate.py --model qwen-7b --summary
```

Runtime on an 8 GB GPU with CPU offload: ~2 s per forward pass; extraction ≈ 1 h per model, ablation with 200 control directions ≈ 9 h per model per variant (resumable).
