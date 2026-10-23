# Does Moral Weight Track Sentience or Cuisine?

Is the farmed / non-farmed asymmetry in LLM moral judgement (Jotautaitė et al., 2025, arXiv:2508.11534) carried by a linear "food status" direction in the residual stream, and does removing that direction close the gap in how much the model thinks an animal can suffer?

Design, metrics and decision thresholds are fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) (in Russian) before any model run; every later change is logged in its §11.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1 stimuli | `build_stimuli.py` | `stimuli.json` — tokenization, frequency, template-length checks |
| 2 extraction | `extract.py --model M` | `logs/M_states.pt`, `logs/M_readout.json` |
| 3 H1 / H1-int / B0 | `analyze.py --model M` | `logs/M_h1.json`, `logs/M_directions.pt` |
| 4 H2 | `ablate.py --model M [--variant ...]`, then `--summary` | `logs/M_ablate_*.json`, `logs/M_h2_*.json` |

`test_analyze.py` checks the stage 3 pipeline on synthetic data (Gram-matrix LOPO = brute force; planted signal is found; signal present in both contexts gives no interaction).

Models: `qwen-7b` (discovery), `llama-8b` and `gemma-2b` (confirmation), `qwen-14b` (scale extension, 4-bit NF4; preregistration §13). Run from the project venv:

```
..\venv\Scripts\python.exe extract.py --model qwen-7b
..\venv\Scripts\python.exe analyze.py --model qwen-7b
..\venv\Scripts\python.exe ablate.py --model qwen-7b
..\venv\Scripts\python.exe ablate.py --model qwen-7b --summary
```

Runtime on an 8 GB GPU with CPU offload: ~2 s per forward pass; extraction ≈ 1 h per model, ablation with 200 control directions ≈ 9 h per model per variant (resumable).
