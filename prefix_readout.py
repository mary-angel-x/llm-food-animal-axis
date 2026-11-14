"""
prefix_readout.py — the digit readout with the assistant turn started as "Rating: " (preregistration §14).

    python prefix_readout.py --model llama-8b     -> logs/<model>_readout_prefix.json
    python prefix_readout.py --summary            -> logs/prefix_summary.json (gates K and V, B0 and B0b per model)

Llama-8B opens its answer with a preamble, so its unprefixed digit readout is invalid (§6). Qwen-7B and Gemma-2B
answer with a digit either way and calibrate the new format against the old one (gate K).
"""
import argparse
import json

import numpy as np

import extract
from analyze import perm_mean_test
from extract import LOGS, load, readout_prompts, run_readout
from models import MODELS
from part2.analyze import spearman
from stimuli import DIGIT_READOUTS, PAIRS

CALIBRATION = ["qwen-7b", "gemma-2b"]     # models whose unprefixed readout is valid
RHO_MIN, MASS_MIN = 0.8, 0.5


def per_animal(path) -> dict:
    """{animal: {kind: E, kind_mass: mass}}, mean over the three paraphrases."""
    rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["rows"] if r["kind"] in DIGIT_READOUTS]
    out = {}
    for animal in dict.fromkeys(r["animal"] for r in rows):
        out[animal] = {}
        for kind in DIGIT_READOUTS:
            mine = [r for r in rows if r["animal"] == animal and r["kind"] == kind]
            out[animal][kind] = float(np.mean([r["E"] for r in mine]))
            out[animal][f"{kind}_mass"] = float(np.mean([r["mass"] for r in mine]))
    return out


def run(key: str) -> None:
    extract.ANSWER_PREFIX = extract.PREFIX
    model, tokenizer = load(key)
    rows = [r for r in readout_prompts() if r["kind"] in DIGIT_READOUTS]
    run_readout(model, tokenizer, rows, 1)
    (LOGS / f"{key}_readout_prefix.json").write_text(
        json.dumps({"model": MODELS[key]["name"], "prefix": extract.PREFIX, "rows": rows}, indent=1), encoding="utf-8")
    print(f"-> {LOGS / f'{key}_readout_prefix.json'}", flush=True)


def summary() -> None:
    out = {}
    for key in MODELS:
        path = LOGS / f"{key}_readout_prefix.json"
        if not path.exists():
            continue
        new, old = per_animal(path), per_animal(LOGS / f"{key}_readout.json")
        gap = lambda per, kind, sign: np.array([sign * (per[p["food"]][kind] - per[p["nonfood"]][kind]) for p in PAIRS])
        result = {"mass": {kind: float(np.mean([new[a][f"{kind}_mass"] for a in new])) for kind in DIGIT_READOUTS},
                  "old_mass": {kind: float(np.mean([old[a][f"{kind}_mass"] for a in old])) for kind in DIGIT_READOUTS},
                  "rho_old_new": {kind: spearman([old[a][kind] for a in new], [new[a][kind] for a in new])
                                  for kind in DIGIT_READOUTS},
                  "B0": perm_mean_test(gap(new, "scale", -1)),      # suffering: non-food − food
                  "B0b": perm_mean_test(gap(new, "accept", 1)),     # acceptability: food − non-food
                  "old_B0b_gap": float(gap(old, "accept", 1).mean()), "per_animal": new}
        result["V"] = bool(result["mass"]["accept"] >= MASS_MIN)
        result["verdict"] = {"B0": bool(result["mass"]["scale"] >= MASS_MIN and result["B0"]["mean_gap"] > 0
                                        and result["B0"]["p"] < 0.05),
                             "B0b": bool(result["V"] and result["B0b"]["mean_gap"] > 0 and result["B0b"]["p"] < 0.05)}
        out[key] = result
    done = [k for k in CALIBRATION if k in out]
    out["K"] = bool(len(done) == len(CALIBRATION) and all(
        out[k]["rho_old_new"]["accept"] >= RHO_MIN and out[k]["B0b"]["mean_gap"] * out[k]["old_B0b_gap"] > 0 for k in done))
    (LOGS / "prefix_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key in MODELS:
        if key in out:
            r = out[key]
            print(f"{key}: mass suffer/accept {r['mass']['scale']:.2f}/{r['mass']['accept']:.2f} "
                  f"(unprefixed {r['old_mass']['scale']:.2f}/{r['old_mass']['accept']:.2f}); "
                  f"rho old-new suffer/accept {r['rho_old_new']['scale']:+.2f}/{r['rho_old_new']['accept']:+.2f}; "
                  f"B0b {r['B0b']['mean_gap']:+.3f} p={r['B0b']['p']:.4f} (unprefixed {r['old_B0b_gap']:+.3f}); "
                  f"B0 {r['B0']['mean_gap']:+.3f} p={r['B0']['p']:.4f}; V {r['V']}")
    print(f"gate K (calibration, {', '.join(done)} of {', '.join(CALIBRATION)}): {'PASS' if out['K'] else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    summary() if args.summary else run(args.model)
