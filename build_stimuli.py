"""
build_stimuli.py — stage 1: validate stimuli and write stimuli.json.

Checks (preregistration §3.2, §3.6, §6), per model tokenizer:
  - token count of every animal name as it appears in templates (" name");
  - pairs whose members differ in token count → covariate, not exclusion;
  - word frequency (wordfreq Zipf, log10) and its gap inside each pair;
  - readout answers: digits 1..SCALE_MAX and Yes/No must be single tokens;
  - moral[i] vs neutral[i] template length within ±1 token.
Only tokenizers are loaded, no weights.

Usage:
    python build_stimuli.py
    python build_stimuli.py --models qwen-7b
"""
import argparse
import json
from datetime import date
from pathlib import Path

from transformers import AutoTokenizer
from wordfreq import zipf_frequency

from models import MODELS
from stimuli import (PAIRS, AMBIGUOUS, PLACEBO_PAIRS, TEMPLATES, FIT_TEMPLATES,
                     EVAL_TEMPLATES, MEAT_HOMONYM, SCALE_MAX, READOUT)

OUT = Path(__file__).parent / "stimuli.json"
TEMPLATE_TOL = 1   # §3.6: ±1 token
FREQ_GAP = 1.0     # §3.2: one order of magnitude on the Zipf scale


def n_tok(tok, text):
    return len(tok.encode(text, add_special_tokens=False))


def all_pairs():
    """(group, id, x, y) for every two-member pair; main pairs as (food, nonfood)."""
    for p in PAIRS:
        yield "main", p["id"], p["food"], p["nonfood"]
    for p in PLACEBO_PAIRS:
        yield "placebo", p["id"], p["a"], p["b"]


def animal_table(tokenizers):
    names = [n for _, _, x, y in all_pairs() for n in (x, y)] + [a["animal"] for a in AMBIGUOUS]
    return {n: {"zipf": zipf_frequency(n, "en"),
                "meat_homonym": n in MEAT_HOMONYM,
                "tokens": {m: n_tok(t, " " + n) for m, t in tokenizers.items()}}
            for n in names}


def pair_table(animals, keys):
    rows = []
    for group, pid, x, y in all_pairs():
        ax, ay = animals[x], animals[y]
        rows.append({
            "group": group, "id": pid, "x": x, "y": y,
            "zipf_gap": round(ax["zipf"] - ay["zipf"], 2),
            "freq_flag": abs(ax["zipf"] - ay["zipf"]) > FREQ_GAP,
            "token_gap": {m: ax["tokens"][m] - ay["tokens"][m] for m in keys},
        })
    return rows


def check_readout(tokenizers):
    out = {}
    for m, tok in tokenizers.items():
        digits = {str(d): n_tok(tok, str(d)) for d in range(1, SCALE_MAX + 1)}
        yes_no = {w: n_tok(tok, w) for w in ("Yes", "No")}
        out[m] = {"digits": digits, "yes_no": yes_no,
                  "ok": all(v == 1 for v in list(digits.values()) + list(yes_no.values()))}
    return out


def check_templates(tokenizers):
    """moral[i] vs neutral[i] length with the slot filled by one fixed animal."""
    moral, neutral = TEMPLATES["moral"], TEMPLATES["neutral"]
    assert len(moral) == len(neutral) == len(FIT_TEMPLATES) + len(EVAL_TEMPLATES)
    rows = []
    for i, (tm, tn) in enumerate(zip(moral, neutral)):
        lens = {m: (n_tok(t, tm.format(animal="pig")), n_tok(t, tn.format(animal="pig")))
                for m, t in tokenizers.items()}
        rows.append({"i": i, "lens": lens,
                     "ok": all(abs(a - b) <= TEMPLATE_TOL for a, b in lens.values())})
    return rows


def load_tokenizers(keys):
    tokenizers = {}
    for k in keys:
        try:
            tokenizers[k] = AutoTokenizer.from_pretrained(MODELS[k]["name"])
        except Exception as e:  # gated / not downloaded
            print(f"[skip] {k}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    return tokenizers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()

    tokenizers = load_tokenizers(args.models)
    keys = list(tokenizers)
    animals = animal_table(tokenizers)
    pairs = pair_table(animals, keys)
    readout = check_readout(tokenizers)
    templates = check_templates(tokenizers)

    print(f"\nmodels: {keys}")
    print(f"{'group':<8}{'id':<5}{'x':<12}{'y':<12}{'dzipf':>6}  tokens x-y per model")
    for r in pairs:
        gaps = " ".join(f"{r['token_gap'][m]:+d}" for m in keys)
        mark = " <- freq" if r["freq_flag"] else ""
        print(f"{r['group']:<8}{r['id']:<5}{r['x']:<12}{r['y']:<12}{r['zipf_gap']:>+6.2f}  {gaps}{mark}")
    n_tok_bad = sum(any(r["token_gap"].values()) for r in pairs if r["group"] == "main")
    n_freq_bad = sum(r["freq_flag"] for r in pairs if r["group"] == "main")
    print(f"main pairs: token mismatch {n_tok_bad}/{len(PAIRS)}, freq gap >{FREQ_GAP} {n_freq_bad}/{len(PAIRS)}")

    print("\nreadout:")
    for m, r in readout.items():
        print(f"  {m}: ok={r['ok']}  digits={set(r['digits'].values())}  yes/no={r['yes_no']}")

    print("\ntemplates (moral, neutral):")
    for r in templates:
        print(f"  {r['i']} ok={r['ok']}  {r['lens']}")

    OUT.write_text(json.dumps({
        "built": date.today().isoformat(),
        "models": {k: MODELS[k]["name"] for k in keys},
        "pairs": PAIRS, "ambiguous": AMBIGUOUS, "placebo": PLACEBO_PAIRS,
        "templates": TEMPLATES, "fit_templates": FIT_TEMPLATES, "eval_templates": EVAL_TEMPLATES,
        "readout": READOUT, "scale_max": SCALE_MAX,
        "checks": {"animals": animals, "pairs": pairs, "readout": readout, "templates": templates},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
