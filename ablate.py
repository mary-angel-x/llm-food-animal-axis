"""
ablate.py — stage 4: directional ablation and controls (preregistration §7).

Fold k (held-out pair k): remove v_{-k} at the output of decoder layer L_k,
then read out food_k and nonfood_k. Controls per fold, at the same layer:
100 random unit directions and 100 placebo directions (v from 13 random
placebo pairs, random orientation). Control j across all folds is one
replicate of the closure statistic C.

Variants: all_pos (primary), last_pos (secondary), all_layers (exploratory,
v_{-k} removed after every layer).

Results are written after every fold, so a stopped run resumes where it left.

Output: logs/<model>_ablate_<variant>.json

Usage:
    python ablate.py --model qwen-7b
    python ablate.py --model qwen-7b --variant last_pos
    python ablate.py --model qwen-7b --summary
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze import load_states, mean_state, d_z
from extract import load, chat, decoder_layers, forward_last, answer_ids, readout_scores
from models import MODELS
from stimuli import PAIRS, PLACEBO_PAIRS, FIT_TEMPLATES, READOUT

LOGS = Path(__file__).parent / "logs"
SEED = 0
N_CONTROLS = 100


class Ablator:
    """Forward hooks that project v out of a layer's output."""

    def __init__(self, layers, layer_ids, last_only):
        self.v = None
        self.last_only = last_only
        self.handles = [layers[i].register_forward_hook(self._hook) for i in layer_ids]

    def _hook(self, module, args, output):
        if self.v is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        v = self.v.to(h.device)
        hf = h.float()
        if self.last_only:
            hf[:, -1] -= (hf[:, -1] @ v)[:, None] * v
        else:
            hf = hf - (hf @ v)[..., None] * v
        h = hf.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        for h in self.handles:
            h.remove()


def placebo_direction(states, index, layer, rng):
    nos = rng.choice(len(PLACEBO_PAIRS), len(PAIRS), replace=False)
    sign = rng.choice([-1, 1], len(PAIRS))
    v = np.mean([s * (mean_state(states, index, PLACEBO_PAIRS[n]["a"], "moral", FIT_TEMPLATES)[layer]
                      - mean_state(states, index, PLACEBO_PAIRS[n]["b"], "moral", FIT_TEMPLATES)[layer])
                 for n, s in zip(nos, sign)], axis=0)
    return v / np.linalg.norm(v)


def random_direction(dim, rng):
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def score(model, tokenizer, ids, animal, kinds):
    """Readout averaged over paraphrases: {"E", "mass"} and/or {"yes_no"}."""
    out = {}
    for kind in kinds:
        vals = [readout_scores(forward_last(model, tokenizer, [chat(tokenizer, t.format(animal=animal))])[0],
                               kind, *ids) for t in READOUT[kind]]
        for key in vals[0]:
            out[key] = float(np.mean([v[key] for v in vals]))
    return out


def run(args):
    out_path = LOGS / f"{args.model}_ablate_{args.variant}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"folds": {}}

    dirs = torch.load(LOGS / f"{args.model}_directions.pt")
    states, index = load_states(args.model)
    model, tokenizer = load(args.model)
    ids = answer_ids(tokenizer)
    layers = decoder_layers(model)
    dim = states.shape[-1]

    for k, pair in enumerate(PAIRS):
        if str(pair["id"]) in done["folds"]:
            continue
        L = dirs["layers"][k]
        rng = np.random.default_rng([SEED, pair["id"]])  # same controls on resume
        controls = ([("random", random_direction(dim, rng)) for _ in range(args.controls)]
                    + [("placebo", placebo_direction(states, index, L, rng)) for _ in range(args.controls)])

        hook_ids = range(len(layers)) if args.variant == "all_layers" else [L]
        abl = Ablator(layers, hook_ids, last_only=args.variant == "last_pos")
        fold = {"layer": L, "main": {}, "controls": {"random": [], "placebo": []}}

        abl.v = dirs["v"][k]
        for role in ("food", "nonfood"):
            fold["main"][role] = score(model, tokenizer, ids, pair[role], ["scale", "yes_no"])
        for n, (kind, v) in enumerate(controls):
            abl.v = torch.tensor(v, dtype=torch.float32)
            e = {role: score(model, tokenizer, ids, pair[role], ["scale"])["E"] for role in ("food", "nonfood")}
            fold["controls"][kind].append(e)
            print(f"pair {pair['id']} control {n + 1}/{len(controls)}", flush=True)
        abl.remove()

        done["folds"][str(pair["id"])] = fold
        out_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(f"pair {pair['id']} done: {fold['main']}", flush=True)


def summary(args):
    """§7.2–7.3: closure C vs the 95th percentile of each control."""
    res = json.loads((LOGS / f"{args.model}_ablate_{args.variant}.json").read_text(encoding="utf-8"))
    base = json.loads((LOGS / f"{args.model}_h1.json").read_text(encoding="utf-8"))["behavior"]["per_animal"]
    folds = [(p, res["folds"][str(p["id"])]) for p in PAIRS if str(p["id"]) in res["folds"]]

    gap = lambda e_food, e_non: e_non - e_food
    g_base = np.array([gap(base[p["food"]]["E"], base[p["nonfood"]]["E"]) for p, _ in folds])
    g_abl = np.array([gap(f["main"]["food"]["E"], f["main"]["nonfood"]["E"]) for _, f in folds])
    C = float((g_base - g_abl).mean())
    yn_base = np.array([gap(base[p["food"]]["yes_no"], base[p["nonfood"]]["yes_no"]) for p, _ in folds])
    yn_abl = np.array([gap(f["main"]["food"]["yes_no"], f["main"]["nonfood"]["yes_no"]) for _, f in folds])
    C_yn = float((yn_base - yn_abl).mean())

    out = {"variant": args.variant, "n_folds": len(folds), "C": C,
           "C_fraction": C / float(g_base.mean()), "C_yes_no": C_yn,
           "d_z_closure": float(d_z(g_base - g_abl)),
           "shift_food": float(np.mean([f["main"]["food"]["E"] - base[p["food"]]["E"] for p, f in folds])),
           "shift_nonfood": float(np.mean([f["main"]["nonfood"]["E"] - base[p["nonfood"]]["E"] for p, f in folds])),
           "min_mass_ablated": float(min(min(f["main"][r]["mass"] for r in ("food", "nonfood")) for _, f in folds))}
    for kind in ("random", "placebo"):
        n = min(len(f["controls"][kind]) for _, f in folds)
        Cj = [np.mean([g_base[i] - gap(f["controls"][kind][j]["food"], f["controls"][kind][j]["nonfood"])
                       for i, (_, f) in enumerate(folds)]) for j in range(n)]
        out[f"{kind}_p95"] = float(np.percentile(Cj, 95))
        out[f"{kind}_mean"] = float(np.mean(Cj))
    out["H2"] = bool(C > out["random_p95"] and C > out["placebo_p95"]
                     and np.sign(C) == np.sign(C_yn) and out["min_mass_ablated"] >= 0.5)
    (LOGS / f"{args.model}_h2_{args.variant}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--variant", default="all_pos", choices=["all_pos", "last_pos", "all_layers"])
    ap.add_argument("--controls", type=int, default=N_CONTROLS)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    summary(args) if args.summary else run(args)


if __name__ == "__main__":
    main()
