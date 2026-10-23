"""
ablate.py — stage 4: directional ablation and controls (preregistration §7, §12.4).

Fold k (held-out pair k): remove v_{-k} at the output of decoder layer L*
(v0.3 fixed layer), then read out food_k and nonfood_k. Controls per fold, at
the same layer: 100 random unit directions and 100 placebo directions (v from
13 random placebo pairs, random orientation). Control j across all folds is
one replicate of the closure statistic.

Controls are run only for readouts whose precondition passed in
logs/<model>_v03.json: "scale" if B0 (H2a), "accept" if B0b (H2b).

Variants: all_pos (primary), last_pos (secondary), all_layers (exploratory).
Results are written after every fold, so a stopped run resumes where it left.

Output: logs/<model>_ablate_<variant>.json, logs/<model>_h2_<variant>.json

Usage:
    python ablate.py --model llama-8b
    python ablate.py --model llama-8b --summary
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
# gap that H2 tries to close, per readout: (key, sign) with gap = sign·(nonfood − food)
GAPS = {"scale": ("E", 1), "accept": ("accept", -1)}
PRECONDITION = {"scale": "B0", "accept": "B0b"}


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
    """Readout averaged over paraphrases, keyed like analyze.behavior: E, mass, yes_no, accept, accept_mass."""
    rename = {("scale", "E"): "E", ("scale", "mass"): "mass", ("yes_no", "yes_no"): "yes_no",
              ("accept", "E"): "accept", ("accept", "mass"): "accept_mass"}
    out = {}
    for kind in kinds:
        vals = [readout_scores(forward_last(model, tokenizer, [chat(tokenizer, t.format(animal=animal))])[0],
                               kind, *ids) for t in READOUT[kind]]
        for field in vals[0]:
            out[rename[(kind, field)]] = float(np.mean([v[field] for v in vals]))
    return out


def control_kinds(key):
    verdict = json.loads((LOGS / f"{key}_v03.json").read_text(encoding="utf-8"))["verdict"]
    return [kind for kind, pre in PRECONDITION.items() if verdict.get(pre)]


def run(args):
    kinds = control_kinds(args.model)
    if not kinds:
        print("no precondition (B0 / B0b) passed: H2 is not tested for this model (§6, §12.4)")
        return
    out_path = LOGS / f"{args.model}_ablate_{args.variant}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"kinds": kinds, "folds": {}}

    dirs = torch.load(LOGS / f"{args.model}_directions_v03.pt")
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
            fold["main"][role] = score(model, tokenizer, ids, pair[role], ["scale", "yes_no", "accept"])
        for n, (ctrl, v) in enumerate(controls):
            abl.v = torch.tensor(v, dtype=torch.float32)
            fold["controls"][ctrl].append({role: score(model, tokenizer, ids, pair[role], kinds)
                                           for role in ("food", "nonfood")})
            print(f"pair {pair['id']} control {n + 1}/{len(controls)}", flush=True)
        abl.remove()

        done["folds"][str(pair["id"])] = fold
        out_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(f"pair {pair['id']} done: {fold['main']}", flush=True)


def summary(args):
    """§7.2–7.3, §12.4: closure vs the 95th percentile of each control, per readout."""
    res = json.loads((LOGS / f"{args.model}_ablate_{args.variant}.json").read_text(encoding="utf-8"))
    base = json.loads((LOGS / f"{args.model}_v03.json").read_text(encoding="utf-8"))["behavior"]["per_animal"]
    folds = [(p, res["folds"][str(p["id"])]) for p in PAIRS if str(p["id"]) in res["folds"]]
    gap = lambda d_food, d_non, key, sign: sign * (d_non[key] - d_food[key])

    out = {"variant": args.variant, "n_folds": len(folds)}
    for kind in res["kinds"]:
        key, sign = GAPS[kind]
        g_base = np.array([gap(base[p["food"]], base[p["nonfood"]], key, sign) for p, _ in folds])
        g_abl = np.array([gap(f["main"]["food"], f["main"]["nonfood"], key, sign) for _, f in folds])
        C = float((g_base - g_abl).mean())
        r = {"C": C, "C_fraction": C / float(g_base.mean()), "d_z_closure": float(d_z(g_base - g_abl)),
             "shift_food": float(np.mean([f["main"]["food"][key] - base[p["food"]][key] for p, f in folds])),
             "shift_nonfood": float(np.mean([f["main"]["nonfood"][key] - base[p["nonfood"]][key] for p, f in folds])),
             "min_mass_ablated": float(min(f["main"][role]["mass" if kind == "scale" else "accept_mass"]
                                           for _, f in folds for role in ("food", "nonfood")))}
        for ctrl in ("random", "placebo"):
            n = min(len(f["controls"][ctrl]) for _, f in folds)
            Cj = [np.mean([g_base[i] - gap(f["controls"][ctrl][j]["food"], f["controls"][ctrl][j]["nonfood"], key, sign)
                           for i, (_, f) in enumerate(folds)]) for j in range(n)]
            r[f"{ctrl}_p95"] = float(np.percentile(Cj, 95))
            r[f"{ctrl}_mean"] = float(np.mean(Cj))
        passed = C > r["random_p95"] and C > r["placebo_p95"] and r["min_mass_ablated"] >= 0.5
        if kind == "scale":  # §6: Yes/No must agree in sign
            yn = lambda f, b: gap(f, b, "yes_no", 1)
            C_yn = float(np.mean([yn(base[p["food"]], base[p["nonfood"]]) - yn(f["main"]["food"], f["main"]["nonfood"])
                                  for p, f in folds]))
            r["C_yes_no"] = C_yn
            passed = passed and np.sign(C) == np.sign(C_yn)
        r["pass"] = bool(passed)
        out["H2a" if kind == "scale" else "H2b"] = r
    (LOGS / f"{args.model}_h2_{args.variant}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


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
