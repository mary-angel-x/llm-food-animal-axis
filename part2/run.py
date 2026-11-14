"""
run.py — Part 2 forward passes for one model: digit readout and L* states of every entity.

    python -m part2.run --model qwen-7b        (from the moral_AI folder)

Output: part2/logs/<model>_readout.json  rows {entity, kind, template, E, mass}
        part2/logs/<model>_states.pt     {"rows": [{entity, context, template}], "states": [n, d] at L*, "layer": L*}
Reuses the Part 1 loading, chat format, padding-invariant forward and digit readout unchanged.
"""
import argparse
import json
from pathlib import Path

import torch

import extract
from extract import LastTokenRecorder, answer_ids, chat, decoder_layers, forward_last, load, readout_scores
from models import MODELS
from stimuli import EVAL_TEMPLATES, TEMPLATES

from .entities import READOUT, all_entities

ROOT = Path(__file__).resolve().parent
LOGS, PART1_LOGS = ROOT / "logs", ROOT.parent / "logs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--answer-prefix", action="store_true",
                        help='Part 1 §14: assistant turn starts with "Rating: "; readout only -> <model>_readout_prefix.json')
    args = parser.parse_args()
    suffix = "_prefix" if args.answer_prefix else ""
    if args.answer_prefix:
        extract.ANSWER_PREFIX = extract.PREFIX
    LOGS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    layer = torch.load(PART1_LOGS / f"{args.model}_directions_v03.pt")["layers"][0]
    model, tokenizer = load(args.model)
    digit_ids, yes_id, no_id = answer_ids(tokenizer)

    rows = [{"entity": e, "kind": kind, "template": t, "text": template.format(x=e)}
            for e in all_entities() for kind, templates in READOUT.items() for t, template in enumerate(templates)]
    for i, row in enumerate(rows):                         # batch size 1, as in Part 1
        logits = forward_last(model, tokenizer, [chat(tokenizer, row["text"])])[0]
        row.update(readout_scores(logits, "scale", digit_ids, yes_id, no_id))
        if (i + 1) % 60 == 0:
            print(f"readout {i + 1}/{len(rows)}", flush=True)
    (LOGS / f"{args.model}_readout{suffix}.json").write_text(
        json.dumps({"model": MODELS[args.model]["name"], "rows": rows}, indent=1), encoding="utf-8")
    if args.answer_prefix:                                  # the states do not depend on the answer format
        return

    contexts = [{"entity": e, "context": ctx, "template": t, "text": TEMPLATES[ctx][t].format(animal=e)}
                for e in all_entities() for ctx in ("moral", "neutral") for t in EVAL_TEMPLATES]
    recorder = LastTokenRecorder(decoder_layers(model))
    states = []
    for i, row in enumerate(contexts):
        forward_last(model, tokenizer, [chat(tokenizer, row["text"])])
        states.append(recorder.out[layer][0].half())
        if (i + 1) % 60 == 0:
            print(f"states {i + 1}/{len(contexts)}", flush=True)
    recorder.remove()
    torch.save({"model": MODELS[args.model]["name"], "rows": contexts, "states": torch.stack(states), "layer": layer},
               LOGS / f"{args.model}_states.pt")
    print(f"-> {LOGS}", flush=True)


if __name__ == "__main__":
    main()
