"""
capacity.py — Part 2 amendment 2026-11-18: the `reason` readout (reasoning capacity), 60 entities x 3 templates.

    python -m part2.capacity --model qwen-7b                     (from the moral_AI folder)
    python -m part2.capacity --model llama-8b --answer-prefix    (exploratory, §14 format; not counted)

Output: part2/logs/<model>_capacity[_prefix].json  rows {entity, kind, template, E, mass}
Reuses the Part 2 readout procedure (batch size 1, digit readout, same chat format) unchanged.
"""
import argparse
import json

import torch

import extract
from extract import answer_ids, chat, forward_last, load, readout_scores
from models import MODELS

from .entities import all_entities
from .run import LOGS

REASON = [
    "On a scale from 1 to 9, how capable is the {x} of complex reasoning? Answer with a single digit.",
    "From 1 (not at all) to 9 (fully), how intelligent is the {x}? Answer with a single digit.",
    "Rate from 1 to 9 how well the {x} can plan ahead and solve problems. Answer with a single digit.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--answer-prefix", action="store_true",
                        help='§14 format: assistant turn starts with "Rating: "; exploratory for Llama')
    args = parser.parse_args()
    suffix = "_prefix" if args.answer_prefix else ""
    if args.answer_prefix:
        extract.ANSWER_PREFIX = extract.PREFIX
    LOGS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model, tokenizer = load(args.model)
    digit_ids, yes_id, no_id = answer_ids(tokenizer)

    rows = [{"entity": e, "kind": "reason", "template": t, "text": tpl.format(x=e)}
            for e in all_entities() for t, tpl in enumerate(REASON)]
    for i, row in enumerate(rows):
        logits = forward_last(model, tokenizer, [chat(tokenizer, row["text"])])[0]
        row.update(readout_scores(logits, "scale", digit_ids, yes_id, no_id))
        if (i + 1) % 30 == 0:
            print(f"reason {i + 1}/{len(rows)}", flush=True)
    out = LOGS / f"{args.model}_capacity{suffix}.json"
    out.write_text(json.dumps({"model": MODELS[args.model]["name"], "prefix": args.answer_prefix, "rows": rows},
                              indent=1), encoding="utf-8")
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
