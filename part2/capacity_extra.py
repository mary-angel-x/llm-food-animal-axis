"""
capacity_extra.py — Part 2 amendment 2026-11-19: behavioural readouts for the extra §15 models.

    python -m part2.capacity_extra --model phi-3.5      (from the moral_AI folder)

Writes part2/logs/<model>_readout.json (suffer, accept: part2.run readout, unchanged)
and part2/logs/<model>_capacity.json (reason: part2.capacity, unchanged).
No hidden states and no Part 1 directions: the §15 test reads behaviour only.
"""
import argparse
import json

import torch

from extract import answer_ids, chat, forward_last, load, readout_scores
from models import MODELS

from .capacity import REASON
from .entities import READOUT, all_entities
from .run import LOGS


def read(model, tokenizer, rows, label):
    digit_ids, yes_id, no_id = answer_ids(tokenizer)
    for i, row in enumerate(rows):                         # batch size 1, as in Part 1 and Part 2
        logits = forward_last(model, tokenizer, [chat(tokenizer, row["text"])])[0]
        row.update(readout_scores(logits, "scale", digit_ids, yes_id, no_id))
        if (i + 1) % 60 == 0:
            print(f"{label} {i + 1}/{len(rows)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model, tokenizer = load(args.model)
    name = MODELS[args.model]["name"]

    base = [{"entity": e, "kind": kind, "template": t, "text": template.format(x=e)}
            for e in all_entities() for kind, templates in READOUT.items() for t, template in enumerate(templates)]
    read(model, tokenizer, base, "readout")
    (LOGS / f"{args.model}_readout.json").write_text(json.dumps({"model": name, "rows": base}, indent=1),
                                                     encoding="utf-8")

    cap = [{"entity": e, "kind": "reason", "template": t, "text": tpl.format(x=e)}
           for e in all_entities() for t, tpl in enumerate(REASON)]
    read(model, tokenizer, cap, "reason")
    (LOGS / f"{args.model}_capacity.json").write_text(json.dumps({"model": name, "prefix": False, "rows": cap},
                                                                 indent=1), encoding="utf-8")
    print(f"-> {LOGS}", flush=True)


if __name__ == "__main__":
    main()
