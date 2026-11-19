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

from extract import SCALE_MAX, chat, forward_last, load, readout_scores
from models import MODELS

from .capacity import REASON
from .entities import READOUT, all_entities
from .run import LOGS


def answer_ids(tokenizer):
    """extract.answer_ids, except that a leading SentencePiece space piece ('▁' alone) is dropped (Phi-3.5).
    Amendment 2026-11-19; for tokenizers without that piece the result equals extract.answer_ids."""
    def one(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 2 and tokenizer.convert_ids_to_tokens(ids[0]) == "▁":
            ids = ids[1:]
        return ids
    digits = [one(str(d)) for d in range(1, SCALE_MAX + 1)]
    yes, no = one("Yes"), one("No")
    assert all(len(x) == 1 for x in digits + [yes, no]), "answer tokens must be single tokens"
    return [d[0] for d in digits], yes[0], no[0]


def read(model, tokenizer, rows, label, space_first=False):
    """space_first (Mistral, deviation 2026-11-19): append the lone SentencePiece space the model emits first,
    read the digit at the next position, and log that space's probability at the first position (check >= 0.9)."""
    digit_ids, yes_id, no_id = answer_ids(tokenizer)
    space_id = tokenizer.convert_tokens_to_ids("▁") if space_first else None
    for i, row in enumerate(rows):                         # batch size 1, as in Part 1 and Part 2
        if space_first:
            ids = tokenizer(chat(tokenizer, row["text"]), add_special_tokens=False)["input_ids"] + [space_id]
            with torch.no_grad():
                out = model(input_ids=torch.tensor([ids], device=model.device)).logits[0].float().cpu()
            logits = out[-1]
            row["space_p"] = float(out[-2].softmax(-1)[space_id])
        else:
            logits = forward_last(model, tokenizer, [chat(tokenizer, row["text"])])[0]
        row.update(readout_scores(logits, "scale", digit_ids, yes_id, no_id))
        if (i + 1) % 60 == 0:
            print(f"{label} {i + 1}/{len(rows)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--space-first", action="store_true", help="Mistral: see read()")
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model, tokenizer = load(args.model)
    name = MODELS[args.model]["name"]

    base = [{"entity": e, "kind": kind, "template": t, "text": template.format(x=e)}
            for e in all_entities() for kind, templates in READOUT.items() for t, template in enumerate(templates)]
    read(model, tokenizer, base, "readout", args.space_first)
    (LOGS / f"{args.model}_readout.json").write_text(json.dumps({"model": name, "space_first": args.space_first, "rows": base}, indent=1),
                                                     encoding="utf-8")

    cap = [{"entity": e, "kind": "reason", "template": t, "text": tpl.format(x=e)}
           for e in all_entities() for t, tpl in enumerate(REASON)]
    read(model, tokenizer, cap, "reason", args.space_first)
    (LOGS / f"{args.model}_capacity.json").write_text(json.dumps({"model": name, "prefix": False, "space_first": args.space_first, "rows": cap},
                                                                 indent=1), encoding="utf-8")
    print(f"-> {LOGS}", flush=True)


if __name__ == "__main__":
    main()
