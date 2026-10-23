"""
extract.py — stage 2: residual-stream states and baseline readout (preregistration §5.1, §6).

For one model:
  - every animal × context × template → output of every decoder layer at the
    last prompt token (captured by hooks, the same point §7 ablates);
  - every animal × readout prompt → digit distribution (1..SCALE_MAX) and
    logit(Yes) − logit(No) at the first answer position.

Output: logs/<model>_states.pt (float16 [n_prompts, n_layers, d] + metadata),
        logs/<model>_readout.json.

Usage:
    python extract.py --model qwen-7b
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models import MODELS
from stimuli import TEMPLATES, READOUT, SCALE_MAX, DIGIT_READOUTS, all_animals

LOGS = Path(__file__).parent / "logs"


def chat(tokenizer, text):
    """User message only, generation prompt appended (§5.1)."""
    return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)


def decoder_layers(model):
    return model.model.layers


class LastTokenRecorder:
    """Forward hooks that keep each layer's output at the last position."""

    def __init__(self, layers):
        self.out = [None] * len(layers)
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(layers)]

    def _hook(self, i):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out[i] = h[:, -1].detach().float().cpu()
        return fn

    def stacked(self):
        return torch.stack(self.out, dim=1)  # [batch, n_layers, d]

    def remove(self):
        for h in self.handles:
            h.remove()


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


@torch.no_grad()
def forward_last(model, tokenizer, texts):
    """Left-padded batch → logits at the last position [batch, vocab]."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # padding-invariant positions
    return model(**enc, position_ids=pos).logits[:, -1].float().cpu()


def answer_ids(tokenizer):
    one = lambda s: tokenizer.encode(s, add_special_tokens=False)
    digits = [one(str(d)) for d in range(1, SCALE_MAX + 1)]
    yes, no = one("Yes"), one("No")
    assert all(len(x) == 1 for x in digits + [yes, no]), "answer tokens must be single tokens"
    return [d[0] for d in digits], yes[0], no[0]


def context_prompts():
    rows = []
    for animal in all_animals():
        for ctx in ("moral", "neutral"):
            for t, tpl in enumerate(TEMPLATES[ctx]):
                rows.append({"animal": animal, "context": ctx, "template": t,
                             "text": tpl.format(animal=animal)})
    return rows


def readout_prompts():
    rows = []
    for animal in all_animals():
        for kind, tpls in READOUT.items():
            for t, tpl in enumerate(tpls):
                rows.append({"animal": animal, "kind": kind, "template": t,
                             "text": tpl.format(animal=animal)})
    return rows


def readout_scores(logits, kind, digit_ids, yes_id, no_id):
    """Per row: E over digits and digit mass, or Yes−No logit difference."""
    if kind in DIGIT_READOUTS:
        p = torch.softmax(logits, -1)[digit_ids]
        mass = p.sum().item()
        e = (p * torch.arange(1, SCALE_MAX + 1)).sum().item() / mass
        return {"E": e, "mass": mass}
    return {"yes_no": (logits[yes_id] - logits[no_id]).item()}


def run_readout(model, tokenizer, rows, batch_size):
    digit_ids, yes_id, no_id = answer_ids(tokenizer)
    for chunk in batches(rows, batch_size):
        logits = forward_last(model, tokenizer, [chat(tokenizer, r["text"]) for r in chunk])
        for r, lg in zip(chunk, logits):
            r.update(readout_scores(lg, r["kind"], digit_ids, yes_id, no_id))
    return rows


def load(key):
    cfg = MODELS[key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = None
    if cfg.get("quant") == "nf4":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["name"], dtype=torch.bfloat16,
                                                 device_map=cfg.get("device_map", "auto"), quantization_config=quant)
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 = bit-reproducible; batching changes bf16 kernels (~2% drift) and is not faster with CPU offload")
    ap.add_argument("--readout-only", action="store_true", help="skip states, redo the readout file")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N prompts of each kind")
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    torch.manual_seed(0)

    model, tokenizer = load(args.model)

    tag = "" if args.limit is None else "_smoke"
    if not args.readout_only:
        rows = context_prompts()[:args.limit]
        rec = LastTokenRecorder(decoder_layers(model))
        states = []
        for i, chunk in enumerate(batches(rows, args.batch_size)):
            forward_last(model, tokenizer, [chat(tokenizer, r["text"]) for r in chunk])
            states.append(rec.stacked().half())
            print(f"states {min((i + 1) * args.batch_size, len(rows))}/{len(rows)}", flush=True)
        rec.remove()
        torch.save({"model": MODELS[args.model]["name"], "rows": rows,
                    "states": torch.cat(states)}, LOGS / f"{args.model}_states{tag}.pt")

    readout = run_readout(model, tokenizer, readout_prompts()[:args.limit], args.batch_size)
    (LOGS / f"{args.model}_readout{tag}.json").write_text(
        json.dumps({"model": MODELS[args.model]["name"], "rows": readout}, indent=1), encoding="utf-8")
    print(f"-> {LOGS}")


if __name__ == "__main__":
    main()
