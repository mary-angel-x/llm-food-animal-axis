"""
models.py — model registry (preregistration §4).

For each model: run key → Hugging Face name, short name for plots, role.
"""
MODELS = {
    "qwen-7b": {"name": "Qwen/Qwen2.5-7B-Instruct",
                "short": "Qwen-7B", "role": "main"},
    # gated: needs huggingface-cli login and accepting the license on the model page
    "llama-8b": {"name": "meta-llama/Llama-3.1-8B-Instruct",
                 "short": "Llama-8B", "role": "transfer"},
    "gemma-2b": {"name": "google/gemma-2-2b-it",
                 "short": "Gemma-2B", "role": "transfer"},
    # v0.5 §13: scale extension. 4-bit NF4 so that all 48 layers fit the 8 GB GPU;
    # embeddings and lm_head (not quantized by bitsandbytes) stay on the CPU.
    "qwen-14b": {"name": "Qwen/Qwen2.5-14B-Instruct",
                 "short": "Qwen-14B", "role": "scale", "quant": "nf4",
                 "device_map": {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0,
                                "model.rotary_emb": 0, "lm_head": "cpu"}},
}
