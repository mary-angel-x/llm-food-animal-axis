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
    # Part 2 §15 only (amendment 2026-11-19): behavioural readouts, no Part 1 directions
    "phi-3.5": {"name": "microsoft/Phi-3.5-mini-instruct",
                "short": "Phi-3.5", "role": "capacity"},
    "llama-3b": {"name": "meta-llama/Llama-3.2-3B-Instruct",
                 "short": "Llama-3B", "role": "capacity"},
}
