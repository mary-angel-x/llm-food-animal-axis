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
}
