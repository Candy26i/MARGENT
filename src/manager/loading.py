"""Manager checkpoint loading shared by collection, SFT and evaluation.

One rule for what a manager "source" is: a local PEFT adapter directory (has
adapter_config.json) is loaded on top of ``base_model``; anything else (a full
checkpoint directory or a Hugging Face model id) is loaded as-is.
"""
from __future__ import annotations

import os
from typing import Any, Tuple


def is_adapter_dir(path: str) -> bool:
    """True when ``path`` is a local PEFT adapter directory."""
    return bool(path) and os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def load_manager(base_model: str, source: str, device: str, dtype: Any) -> Tuple[Any, Any]:
    """Load the frozen manager for inference.

    Returns (tokenizer, model): the tokenizer with left padding and a pad token,
    the model in eval mode on ``device``. ``source`` is an adapter directory, a
    full checkpoint directory or a Hugging Face id (see module docstring).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if source.startswith(("./", "../", "/", "~", "outputs/")) and not os.path.isdir(os.path.expanduser(source)):
        raise FileNotFoundError(f"manager checkpoint directory not found: {source}")

    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    if is_adapter_dir(source):
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, source).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    model.eval()
    return tok, model
