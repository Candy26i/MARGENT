"""Manager SFT on per-turn (prompt, response) trajectories.

The training JSONL is produced by ``build_marginal_sft``
(``manager_sft_marginal.jsonl``). Each row holds a chat ``prompt`` (a list of
messages, possibly ending in tool outputs) and the assistant ``response`` turn
to supervise: either a tool call or the DRAFT_ANSWER_/ANSWER_ commit. Only the
response tokens receive labels; the rendered prompt is masked.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ..utils.io import read_jsonl
from ..utils.seed import set_seed

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from .chat_template import render_chat


@dataclass
class ManagerSFTConfig:
    base_model: str
    train_jsonl: str
    out_dir: str
    init_model_or_adapter: Optional[str] = None
    seed: int = 42
    max_seq_len: int = 4096
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    bf16: bool = True


def _mask_prefix_len(prompt_ids: List[int], full_ids: List[int]) -> int:
    """Common token prefix of the prompt-only and full renders. See
    subagents/train.py: len(prompt_ids) is wrong for templates (e.g. Qwen3 with
    enable_thinking=False) whose generation prompt is not a strict prefix of
    the full render — it would mask the first response tokens."""
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    return i


def _tokenize_manager_sft(rows: List[Dict[str, Any]], tok, max_seq_len: int, tools=None) -> Dataset:
    eos = tok.eos_token or ""

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msgs = ex["prompt"]
        response_msgs = ex["response"]
        if isinstance(response_msgs, dict):
            response_msgs = [response_msgs]
        elif isinstance(response_msgs, str):
            response_msgs = [{"role": "assistant", "content": response_msgs}]

        prompt_text = render_chat(tok, prompt_msgs, add_generation_prompt=True, tools=tools)
        full_text = render_chat(tok, prompt_msgs + response_msgs, add_generation_prompt=False, tools=tools)
        if eos and not full_text.rstrip().endswith(eos):
            full_text = full_text + eos

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = tok(full_text, add_special_tokens=False)
        input_ids = full["input_ids"][:max_seq_len]
        attention_mask = full["attention_mask"][:max_seq_len]
        plen = min(_mask_prefix_len(prompt_ids, full["input_ids"]), max_seq_len)
        labels = ([-100] * plen) + input_ids[plen:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = Dataset.from_list(rows)
    return ds.map(_map, remove_columns=ds.column_names)


def train_manager_sft(cfg: ManagerSFTConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.use_lora and not PEFT_AVAILABLE:
        raise RuntimeError("peft is required when manager SFT is configured with LoRA.")

    init_source = cfg.init_model_or_adapter or cfg.base_model
    tok = AutoTokenizer.from_pretrained(init_source, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
    is_adapter_init = bool(
        cfg.init_model_or_adapter
        and os.path.isdir(cfg.init_model_or_adapter)
        and os.path.exists(os.path.join(cfg.init_model_or_adapter, "adapter_config.json"))
    )
    is_full_init = bool(
        cfg.init_model_or_adapter
        and os.path.isdir(cfg.init_model_or_adapter)
        and os.path.exists(os.path.join(cfg.init_model_or_adapter, "config.json"))
        and not is_adapter_init
    )
    if is_adapter_init:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is required to continue SFT from a manager adapter.")
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(
            base, cfg.init_model_or_adapter, is_trainable=cfg.use_lora
        ).to(device)
        if not cfg.use_lora:
            model = model.merge_and_unload().to(device)
        print(f"[MANAGER_SFT] continuing from adapter -> {cfg.init_model_or_adapter}")
    elif is_full_init:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.init_model_or_adapter, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        print(f"[MANAGER_SFT] continuing from full checkpoint -> {cfg.init_model_or_adapter}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    model.config.use_cache = False
    if not cfg.use_lora:
        for param in model.parameters():
            param.requires_grad_(True)

    if cfg.use_lora and PEFT_AVAILABLE and not is_adapter_init:
        candidate = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = {n.split(".")[-1] for n, _ in model.named_modules()}
        target = [m for m in candidate if m in present] or ["q_proj", "v_proj"]
        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target,
        )
        model = get_peft_model(model, lconf)
        print(f"[MANAGER_SFT/LoRA] r={cfg.lora_r} alpha={cfg.lora_alpha} target_modules={target}")

    rows = read_jsonl(cfg.train_jsonl)
    if not rows:
        raise ValueError(f"No rows in {cfg.train_jsonl}")
    print(f"[MANAGER_SFT] tokenizing {len(rows)} rows ...")
    from .marginal_value import _tool_schemas
    manager_tools = _tool_schemas("environment")
    print(f"[MANAGER_SFT] rendering with {len(manager_tools)} tool schemas")
    train_ds = _tokenize_manager_sft(rows, tok, cfg.max_seq_len, tools=manager_tools)
    total_steps = (len(train_ds) // (cfg.per_device_batch_size * cfg.gradient_accumulation_steps)) * cfg.num_train_epochs
    if cfg.max_steps > 0:
        total_steps = min(total_steps, cfg.max_steps)
    print(f"[MANAGER_SFT] {len(train_ds)} train examples | ~{total_steps} steps | lr={cfg.learning_rate} | epochs={cfg.num_train_epochs}")
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=1,
        save_strategy="epoch",
        bf16=(cfg.bf16 and device == "cuda"),
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[MANAGER_SFT] saved -> {cfg.out_dir}")
