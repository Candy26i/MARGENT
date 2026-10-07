"""Manager stages: interventional collection and marginal-value distillation.

  - run_build_marginal_sft: collect paired counterfactual branches and write
    manager_sft_marginal.jsonl (the method's training data).
  - run_train_manager_sft: per-turn SFT of the manager on that JSONL.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..benchmarks.base import StandardRow
from .context import StageContext


# ---------------- Stage: counterfactual marginal-value SFT ----------------

def run_build_marginal_sft(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    n_samples: int = 300,
    max_depth: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_commit_rescue_ratio: float = 1.0,
    task_description: str = "",
    output_dir: Optional[str] = None,
    subagent_server_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect paired counterfactual branches and build routing SFT data.

    The direct answer is treated as the manager's initial draft.  Advisor
    sequences are forced breadth-first and ground truth selects the shortest
    sequence that actually corrects that draft.  No synthetic GT draft or
    per-call reward is used.
    """
    from ..manager.marginal_value import MarginalValueConfig, build_marginal_value_sft

    binding = "argument" if ctx.binding_mode == "argument" else "environment"
    cfg = MarginalValueConfig(
        base_model=ctx.base_model,
        manager_dir=manager_dir or ctx.base_model,
        rows=rows,
        out_dir=output_dir or ctx.marginal_value_dir(),
        extractor_adapter=ctx.adapter_path("extractor"),
        reasoner_adapter=ctx.adapter_path("reasoner"),
        verifier_adapter=ctx.adapter_path("verifier"),
        seed=ctx.seed,
        n_samples=n_samples,
        max_depth=max_depth,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        binding_mode=binding,
        task_description=task_description,
        max_commit_rescue_ratio=max_commit_rescue_ratio,
        subagent_server_url=subagent_server_url,
    )
    return build_marginal_value_sft(cfg)


# --------------------- Stage: manager SFT ---------------------

def run_train_manager_sft(
    ctx: StageContext,
    train_jsonl: Optional[str] = None,
    init_model_or_adapter: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 1,
    lr: float = 2e-5,
    max_seq_len: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_steps: int = -1,
) -> Dict[str, Any]:
    from ..manager.sft import ManagerSFTConfig, train_manager_sft

    if train_jsonl is None:
        train_jsonl = os.path.join(ctx.marginal_value_dir(), "manager_sft_marginal.jsonl")
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(f"manager SFT input not found: {train_jsonl}")

    out_dir = output_dir or ctx.manager_sft_dir()
    cfg = ManagerSFTConfig(
        base_model=ctx.base_model,
        train_jsonl=train_jsonl,
        out_dir=out_dir,
        init_model_or_adapter=init_model_or_adapter,
        seed=ctx.seed,
        max_seq_len=max_seq_len,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_steps=max_steps,
    )
    train_manager_sft(cfg)
    return {
        "manager_sft_dir": out_dir,
        "init_model_or_adapter": init_model_or_adapter or ctx.base_model,
    }
