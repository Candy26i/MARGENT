"""Shared paths and configuration across pipeline stages.

Output paths are auto-namespaced by teacher_id so different teachers'
artifacts never collide.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class StageContext:
    """Shared paths and configuration across stages."""
    base_model: str
    teacher_id: str                       # e.g. "mmlu_pro_gpt54", used for manager/eval paths
    teacher_provider: str = ""            # filled when a teacher is built
    teacher_model: str = ""
    output_root: str = "outputs"
    seed: int = 42
    binding_mode: str = "auto"
    subagent_teacher_id: str = ""        # if set, subagent adapters come from this id instead of teacher_id

    # Auto-derived sub-roots
    sft_data_root: str = field(init=False)
    adapter_root: str = field(init=False)
    manager_root: str = field(init=False)
    cache_dir: str = field(init=False)
    eval_root: str = field(init=False)

    def __post_init__(self) -> None:
        teacher_slug = self._slug(self.teacher_id)
        adapter_slug = self._slug(self.subagent_teacher_id or self.teacher_id)
        self.sft_data_root = os.path.join(self.output_root, "sft_data", teacher_slug)
        self.adapter_root = os.path.join(self.output_root, "adapters", adapter_slug)
        self.manager_root = os.path.join(self.output_root, "manager", teacher_slug)
        self.cache_dir = os.path.join(self.output_root, "teacher_cache", teacher_slug)
        self.eval_root = os.path.join(self.output_root, "eval", teacher_slug)
        for p in (self.sft_data_root, self.adapter_root, self.manager_root,
                  self.cache_dir, self.eval_root):
            os.makedirs(p, exist_ok=True)

    @staticmethod
    def _slug(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
        return s.strip("_") or "unnamed"

    def adapter_path(self, kind: str) -> str:
        return os.path.join(self.adapter_root, f"{kind}_adapter")

    def sft_jsonl_path(self, kind: str) -> str:
        return os.path.join(self.sft_data_root, f"{kind}_sft.jsonl")

    def sft_log_path(self, kind: str) -> str:
        return os.path.join(self.sft_data_root, f"{kind}_synth_log.jsonl")

    def marginal_value_dir(self) -> str:
        return os.path.join(self.manager_root, "marginal_value")

    def manager_sft_dir(self) -> str:
        return os.path.join(self.manager_root, "sft_marginal")
