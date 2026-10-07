"""Sub-agent stages: SFT data synthesis, the local DeepSeek JSONL bridge,
sub-agent SFT training and the schema-validity evaluation.

Each `run_*` is a thin orchestrator that takes a StageContext + a few
explicit args and returns a small result dict (paths produced, stats).
The CLI maps argparse flags to these calls.
"""
from __future__ import annotations

import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..benchmarks.base import StandardRow, question_hash
from ..subagents.prompts.extractor import build_extractor_synth_prompt
from ..subagents.prompts.reasoner import build_reasoner_synth_prompt
from ..subagents.prompts.verifier import build_verifier_synth_prompt
from ..subagents.prompts.runtime_prompts import build_runtime_messages
from ..teachers.base import TeacherClient, build_teacher_client
from ..utils.cache import TeacherCallCache
from ..utils.io import read_jsonl, write_json, write_jsonl
from ..utils.leakage import LeakageAuditor
from ..utils.seed import set_seed
from .context import StageContext

if TYPE_CHECKING:
    from ..subagents.schemas import AgentKind


# --------------------- Helpers ---------------------

def _agent_kind_value(agent_kind: Any) -> str:
    return str(getattr(agent_kind, "value", agent_kind)).strip()


def _build_local_teacher_prompt(
    agent_kind: Any,
    row: StandardRow,
    candidate_answer: str = "",
) -> List[Dict[str, str]]:
    kind = _agent_kind_value(agent_kind)
    if kind == "extractor":
        return build_extractor_synth_prompt(row.question, row.context, row.choices)
    if kind == "reasoner":
        return build_reasoner_synth_prompt(row.question, row.context, row.choices)
    if kind == "verifier":
        return build_verifier_synth_prompt(
            row.question, row.context, row.choices, candidate_answer=candidate_answer
        )
    raise ValueError(f"Unknown agent_kind: {agent_kind}")

def _build_teacher(provider: str, model: str, ctx: StageContext) -> TeacherClient:
    teacher = build_teacher_client(provider=provider, model=model)
    ctx.teacher_provider = teacher.provider
    ctx.teacher_model = teacher.model
    return teacher


# --------------------- Stage: subagent SFT data synthesis ---------------------

def run_synthesize_subagent(
    ctx: StageContext,
    rows: List[StandardRow],
    agent_kind: AgentKind,
    teacher_provider: str,
    teacher_model: str,
    n_samples: int = 500,
    base_temperature: float = 0.4,
    max_retries: int = 2,
    use_cache: bool = True,
    max_workers: int = 8,
    symmetric_leakage: bool = False,
) -> Dict[str, Any]:
    from ..subagents.schemas import AgentKind
    from ..subagents.synthesize import synthesize_subagent_data

    agent_kind = AgentKind(_agent_kind_value(agent_kind))
    teacher = _build_teacher(teacher_provider, teacher_model, ctx)
    cache = TeacherCallCache(ctx.cache_dir) if use_cache else None
    auditor = LeakageAuditor()

    out_path = ctx.sft_jsonl_path(agent_kind.value)
    log_path = ctx.sft_log_path(agent_kind.value)

    stats = synthesize_subagent_data(
        rows=rows,
        agent_kind=agent_kind,
        teacher=teacher,
        out_path=out_path,
        cache=cache,
        auditor=auditor,
        n_samples=n_samples,
        base_temperature=base_temperature,
        max_retries_per_sample=max_retries,
        seed=ctx.seed,
        log_path=log_path,
        max_workers=max_workers,
        symmetric_leakage=symmetric_leakage,
    )

    return {
        "agent_kind": agent_kind.value,
        "teacher_provider": teacher.provider,
        "teacher_model": teacher.model,
        "out_path": out_path,
        "log_path": log_path,
        "stats": stats.__dict__,
    }


# --------------------- Stage: local DeepSeek JSONL bridge ---------------------

def run_export_deepseek_subagent_prompts(
    ctx: StageContext,
    rows: List[StandardRow],
    agent_kind: AgentKind,
    out_path: Optional[str] = None,
    n_samples: int = 500,
) -> Dict[str, Any]:
    """Write JSONL prompts for a local batch generator.

    Each row is compatible with the patched DeepSeek `generate_jsonl.py`:
      {"example_id": int, "prompt": [{"role": ..., "content": ...}, ...]}

    Extra fields are intentionally included so `import_deepseek_subagent_responses`
    can reconstruct validated SFT rows even if the generator output only keeps
    example_id/prompt/response.
    """
    from ..subagents.schemas import AgentKind
    from ..subagents.synthesize import _sample_verifier_candidate

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples] if n_samples > 0 else sample
    kind_value = _agent_kind_value(agent_kind)

    if out_path is None:
        out_path = os.path.join(ctx.sft_data_root, f"{kind_value}_deepseek_prompts.jsonl")

    out_rows: List[Dict[str, Any]] = []
    for r in sample:
        # Mirror the online-synthesis behavior: ~50% of verifier samples audit
        # a random candidate. The candidate is stored so the importer can build
        # the matching runtime prompt.
        candidate = (
            _sample_verifier_candidate(AgentKind.VERIFIER, r, ctx.seed)
            if kind_value == "verifier" else ""
        )
        prompt = _build_local_teacher_prompt(
            agent_kind,
            r,
            candidate_answer=candidate,
        )
        out_rows.append({
            "example_id": int(r.example_id),
            "question_hash": question_hash(r.question),
            "benchmark_name": r.benchmark_name,
            "agent_kind": kind_value,
            "question": r.question,
            "context": r.context,
            "choices": dict(r.choices),
            "ground_truth": r.ground_truth,
            "candidate_answer": candidate,
            "prompt": prompt,
        })

    write_jsonl(out_path, out_rows)
    return {"agent_kind": kind_value, "out_path": out_path, "n_rows": len(out_rows)}


def run_import_deepseek_subagent_responses(
    ctx: StageContext,
    agent_kind: AgentKind,
    prompt_jsonl: str,
    response_jsonl: str,
    out_path: Optional[str] = None,
    log_path: Optional[str] = None,
    teacher_model: str = "deepseek-local",
    raw_responses: bool = False,
) -> Dict[str, Any]:
    """Convert local JSONL responses into subagent SFT rows.

    By default responses are parsed, schema-validated, and leakage-audited. With
    raw_responses=True, keep the teacher response text exactly as generated but
    pair it with the runtime subagent prompt. This is useful for experiments
    that intentionally train on unfiltered teacher outputs without teaching the
    model the teacher-data-generation prompt.
    """
    from ..subagents.schemas import AgentKind
    from ..subagents.synthesize import (
        _extract_first_json,
        _gt_audit_keywords,
        _reasoner_choice_coverage_check,
        _validate_schema,
    )

    agent_kind = AgentKind(_agent_kind_value(agent_kind))
    kind_value = agent_kind.value
    if out_path is None:
        out_path = ctx.sft_jsonl_path(kind_value)
    if log_path is None:
        log_path = os.path.join(ctx.sft_data_root, f"{kind_value}_deepseek_import_log.jsonl")

    prompt_rows = read_jsonl(prompt_jsonl)
    response_rows = read_jsonl(response_jsonl)
    prompt_by_id = {int(r["example_id"]): r for r in prompt_rows if r.get("example_id") is not None}

    auditor = LeakageAuditor()
    sft_rows: List[Dict[str, Any]] = []
    log_rows: List[Dict[str, Any]] = []

    for resp_row in response_rows:
        eid = resp_row.get("example_id")
        try:
            eid_int = int(eid)
        except Exception:
            log_rows.append({"example_id": eid, "ok": False, "error": "missing_or_invalid_example_id"})
            continue

        src = prompt_by_id.get(eid_int)
        if src is None:
            log_rows.append({"example_id": eid_int, "ok": False, "error": "example_id_not_in_prompt_jsonl"})
            continue

        row = StandardRow(
            example_id=eid_int,
            benchmark_name=str(src.get("benchmark_name") or "medqa"),
            task_subtype=str(src.get("task_subtype") or ""),
            question=str(src.get("question") or ""),
            choices=dict(src.get("choices") or {}),
            ground_truth=str(src.get("ground_truth") or ""),
            context=str(src.get("context") or ""),
            metadata=dict(src.get("metadata") or {}),
            split=str(src.get("split") or ""),
        )

        text = str(resp_row.get("response") or "")
        # Use the SAME candidate the teacher prompt was built with (stored at
        # export time), so the runtime prompt matches the teacher's context.
        candidate = str(src.get("candidate_answer") or "")
        runtime_prompt = build_runtime_messages(
            agent_kind=kind_value,
            question=row.question,
            context=row.context,
            choices=row.choices,
            candidate_answer=candidate,
        )
        if raw_responses:
            if not text.strip():
                log_rows.append({"example_id": eid_int, "ok": False, "error": "empty_response"})
                continue
            sft_rows.append({
                "example_id": eid_int,
                "question_hash": question_hash(row.question),
                "benchmark_name": row.benchmark_name,
                "agent_kind": kind_value,
                "teacher_provider": "raw_jsonl",
                "teacher_model": teacher_model,
                "prompt": runtime_prompt,
                "response": text.strip(),
            })
            log_rows.append({"example_id": eid_int, "ok": True, "raw_response": True})
            continue

        obj = _extract_first_json(text)
        if obj is None:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "json_parse_fail",
                "text_preview": text[:400],
            })
            continue

        try:
            model = _validate_schema(agent_kind, obj)
        except Exception as e:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "schema_fail",
                "detail": str(e)[:400],
            })
            continue

        ok_balance, balance_msg = _reasoner_choice_coverage_check(agent_kind, obj, row)
        if not ok_balance:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "balance_fail",
                "detail": balance_msg,
            })
            continue

        kw = _gt_audit_keywords(row)
        audit = auditor.audit(
            generated=obj,
            ground_truth_label=kw["ground_truth_label"],
            ground_truth_text=kw["ground_truth_text"],
            token_form=kw["token_form"],
            all_choice_texts=list(row.choices.values()),
        )
        if audit.leaked:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "leakage_fail",
                "matches": audit.matches[:3],
            })
            continue

        sft_rows.append({
            "example_id": eid_int,
            "question_hash": question_hash(row.question),
            "benchmark_name": row.benchmark_name,
            "agent_kind": kind_value,
            "teacher_provider": "deepseek_local",
            "teacher_model": teacher_model,
            "candidate_answer": candidate,
            "prompt": runtime_prompt,
            "response": json.dumps(model.model_dump(), ensure_ascii=False),
        })
        log_rows.append({"example_id": eid_int, "ok": True})

    write_jsonl(out_path, sft_rows)
    write_jsonl(log_path, log_rows)
    return {
        "agent_kind": kind_value,
        "prompt_jsonl": prompt_jsonl,
        "response_jsonl": response_jsonl,
        "out_path": out_path,
        "log_path": log_path,
        "n_responses": len(response_rows),
        "n_imported": len(sft_rows),
        "n_failed": len(response_rows) - len(sft_rows),
        "raw_responses": raw_responses,
    }


# --------------------- Stage: subagent SFT training ---------------------

def run_train_subagent(
    ctx: StageContext,
    agent_kind: AgentKind,
    train_jsonl: Optional[str] = None,
    dev_jsonl: Optional[str] = None,
    epochs: int = 3,
    lr: float = 2e-4,
    max_seq_len: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_steps: int = -1,
) -> Dict[str, Any]:
    from ..subagents.train import SFTConfig, train_subagent_sft

    kind_value = _agent_kind_value(agent_kind)
    if train_jsonl is None:
        train_jsonl = ctx.sft_jsonl_path(kind_value)
    out_dir = ctx.adapter_path(kind_value)

    cfg = SFTConfig(
        base_model=ctx.base_model,
        train_jsonl=train_jsonl,
        dev_jsonl=dev_jsonl,
        out_dir=out_dir,
        max_seq_len=max_seq_len,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        seed=ctx.seed,
        max_steps=max_steps,
    )
    train_subagent_sft(cfg)
    return {"agent_kind": kind_value, "adapter_dir": out_dir, "train_jsonl": train_jsonl}


# --------------------- Stage: subagent eval ---------------------

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def run_eval_subagents(
    ctx: StageContext,
    rows: List[StandardRow],
    agent_kinds: List[AgentKind],
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Evaluate each subagent's schema validity rate on a sample of rows.

    We do NOT score correctness here (subagents don't produce final answers);
    we score (1) does it return parseable JSON, (2) does it pass pydantic
    schema validation. This is the basic 'is the subagent functional' check.
    """
    import torch
    from ..subagents.runtime import FrozenSubagent, SubagentPool
    from ..subagents.schemas import AgentKind, SCHEMA_REGISTRY

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    report: Dict[str, Any] = {
        "teacher_id": ctx.teacher_id, "n_samples": len(sample), "by_agent": {},
    }

    pool = SubagentPool()
    kinds = [AgentKind(_agent_kind_value(k)) for k in agent_kinds]
    for kind in kinds:
        adapter = ctx.adapter_path(kind.value)
        if not os.path.exists(adapter):
            print(f"[EVAL] adapter missing for {kind.value}: {adapter}; skipping.")
            continue
        pool.register(FrozenSubagent(ctx.base_model, adapter, kind.value, device))

    out_log_path = os.path.join(ctx.eval_root, "subagent_eval.jsonl")
    rows_log: List[Dict[str, Any]] = []

    for kind in kinds:
        if not pool.has(kind.value):
            continue
        n_total, n_json_ok, n_schema_ok = 0, 0, 0
        for r in sample:
            n_total += 1
            try:
                text = pool.call(
                    agent_kind=kind.value, example_id=r.example_id,
                    question=r.question, context=r.context, choices=r.choices,
                    cache_namespace=f"eval_{kind.value}",
                )
            except Exception as e:
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "error": str(e)[:300]})
                continue

            obj = _try_parse_json(text)
            if obj is None:
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "json_ok": False, "schema_ok": False,
                                 "raw_preview": text[:300]})
                continue
            n_json_ok += 1

            schema_cls = SCHEMA_REGISTRY[kind]
            try:
                schema_cls(**obj)
                n_schema_ok += 1
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "json_ok": True, "schema_ok": True})
            except Exception as e:
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "json_ok": True, "schema_ok": False,
                                 "schema_error": str(e)[:300]})

        report["by_agent"][kind.value] = {
            "n_total": n_total,
            "json_ok_rate": (n_json_ok / n_total) if n_total else 0.0,
            "schema_ok_rate": (n_schema_ok / n_total) if n_total else 0.0,
        }

    write_jsonl(out_log_path, rows_log)
    write_json(os.path.join(ctx.eval_root, "subagent_eval_report.json"), report)
    print("[EVAL/SUBAGENT]", report["by_agent"])
    return report
