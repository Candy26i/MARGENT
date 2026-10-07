"""Manager evaluation stages.

  - run_eval_manager: direct answering without sub-agents (optionally a
    self-consistency baseline).
  - run_eval_manager_tools: the learned delegate-or-commit policy with the
    frozen sub-agents exposed as native tools.
  - run_eval_manager_forced: fixed delegation sequences (fixed-k baselines).
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ..benchmarks.base import StandardRow
from ..manager.chat_template import render_chat
from ..manager.prompt import (
    build_manager_system_prompt,
    build_manager_user_message,
    parse_draft_answer,
    parse_final_answer,
)
from ..utils.io import write_json, write_jsonl
from ..utils.seed import set_seed
from .context import StageContext


def _resolve_binding_mode(ctx: StageContext, manager_dir: str) -> str:
    """Resolve binding mode: explicit ctx setting wins; 'auto' reads the
    manager_run_config.json saved at training time; default 'argument'."""
    binding_mode = ctx.binding_mode
    if binding_mode == "auto":
        run_config = os.path.join(manager_dir, "manager_run_config.json")
        if os.path.exists(run_config):
            try:
                with open(run_config, "r", encoding="utf-8") as f:
                    binding_mode = str(json.load(f).get("binding_mode") or "argument")
            except Exception:
                binding_mode = "argument"
        else:
            binding_mode = "argument"
    return binding_mode


def run_eval_manager(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    n_samples: int = 100,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    task_description: str = "",
    sc_k: int = 1,
    sc_temperature: float = 0.7,
) -> Dict[str, Any]:
    """Evaluate manager accuracy + routing pattern on a sample of rows.

    Note: this uses a SIMPLE one-shot generation (no native tool calling).
    For tool-using eval you'd need to set up the same TRL rollout machinery
    as training; this is a pragmatic accuracy probe.

    sc_k > 1 enables a self-consistency baseline: sample sc_k completions at
    sc_temperature and take the majority vote over parsed answers. This is the
    matched-compute resampling control for RQ1 (compare its token budget to the
    learned orchestrator's delegation budget).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if manager_dir is None:
        manager_dir = ctx.manager_sft_dir()
    # Match the wording the manager saw at training time instead of always
    # claiming argument binding.
    binding_mode = _resolve_binding_mode(ctx, manager_dir)

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    tok = AutoTokenizer.from_pretrained(manager_dir, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    is_adapter = bool(
        os.path.isdir(manager_dir)
        and os.path.exists(os.path.join(manager_dir, "adapter_config.json"))
    )
    if not is_adapter:
        model = AutoModelForCausalLM.from_pretrained(
            manager_dir, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            ctx.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, manager_dir).to(device)
    model.eval()

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    rows_log: List[Dict[str, Any]] = []
    n_correct = 0
    _iter = _tqdm(sample, desc="eval_manager", unit="ex") if _tqdm else sample
    for r in _iter:
        sys_prompt = build_manager_system_prompt(
            label_keys=list(r.choices.keys()), task_description=task_description,
        )
        user_msg = build_manager_user_message(
            example_id=r.example_id, question=r.question,
            context=r.context, choices=r.choices, binding_mode=binding_mode,
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = tok(prompt_text, return_tensors="pt").to(device)

        if sc_k > 1:
            votes: List[str] = []
            previews: List[str] = []
            for _ in range(sc_k):
                gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=max(sc_temperature, 1e-6),
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
                out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                previews.append(out[:200])
                p = parse_final_answer(out, list(r.choices.keys()))
                if p is not None:
                    votes.append(p)
            if votes:
                from collections import Counter
                pred = Counter(votes).most_common(1)[0][0]
            else:
                pred = None
            out = " ||| ".join(previews)
        else:
            do_sample = temperature > 1e-6
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                **({"temperature": max(temperature, 1e-6)} if do_sample else {}),
            )
            out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            pred = parse_final_answer(out, list(r.choices.keys()))

        correct = bool(pred is not None and pred == r.ground_truth)
        if correct:
            n_correct += 1
        rows_log.append({
            "example_id": r.example_id, "ground_truth": r.ground_truth,
            "pred": pred, "correct": correct, "output_preview": out[:600],
        })
        done = len(rows_log)
        if _tqdm and hasattr(_iter, "set_postfix"):
            _iter.set_postfix(acc=f"{n_correct/done:.3f}", correct=n_correct, n=done)

    accuracy = n_correct / max(1, len(sample))
    suffix = f"_sc{sc_k}" if sc_k > 1 else ""
    report = {
        "teacher_id": ctx.teacher_id, "manager_dir": manager_dir,
        "n_samples": len(sample), "accuracy": accuracy,
        "sc_k": sc_k,
    }
    write_jsonl(os.path.join(ctx.eval_root, f"manager_eval{suffix}.jsonl"), rows_log)
    write_json(os.path.join(ctx.eval_root, f"manager_eval_report{suffix}.json"), report)
    print(f"[EVAL/MANAGER] teacher={ctx.teacher_id} acc={accuracy:.3f} sc_k={sc_k} (n={len(sample)})")
    return report


def _manager_tool_schemas(binding_mode: str) -> List[Dict[str, Any]]:
    required = ["example_id"] if binding_mode == "argument" else []
    properties = (
        {
            "example_id": {
                "type": "integer",
                "description": "The current example ID from the user message.",
            }
        }
        if binding_mode == "argument"
        else {}
    )
    verifier_properties = dict(properties)
    verifier_properties["current_draft"] = {
        "type": "string",
        "description": "Your current draft answer key (e.g. \"B\") to audit.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "extractor_tool",
                "description": "Extract decision-relevant factual signals from the question and context.",
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reasoner_tool",
                "description": "Produce a structured reasoning scaffold for the choices.",
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verifier_tool",
                "description": "Identify relevant domain principles and audit the reasoning for logical or computational errors. Pass your current draft answer via current_draft.",
                "parameters": {"type": "object", "properties": verifier_properties, "required": required},
            },
        },
    ]


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


_QWEN35_FUNC_RE = re.compile(r"<function=([A-Za-z0-9_]+)\s*>", re.S)
_QWEN35_PARAM_RE = re.compile(r"<parameter=([A-Za-z0-9_]+)\s*>\s*(.*?)\s*</parameter>", re.S)


def _extract_manager_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse XML tool calls emitted by the chat template.

    Handles the Qwen3 JSON payload form:
        <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    and the Qwen3.5 nested-XML form:
        <tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call>
    """
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text or ""):
        blob = m.group(1)
        name = ""
        args: Any = {}
        try:
            obj = json.loads(blob)
            name = str(obj.get("name") or "").strip()
            args = obj.get("arguments") or {}
        except Exception:
            fm = _QWEN35_FUNC_RE.search(blob or "")
            if fm:
                name = fm.group(1).strip()
                args = {}
                for k, v in _QWEN35_PARAM_RE.findall(blob or ""):
                    v = v.strip()
                    args[k] = int(v) if v.lstrip("-").isdigit() else v
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    content = _TOOL_CALL_RE.sub("", text or "").strip()
    return content, calls


def _tool_call_message(
    tool_name: str, args: Dict[str, Any], call_id: str, content: str = ""
) -> Dict[str, Any]:
    return {
        "role": "assistant",
        # Keep the assistant's own text (DRAFT_ANSWER_ etc.) in the history so
        # eval matches training, where TRL preserves tool-call turn content.
        "content": content,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }],
    }


def _load_manager_for_eval(ctx: StageContext, manager_dir: str, device: str, dtype: Any):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(manager_dir, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    is_full = (
        os.path.exists(os.path.join(manager_dir, "config.json"))
        and not os.path.exists(os.path.join(manager_dir, "adapter_config.json"))
    )
    if is_full:
        model = AutoModelForCausalLM.from_pretrained(
            manager_dir, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            ctx.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, manager_dir).to(device)
    model.eval()
    return tok, model


def run_eval_manager_tools(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    n_samples: int = 100,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    max_tool_calls: int = 3,
    task_description: str = "",
    subagent_server_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the manager with the same frozen subagents used as tools."""
    import torch
    from ..subagents.runtime import FrozenSubagent, SubagentPool

    if manager_dir is None:
        manager_dir = ctx.manager_sft_dir()
    if not os.path.exists(manager_dir):
        raise FileNotFoundError(f"manager_dir not found: {manager_dir}")

    binding_mode = _resolve_binding_mode(ctx, manager_dir)
    if binding_mode == "environment":
        # The local XML tool loop is equivalent to argument binding except the
        # example ID is injected by the evaluator instead of generated by model.
        user_binding_mode = "environment"
    else:
        user_binding_mode = "argument"

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if subagent_server_url:
        from ..subagents.runtime import RemoteSubagentPool
        pool = RemoteSubagentPool(server_url=subagent_server_url)
        print(f"[EVAL] using remote subagent pool -> {subagent_server_url}")
    else:
        pool = SubagentPool()
        # subagent_base_model = "Qwen/Qwen3-4B"
        subagent_base_model = ctx.base_model
        for kind in ("extractor", "reasoner", "verifier"):
            adapter = ctx.adapter_path(kind)
            if os.path.exists(adapter):
                pool.register(FrozenSubagent(subagent_base_model, adapter, kind, device))
        if not pool._agents:
            raise FileNotFoundError(f"No subagent adapters found under {ctx.adapter_root}")

    # pool = SubagentPool()
    # for kind in ("extractor", "reasoner", "verifier"):
    #     adapter = ctx.adapter_path(kind)
    #     if os.path.exists(adapter):
    #         pool.register(FrozenSubagent(ctx.base_model, adapter, kind, device))
    # if not pool._agents:
    #     raise FileNotFoundError(f"No subagent adapters found under {ctx.adapter_root}")

    tok, model = _load_manager_for_eval(ctx, manager_dir, device, dtype)
    tools = _manager_tool_schemas(user_binding_mode)

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    try:
        from tqdm import tqdm as _tqdm2
    except ImportError:
        _tqdm2 = None

    rows_log: List[Dict[str, Any]] = []
    n_correct = 0
    n_valid = 0
    total_tool_calls = 0
    tool_counts: Dict[str, int] = {}
    malformed_tool_calls = 0

    _iter2 = _tqdm2(sample, desc="eval_manager_tools", unit="ex") if _tqdm2 else sample
    for r in _iter2:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_manager_system_prompt(
                    label_keys=list(r.choices.keys()),
                    task_description=task_description,
                ),
            },
            {
                "role": "user",
                "content": build_manager_user_message(
                    example_id=r.example_id,
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    binding_mode=user_binding_mode,
                ),
            },
        ]
        trajectory: List[Dict[str, Any]] = []
        used_tools: List[str] = []
        used_kinds = set()
        final_text = ""

        for step in range(max(1, max_tool_calls + 1)):
            prompt_text = render_chat(tok, messages, add_generation_prompt=True, tools=tools)
            inputs = tok(prompt_text, return_tensors="pt").to(device)
            do_sample = temperature > 1e-6
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                **({"temperature": max(temperature, 1e-6)} if do_sample else {}),
            )
            out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            content, calls = _extract_manager_tool_calls(out)
            final_text = content or out

            if not calls or len(used_tools) >= max_tool_calls:
                messages.append({"role": "assistant", "content": final_text})
                trajectory.append({"role": "assistant", "content": final_text[:2000], "tool_calls": []})
                break

            # Execute every tool call in this turn (up to the remaining
            # budget), mirroring training where TRL runs all emitted calls —
            # taking only calls[0] would silently drop the rest.
            take = calls[: max(0, max_tool_calls - len(used_tools))]
            asst_msg: Dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": []}
            executed: List[Tuple[str, str, str]] = []  # (call_id, tool_name, output)
            for call in take:
                tool_name = call["name"]
                args = dict(call.get("arguments") or {})
                if user_binding_mode == "environment" or "example_id" not in args:
                    args["example_id"] = int(r.example_id)

                call_id = f"eval_{int(r.example_id)}_{len(used_tools)}"
                asst_msg["tool_calls"].append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                })
                used_tools.append(tool_name)
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

                tool_kind = tool_name[:-5] if tool_name.endswith("_tool") else tool_name
                candidate = str(args.get("current_draft") or "") if tool_kind == "verifier" else ""
                if tool_kind in used_kinds:
                    tool_output = '{"error": "tool_already_called", "detail": "each tool may be used at most once"}'
                    executed.append((call_id, tool_name, tool_output))
                    trajectory.append({"role": "assistant", "content": content[:1000], "tool_call": {"name": tool_name, "arguments": args}})
                    continue
                used_kinds.add(tool_kind)
                try:
                    tool_output = pool.call(
                        agent_kind=tool_kind,
                        example_id=int(args.get("example_id", r.example_id)),
                        question=r.question,
                        context=r.context,
                        choices=r.choices,
                        cache_namespace="eval_manager_tools",
                        candidate_answer=candidate,
                    )
                except Exception as e:
                    malformed_tool_calls += 1
                    tool_output = json.dumps({"error": str(e)}, ensure_ascii=False)
                executed.append((call_id, tool_name, tool_output))
                trajectory.append({
                    "role": "assistant",
                    "content": content[:1000],
                    "tool_call": {"name": tool_name, "arguments": args},
                })

            messages.append(asst_msg)
            for call_id, tool_name, tool_output in executed:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": tool_output,
                })
                trajectory.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_output[:2000],
                })

        pred = parse_final_answer(final_text, list(r.choices.keys()))
        correct = bool(pred is not None and pred == r.ground_truth)
        initial_draft: Optional[str] = None
        for event in trajectory:
            if event.get("role") != "assistant":
                continue
            initial_draft = parse_draft_answer(
                str(event.get("content") or ""), list(r.choices.keys())
            )
            if initial_draft is not None:
                break
        # A legacy/direct completion may contain only ANSWER_. Treat that
        # submitted answer as its initial draft for conditional diagnostics.
        if initial_draft is None and not used_tools:
            initial_draft = pred
        initial_draft_correct = bool(
            initial_draft is not None and initial_draft == r.ground_truth
        )
        if pred is not None:
            n_valid += 1
        if correct:
            n_correct += 1
        total_tool_calls += len(used_tools)
        done2 = len(rows_log) + 1
        if _tqdm2 and hasattr(_iter2, "set_postfix"):
            _iter2.set_postfix(
                acc=f"{n_correct/done2:.3f}",
                tools=f"{total_tool_calls/done2:.1f}",
                n=done2,
            )
        rows_log.append({
            "example_id": r.example_id,
            "benchmark_name": r.benchmark_name,
            "task_subtype": r.task_subtype,
            "ground_truth": r.ground_truth,
            "pred": pred,
            "correct": correct,
            "initial_draft": initial_draft,
            "initial_draft_correct": initial_draft_correct,
            "corrected_by_tools": bool(used_tools and initial_draft is not None and not initial_draft_correct and correct),
            "corrupted_by_tools": bool(used_tools and initial_draft_correct and not correct),
            "valid_answer": pred is not None,
            "tool_calls": len(used_tools),
            "tool_names_called": used_tools,
            "final_text": final_text[:2000],
            "trajectory": trajectory,
        })

    n = len(sample)
    with_draft = [r for r in rows_log if r.get("initial_draft") is not None]
    draft_wrong = [r for r in with_draft if not r["initial_draft_correct"]]
    draft_correct = [r for r in with_draft if r["initial_draft_correct"]]
    report = {
        "teacher_id": ctx.teacher_id,
        "manager_dir": manager_dir,
        "n_samples": n,
        "accuracy": n_correct / max(1, n),
        "valid_answer_rate": n_valid / max(1, n),
        "tool_call_rate": sum(1 for r in rows_log if r["tool_calls"] > 0) / max(1, n),
        "avg_tool_calls": total_tool_calls / max(1, n),
        "tool_counts": tool_counts,
        "malformed_tool_calls": malformed_tool_calls,
        "initial_draft_coverage": len(with_draft) / max(1, n),
        "initial_draft_accuracy": sum(r["initial_draft_correct"] for r in with_draft) / max(1, len(with_draft)),
        "call_rate_given_draft_wrong": sum(r["tool_calls"] > 0 for r in draft_wrong) / max(1, len(draft_wrong)),
        "call_rate_given_draft_correct": sum(r["tool_calls"] > 0 for r in draft_correct) / max(1, len(draft_correct)),
        "draft_conditioned_call_gap": (
            sum(r["tool_calls"] > 0 for r in draft_wrong) / max(1, len(draft_wrong))
            - sum(r["tool_calls"] > 0 for r in draft_correct) / max(1, len(draft_correct))
        ),
        "correction_rate": sum(r["corrected_by_tools"] for r in rows_log) / max(1, n),
        "corruption_rate": sum(r["corrupted_by_tools"] for r in rows_log) / max(1, n),
        "binding_mode": binding_mode,
        "subagents": sorted(pool._agents.keys()) if hasattr(pool, "_agents") else ["remote"],
    }
    write_jsonl(os.path.join(ctx.eval_root, "manager_tool_eval.jsonl"), rows_log)
    write_json(os.path.join(ctx.eval_root, "manager_tool_eval_report.json"), report)
    print(
        f"[EVAL/MANAGER_TOOLS] teacher={ctx.teacher_id} "
        f"acc={report['accuracy']:.3f} tool_rate={report['tool_call_rate']:.3f} (n={n})"
    )
    return report


def run_eval_manager_forced(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    forced_tools: Optional[List[str]] = None,
    n_samples: int = 100,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    task_description: str = "",
    out_tag: str = "",
    subagent_server_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the manager under a FIXED delegation sequence (no free choice).

    For each forced advisor, the assistant tool-call turn and the frozen
    advisor's output are injected into the history (mirroring cold-start SFT
    construction); the manager generates only the final answer turn.

    Running this once per advisor subset yields (a) the fixed-k baselines for
    the RQ1 main table and the RQ2 Pareto plot, and (b) the per-question
    inputs for the stopping oracle: oracle reward = max over subsets of
    (correct - cost * k), computed offline from the saved jsonl files.

    The verifier runs its generic audit here (no candidate is passed — the
    manager has not stated a draft in forced mode, and passing ground truth
    would leak).
    """
    import torch
    from ..subagents.runtime import FrozenSubagent, SubagentPool

    forced = [t.strip() for t in (forced_tools or []) if t.strip() and t.strip() != "none"]
    valid_kinds = {"extractor", "reasoner", "verifier"}
    for t in forced:
        if t not in valid_kinds:
            raise ValueError(f"forced tool must be one of {sorted(valid_kinds)}, got {t!r}")

    if manager_dir is None:
        manager_dir = ctx.manager_sft_dir()
    if not os.path.exists(manager_dir):
        raise FileNotFoundError(f"manager_dir not found: {manager_dir}")

    binding_mode = _resolve_binding_mode(ctx, manager_dir)

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if subagent_server_url:
        from ..subagents.runtime import RemoteSubagentPool
        pool = RemoteSubagentPool(server_url=subagent_server_url)
        print(f"[EVAL] using remote subagent pool -> {subagent_server_url}")
    else:
        pool = SubagentPool()
        subagent_base_model = getattr(ctx, "subagent_base_model", "") or ctx.base_model
        for kind in ("extractor", "reasoner", "verifier"):
            adapter = ctx.adapter_path(kind)
            if os.path.exists(adapter):
                pool.register(FrozenSubagent(subagent_base_model, adapter, kind, device))
        if not pool._agents:
            raise FileNotFoundError(f"No subagent adapters found under {ctx.adapter_root}")
        # Fail fast: forced eval often runs many subsets back-to-back;
        # a missing adapter should die here, not at the first pool.call.
        for t in forced:
            if not pool.has(t):
                raise FileNotFoundError(
                    f"forced tool {t!r} has no adapter under {ctx.adapter_root}"
                )
    # pool = SubagentPool()
    # for kind in ("extractor", "reasoner", "verifier"):
    #     adapter = ctx.adapter_path(kind)
    #     if os.path.exists(adapter):
    #         pool.register(FrozenSubagent(ctx.base_model, adapter, kind, device))
    # for t in forced:
    #     if not pool.has(t):
    #         raise FileNotFoundError(f"forced tool {t} has no adapter under {ctx.adapter_root}")

    tok, model = _load_manager_for_eval(ctx, manager_dir, device, dtype)
    tools = _manager_tool_schemas(binding_mode if binding_mode == "argument" else "environment")

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    try:
        from tqdm import tqdm as _tqdm3
    except ImportError:
        _tqdm3 = None

    rows_log: List[Dict[str, Any]] = []
    n_correct = 0
    n_valid = 0
    _iter3 = _tqdm3(sample, desc=f"eval_forced[{','.join(forced) or 'none'}]", unit="ex") if _tqdm3 else sample
    for r in _iter3:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_manager_system_prompt(
                    label_keys=list(r.choices.keys()),
                    task_description=task_description,
                ),
            },
            {
                "role": "user",
                "content": build_manager_user_message(
                    example_id=r.example_id,
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    binding_mode=binding_mode,
                ),
            },
        ]
        for i, kind in enumerate(forced):
            tool_name = f"{kind}_tool"
            args: Dict[str, Any] = (
                {"example_id": int(r.example_id)} if binding_mode == "argument" else {}
            )
            call_id = f"forced_{int(r.example_id)}_{i}"
            messages.append(_tool_call_message(tool_name, args, call_id))
            tool_output = pool.call(
                agent_kind=kind,
                example_id=int(r.example_id),
                question=r.question,
                context=r.context,
                choices=r.choices,
                cache_namespace="eval_forced",
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": tool_output,
            })

        prompt_text = render_chat(tok, messages, add_generation_prompt=True, tools=tools)
        inputs = tok(prompt_text, return_tensors="pt").to(device)
        do_sample = temperature > 1e-6
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            **({"temperature": max(temperature, 1e-6)} if do_sample else {}),
        )
        out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        content, _extra_calls = _extract_manager_tool_calls(out)
        final_text = content or out
        pred = parse_final_answer(final_text, list(r.choices.keys()))
        correct = bool(pred is not None and pred == r.ground_truth)
        if pred is not None:
            n_valid += 1
        if correct:
            n_correct += 1
        rows_log.append({
            "example_id": r.example_id,
            "benchmark_name": r.benchmark_name,
            "task_subtype": r.task_subtype,
            "ground_truth": r.ground_truth,
            "pred": pred,
            "correct": correct,
            "valid_answer": pred is not None,
            "tool_calls": len(forced),
            "forced_tools": list(forced),
            "final_text": final_text[:1200],
        })

    n = len(sample)
    tag = out_tag or (",".join(forced) if forced else "none")
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    report = {
        "teacher_id": ctx.teacher_id,
        "manager_dir": manager_dir,
        "forced_tools": list(forced),
        "k": len(forced),
        "n_samples": n,
        "accuracy": n_correct / max(1, n),
        "valid_answer_rate": n_valid / max(1, n),
        "binding_mode": binding_mode,
    }
    write_jsonl(os.path.join(ctx.eval_root, f"manager_forced_{safe_tag}.jsonl"), rows_log)
    write_json(os.path.join(ctx.eval_root, f"manager_forced_{safe_tag}_report.json"), report)
    print(
        f"[EVAL/FORCED] tools=[{tag}] acc={report['accuracy']:.3f} "
        f"valid={report['valid_answer_rate']:.3f} (n={n})"
    )
    return report
