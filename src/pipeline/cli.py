"""CLI entry point.

Usage examples (see README for full walkthroughs):

  # Synthesize 500 reasoner samples with Claude as teacher
  python -m src.pipeline.cli synth_subagent \\
      --teacher_provider anthropic --teacher_model claude-sonnet-4-5 \\
      --teacher_id claude_sonnet_4_5 \\
      --agent_kind reasoner --n_samples 500

  # Train the reasoner subagent
  python -m src.pipeline.cli train_subagent \\
      --teacher_id claude_sonnet_4_5 --agent_kind reasoner

  # Collect paired counterfactual branches and build the routing SFT data
  python -m src.pipeline.cli build_marginal_sft \\
      --teacher_id claude_sonnet_4_5

  # Distill the manager on manager_sft_marginal.jsonl
  python -m src.pipeline.cli train_manager_sft \\
      --teacher_id claude_sonnet_4_5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from ..benchmarks.base import StandardRow
from ..utils.io import read_jsonl
from .context import StageContext
from .data import _split_rows, run_load_aqua_rat, run_load_gpqa, run_load_medqa, run_load_mmlu_pro
from .subagent_stages import (
    run_eval_subagents,
    run_export_deepseek_subagent_prompts,
    run_import_deepseek_subagent_responses,
    run_synthesize_subagent,
    run_train_subagent,
)
from .manager_stages import run_build_marginal_sft, run_train_manager_sft
from .eval_stages import run_eval_manager, run_eval_manager_forced, run_eval_manager_tools


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent_routing")
    parser.add_argument("stage", type=str, choices=[
        "load_medqa",
        "load_mmlu_pro",
        "load_gpqa",
        "load_aqua_rat",
        "synth_subagent",
        "export_deepseek_jsonl",
        "import_deepseek_jsonl",
        "train_subagent",
        "eval_subagents",
        "build_marginal_sft",
        "train_manager_sft",
        "eval_manager",
        "eval_manager_tools",
        "eval_manager_forced",
    ])

    # Context-level flags
    ctx_group = parser.add_argument_group("context", "Model, run namespace and output layout")
    ctx_group.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B",
                           help="Manager and sub-agent base model (paper: Qwen/Qwen3.5-9B).")
    ctx_group.add_argument("--teacher_id", type=str, default="default",
                           help="Logical id used to namespace outputs (e.g. mmlu_pro_gpt54).")
    ctx_group.add_argument("--subagent_teacher_id", type=str, default="",
                           help="If set, load subagent adapters from this teacher_id's adapter dir instead of --teacher_id. "
                                "Use when reusing subagents trained under a different run (e.g. --teacher_id mmlu_pro_gpt54 "
                                "--subagent_teacher_id mmlu_pro_claude).")
    ctx_group.add_argument("--output_root", type=str, default="outputs")
    ctx_group.add_argument("--seed", type=int, default=42)
    ctx_group.add_argument("--binding_mode", type=str, default="auto",
                           choices=["auto", "environment", "argument"],
                           help="Tool-binding wording. auto = environment for build_marginal_sft and "
                                "train_manager_sft; the eval stages read manager_run_config.json from "
                                "the manager directory (environment when absent).")
    ctx_group.add_argument("--task_description", type=str, default="")
    ctx_group.add_argument("--subagent_server_url", type=str, default="",
                           help="vLLM HTTP server URL for subagents, e.g. http://localhost:8000. "
                                "When set, no subagent weights are loaded into the calling process.")

    # MedQA loading
    medqa_group = parser.add_argument_group("medqa", "MedQA loading (load_medqa; default data source)")
    medqa_group.add_argument("--medqa_source", type=str, default="hf", choices=["hf", "local"])
    medqa_group.add_argument("--medqa_hf_dataset", type=str, default="GBaker/MedQA-USMLE-4-options")
    medqa_group.add_argument("--medqa_local_path", type=str, default="")
    medqa_group.add_argument("--medqa_hf_cache", type=str, default="")
    medqa_group.add_argument("--medqa_max", type=int, default=0)
    medqa_group.add_argument("--medqa_normalized_cache", type=str, default="")
    medqa_group.add_argument("--medqa_refresh_cache", action="store_true",
                             help="Reload MedQA from the requested source/path and overwrite the normalized cache.")

    # GPQA loading
    gpqa_group = parser.add_argument_group("gpqa", "GPQA loading (load_gpqa)")
    gpqa_group.add_argument("--gpqa_hf_dataset", type=str, default="Idavidrein/gpqa")
    gpqa_group.add_argument("--gpqa_subsets", type=str, default="gpqa_diamond",
                            help="Comma-separated GPQA subset names: gpqa_main, gpqa_diamond, "
                                 "gpqa_extended, or 'all'. Default: gpqa_diamond.")
    gpqa_group.add_argument("--gpqa_hf_cache", type=str, default="")
    gpqa_group.add_argument("--gpqa_max", type=int, default=0)
    gpqa_group.add_argument("--gpqa_answer_seed", type=int, default=42,
                            help="Seed for A/B/C/D answer shuffling (keeps mapping deterministic).")
    gpqa_group.add_argument("--gpqa_exclude_subsets", type=str, default="",
                            help="Comma-separated GPQA subsets whose questions are REMOVED from "
                                 "the loaded rows. GPQA subsets are nested (diamond ⊆ main ⊆ "
                                 "extended); pass 'gpqa_diamond' when training on main/extended "
                                 "and evaluating on diamond to avoid contamination.")
    gpqa_group.add_argument("--gpqa_normalized_cache", type=str, default="")
    gpqa_group.add_argument("--gpqa_refresh_cache", action="store_true")

    # MMLU-Pro loading
    mmlu_group = parser.add_argument_group("mmlu_pro", "MMLU-Pro loading (load_mmlu_pro)")
    mmlu_group.add_argument("--mmlu_pro_hf_dataset", type=str, default="TIGER-Lab/MMLU-Pro")
    mmlu_group.add_argument("--mmlu_pro_categories", type=str, default="",
                            help="Comma-separated category names to keep, e.g. 'math,physics'. "
                                 "Empty means all categories.")
    mmlu_group.add_argument("--mmlu_pro_hf_cache", type=str, default="")
    mmlu_group.add_argument("--mmlu_pro_max", type=int, default=0)
    mmlu_group.add_argument("--mmlu_pro_splits", type=str, default="test,validation",
                            help="Comma-separated HF split names to load.")
    mmlu_group.add_argument("--mmlu_pro_normalized_cache", type=str, default="")
    mmlu_group.add_argument("--mmlu_pro_refresh_cache", action="store_true")

    # AQuA-RAT loading
    aqua_group = parser.add_argument_group("aqua_rat", "AQuA-RAT loading (load_aqua_rat)")
    aqua_group.add_argument("--aqua_rat_hf_dataset", type=str, default="deepmind/aqua_rat")
    aqua_group.add_argument("--aqua_rat_splits", type=str, default="test",
                            help="Comma-separated HF split names to load (train, validation, test). "
                                 "Default: test (the paper's 254-question evaluation set).")
    aqua_group.add_argument("--aqua_rat_hf_cache", type=str, default="")
    aqua_group.add_argument("--aqua_rat_max", type=int, default=0)
    aqua_group.add_argument("--aqua_rat_normalized_cache", type=str, default="")
    aqua_group.add_argument("--aqua_rat_refresh_cache", action="store_true")

    # Split sizes
    split_group = parser.add_argument_group("splits", "Train/dev/test split sizes")
    split_group.add_argument("--train_size", type=int, default=600, help="(paper, MedQA: 1400)")
    split_group.add_argument("--dev_size", type=int, default=100, help="(paper, MedQA: 200)")
    split_group.add_argument("--test_size", type=int, default=200, help="(paper, MedQA: 500)")

    # Synth
    synth_group = parser.add_argument_group(
        "subagent synthesis",
        "synth_subagent / export_deepseek_jsonl / import_deepseek_jsonl",
    )
    synth_group.add_argument("--teacher_provider", type=str, default="",
                             choices=["", "anthropic", "claude", "openai", "gpt", "deepseek"])
    synth_group.add_argument("--teacher_model", type=str, default="")
    synth_group.add_argument("--agent_kind", type=str, default="",
                             choices=["", "extractor", "reasoner", "verifier"])
    synth_group.add_argument("--n_samples", type=int, default=500)
    synth_group.add_argument("--synth_temperature", type=float, default=0.4)
    synth_group.add_argument("--synth_max_retries", type=int, default=2)
    synth_group.add_argument("--synth_workers", type=int, default=8,
                             help="Parallel teacher API calls during synthesis (default 8).")
    synth_group.add_argument("--synth_no_cache", action="store_true")
    synth_group.add_argument("--synth_symmetric_leakage", action="store_true",
                             help="Leakage-audit against ALL choice texts instead of only the "
                                  "ground-truth text. Removes the negative-space bias where the "
                                  "one never-restated choice is exactly the answer.")
    synth_group.add_argument("--deepseek_prompt_jsonl", type=str, default="",
                             help="Prompt JSONL for local DeepSeek batch generation.")
    synth_group.add_argument("--deepseek_response_jsonl", type=str, default="",
                             help="Response JSONL produced by local DeepSeek generate_jsonl.py.")
    synth_group.add_argument("--deepseek_sft_jsonl", type=str, default="",
                             help="Optional imported SFT JSONL output path.")
    synth_group.add_argument("--deepseek_teacher_model", type=str, default="deepseek-local",
                             help="Metadata model name used when importing local DeepSeek responses.")
    synth_group.add_argument("--deepseek_import_raw_responses", action="store_true",
                             help="Import response text as-is, but pair it with runtime subagent prompts instead of validating/filtering.")

    # Subagent SFT (the --sft_* batch/sequence flags are shared with train_manager_sft)
    sft_group = parser.add_argument_group("subagent sft", "train_subagent (batch/sequence flags also apply to train_manager_sft)")
    sft_group.add_argument("--sft_epochs", type=int, default=3)
    sft_group.add_argument("--sft_lr", type=float, default=2e-4,
                           help="Sub-agent SFT learning rate (the README walkthrough uses 5e-5).")
    sft_group.add_argument("--sft_max_seq_len", type=int, default=4096)
    sft_group.add_argument("--sft_bs", type=int, default=1)
    sft_group.add_argument("--sft_grad_accum", type=int, default=8)
    sft_group.add_argument("--sft_max_steps", type=int, default=-1)
    sft_group.add_argument("--sft_no_lora", action="store_true")
    sft_group.add_argument("--sft_train_jsonl", type=str, default="",
                           help="Optional explicit SFT JSONL path for train_subagent. Must contain prompt and response fields.")
    sft_group.add_argument("--sft_dev_jsonl", type=str, default="")

    # Counterfactual marginal-value routing data
    mv_group = parser.add_argument_group("marginal value", "build_marginal_sft (interventional collection)")
    mv_group.add_argument("--mv_manager_dir", type=str, default="",
                          help="Manager checkpoint used to generate the shared initial draft and revise counterfactual branches. Empty uses --base_model.")
    mv_group.add_argument("--mv_n_samples", type=int, default=300,
                          help="Number of training questions used for marginal-value branch collection.")
    mv_group.add_argument("--mv_max_depth", type=int, default=1, choices=[1, 2, 3],
                          help="Maximum number of distinct sub-agents in a forced counterfactual branch. Start with 1; use 2/3 only if the one-step oracle leaves useful headroom (paper: 3 on MedQA and AQuA-RAT, 2 on MMLU-Pro and GPQA).")
    mv_group.add_argument("--mv_max_new_tokens", type=int, default=512,
                          help="Manager token budget for each counterfactual answer probe.")
    mv_group.add_argument("--mv_temperature", type=float, default=0.0,
                          help="Counterfactual manager sampling temperature. Keep 0 for paired, deterministic comparisons.")
    mv_group.add_argument("--mv_max_commit_rescue_ratio", type=float, default=1.0,
                          help="Cap direct-correct commit decisions per rescued decision in marginal SFT. Negative keeps all commits; 1.0 balances commit and rescue decisions (the paper's rho).")
    mv_group.add_argument("--mv_output_dir", type=str, default="",
                          help="Optional output directory for counterfactual records, report, and manager_sft_marginal.jsonl.")
    mv_group.add_argument("--exclude_sft_example_ids", action="append", default=[],
                          help="JSONL path(s), comma-separated or repeated, whose example_id values are excluded from the manager training rows.")

    # Manager SFT
    msft_group = parser.add_argument_group("manager sft", "train_manager_sft (marginal-value distillation)")
    msft_group.add_argument("--manager_sft_train_jsonl", type=str, default="",
                            help="Optional explicit manager SFT JSONL for train_manager_sft.")
    msft_group.add_argument("--manager_sft_init_adapter", type=str, default="",
                            help="Optional existing manager adapter/full checkpoint to continue SFT from instead of restarting from --base_model.")
    msft_group.add_argument("--manager_sft_output_dir", type=str, default="",
                            help="Optional explicit output directory for train_manager_sft.")
    msft_group.add_argument("--manager_sft_lr", type=float, default=2e-5,
                            help="Manager SFT learning rate (paper: 1e-5).")
    msft_group.add_argument("--manager_sft_epochs", type=int, default=1)

    # Eval
    eval_group = parser.add_argument_group("eval", "eval_subagents / eval_manager / eval_manager_tools / eval_manager_forced")
    eval_group.add_argument("--eval_n_samples", type=int, default=100)
    eval_group.add_argument("--eval_kinds", type=str, default="extractor,reasoner,verifier")
    eval_group.add_argument("--eval_manager_dir", type=str, default="")
    eval_group.add_argument("--eval_temperature", type=float, default=0.0)
    eval_group.add_argument("--eval_max_new_tokens", type=int, default=1024)
    eval_group.add_argument("--eval_max_tool_calls", type=int, default=3)
    eval_group.add_argument("--eval_forced_tools", type=str, default="none",
                            help="Fixed delegation sequence for eval_manager_forced: comma-separated "
                                 "sub-agent kinds, e.g. 'extractor,reasoner,verifier', or 'none' for "
                                 "the zero-delegation baseline. Running every subset yields fixed-k "
                                 "baselines and the per-question stopping oracle.")
    eval_group.add_argument("--eval_out_tag", type=str, default="",
                            help="Optional filename tag for eval_manager_forced outputs.")
    eval_group.add_argument("--eval_sc_k", type=int, default=1,
                            help="Self-consistency baseline for eval_manager: sample k completions "
                                 "and majority-vote (k=1 disables; use as the matched-compute "
                                 "resampling control).")
    eval_group.add_argument("--eval_sc_temperature", type=float, default=0.7,
                            help="Sampling temperature for the self-consistency baseline.")

    return parser.parse_args()


def _ctx_from(args) -> StageContext:
    return StageContext(
        base_model=args.base_model,
        teacher_id=args.teacher_id,
        output_root=args.output_root,
        seed=args.seed,
        binding_mode=args.binding_mode,
        subagent_teacher_id=args.subagent_teacher_id,
    )


def _load_medqa_splits(args) -> dict:
    """Load MedQA, split into train/dev/test, also serialize splits to disk."""
    cache = args.medqa_normalized_cache or os.path.join(
        args.output_root, "data", "medqa_normalized.jsonl"
    )
    if args.medqa_refresh_cache or not os.path.exists(cache):
        rows = run_load_medqa(
            source=args.medqa_source,
            hf_dataset=args.medqa_hf_dataset,
            local_path=(args.medqa_local_path or None),
            hf_cache_dir=(args.medqa_hf_cache or None),
            max_examples=args.medqa_max,
            cache_normalized_path=cache,
        )
    else:
        rows = [StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_MEDQA] loaded cached {len(rows)} rows -> {cache}")

    train, dev, test = _split_rows(
        rows=rows, train_size=args.train_size, dev_size=args.dev_size,
        test_size=args.test_size, seed=args.seed,
    )
    print(f"[SPLIT] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
    return {"all": rows, "train": train, "dev": dev, "test": test}


def _using_gpqa(args) -> bool:
    # NOTE: --gpqa_subsets has a non-empty default, so only the cache path
    # (or the load_gpqa stage itself) activates the GPQA branch.
    return bool(args.gpqa_normalized_cache)


def _using_mmlu_pro(args) -> bool:
    return bool(args.mmlu_pro_normalized_cache or args.mmlu_pro_categories != "")


def _using_aqua_rat(args) -> bool:
    # NOTE: --aqua_rat_splits has a non-empty default, so only the cache path
    # (or the load_aqua_rat stage itself) activates the AQuA-RAT branch.
    return bool(args.aqua_rat_normalized_cache)


def _load_gpqa_or_cache(args) -> List[StandardRow]:
    cache = args.gpqa_normalized_cache or os.path.join(
        args.output_root, "data", "gpqa_normalized.jsonl"
    )
    if args.gpqa_refresh_cache or not os.path.exists(cache):
        rows = run_load_gpqa(
            dataset_name=args.gpqa_hf_dataset,
            subsets=args.gpqa_subsets,
            hf_cache_dir=(args.gpqa_hf_cache or None),
            max_examples=args.gpqa_max,
            answer_seed=args.gpqa_answer_seed,
            cache_normalized_path=cache,
            exclude_subsets=args.gpqa_exclude_subsets,
        )
    else:
        rows = [StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_GPQA] loaded cached {len(rows)} rows -> {cache}")
    return rows


def _load_mmlu_pro_or_cache(args) -> List[StandardRow]:
    cache = args.mmlu_pro_normalized_cache or os.path.join(
        args.output_root, "data", "mmlu_pro_normalized.jsonl"
    )
    if args.mmlu_pro_refresh_cache or not os.path.exists(cache):
        rows = run_load_mmlu_pro(
            dataset_name=args.mmlu_pro_hf_dataset,
            categories=args.mmlu_pro_categories,
            hf_cache_dir=(args.mmlu_pro_hf_cache or None),
            max_examples=args.mmlu_pro_max,
            splits=args.mmlu_pro_splits,
            cache_normalized_path=cache,
        )
    else:
        rows = [StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_MMLU_PRO] loaded cached {len(rows)} rows -> {cache}")
    return rows


def _load_aqua_rat_or_cache(args) -> List[StandardRow]:
    cache = args.aqua_rat_normalized_cache or os.path.join(
        args.output_root, "data", "aqua_rat_normalized.jsonl"
    )
    if args.aqua_rat_refresh_cache or not os.path.exists(cache):
        rows = run_load_aqua_rat(
            dataset_name=args.aqua_rat_hf_dataset,
            splits=args.aqua_rat_splits,
            hf_cache_dir=(args.aqua_rat_hf_cache or None),
            max_examples=args.aqua_rat_max,
            cache_normalized_path=cache,
        )
    else:
        rows = [StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_AQUA_RAT] loaded cached {len(rows)} rows -> {cache}")
    return rows


# (stage, flag-based activation, loader, log tag); the first active entry wins.
_BENCHMARKS = (
    ("load_mmlu_pro", _using_mmlu_pro, _load_mmlu_pro_or_cache, "MMLU_PRO"),
    ("load_gpqa", _using_gpqa, _load_gpqa_or_cache, "GPQA"),
    ("load_aqua_rat", _using_aqua_rat, _load_aqua_rat_or_cache, "AQUA_RAT"),
)


def _load_benchmark_splits(args) -> dict:
    """Load the requested benchmark and return train/dev/test splits.

    Priority (first active wins): mmlu_pro > gpqa > aqua_rat > medqa. A
    benchmark is active when its stage is requested or its normalized cache
    (for MMLU-Pro also a category filter) is set. GPQA and MMLU-Pro have no
    predefined train/dev/test split, so rows are split deterministically using
    --train_size / --dev_size / --test_size. AQuA-RAT keeps its own
    train/validation/test labels (validation -> dev).
    """
    for stage, using, load, tag in _BENCHMARKS:
        if args.stage == stage or using(args):
            rows = load(args)
            train, dev, test = _split_rows(
                rows=rows, train_size=args.train_size, dev_size=args.dev_size,
                test_size=args.test_size, seed=args.seed,
            )
            print(f"[SPLIT/{tag}] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
            return {"all": rows, "train": train, "dev": dev, "test": test}

    return _load_medqa_splits(args)


def _load_eval_rows(args) -> List[StandardRow]:
    data = _load_benchmark_splits(args)
    if args.test_size <= 0:
        return data["dev"]
    return data["test"] or data["dev"]


def _exclude_sft_rows(rows: List[StandardRow], paths: List[str]) -> List[StandardRow]:
    from ..benchmarks.base import question_hash

    expanded: List[str] = []
    for item in paths or []:
        expanded.extend([p.strip() for p in item.split(",") if p.strip()])
    if not expanded:
        return rows

    exclude_ids = set()
    exclude_hashes = set()
    for path in expanded:
        if not os.path.exists(path):
            raise FileNotFoundError(f"exclude_sft_example_ids path not found: {path}")
        for row in read_jsonl(path):
            if row.get("example_id") is not None:
                exclude_ids.add(int(row["example_id"]))
            # question_hash survives normalized-cache rebuilds; example_id does not.
            if row.get("question_hash"):
                exclude_hashes.add(str(row["question_hash"]))

    kept = [
        r for r in rows
        if int(r.example_id) not in exclude_ids
        and question_hash(r.question) not in exclude_hashes
    ]
    print(
        f"[EXCLUDE_SFT] files={len(expanded)} ids={len(exclude_ids)} "
        f"hashes={len(exclude_hashes)} train_rows {len(rows)} -> {len(kept)}"
    )
    if not kept:
        raise ValueError("No manager training rows left after excluding SFT example_ids.")
    return kept


def main() -> None:
    args = _parse_args()
    ctx = _ctx_from(args)

    if args.stage == "load_medqa":
        _load_medqa_splits(args)
        return

    if args.stage.startswith("load_"):
        _load_benchmark_splits(args)
        return

    if args.stage == "synth_subagent":
        if not (args.teacher_provider and args.teacher_model and args.agent_kind):
            sys.exit("synth_subagent requires --teacher_provider, --teacher_model, --agent_kind")
        data = _load_benchmark_splits(args)
        kind = args.agent_kind
        # Synthesize on the train pool
        result = run_synthesize_subagent(
            ctx=ctx, rows=data["train"], agent_kind=kind,
            teacher_provider=args.teacher_provider, teacher_model=args.teacher_model,
            n_samples=args.n_samples,
            base_temperature=args.synth_temperature,
            max_retries=args.synth_max_retries,
            use_cache=(not args.synth_no_cache),
            max_workers=args.synth_workers,
            symmetric_leakage=args.synth_symmetric_leakage,
        )
        print("[SYNTH]", result)
        return

    if args.stage == "export_deepseek_jsonl":
        if not args.agent_kind:
            sys.exit("export_deepseek_jsonl requires --agent_kind")
        data = _load_benchmark_splits(args)
        kind = args.agent_kind
        result = run_export_deepseek_subagent_prompts(
            ctx=ctx,
            rows=data["train"],
            agent_kind=kind,
            out_path=(args.deepseek_prompt_jsonl or None),
            n_samples=args.n_samples,
        )
        print("[EXPORT_DEEPSEEK_JSONL]", result)
        return

    if args.stage == "import_deepseek_jsonl":
        if not args.agent_kind:
            sys.exit("import_deepseek_jsonl requires --agent_kind")
        if not (args.deepseek_prompt_jsonl and args.deepseek_response_jsonl):
            sys.exit("import_deepseek_jsonl requires --deepseek_prompt_jsonl and --deepseek_response_jsonl")
        kind = args.agent_kind
        result = run_import_deepseek_subagent_responses(
            ctx=ctx,
            agent_kind=kind,
            prompt_jsonl=args.deepseek_prompt_jsonl,
            response_jsonl=args.deepseek_response_jsonl,
            out_path=(args.deepseek_sft_jsonl or None),
            teacher_model=args.deepseek_teacher_model,
            raw_responses=args.deepseek_import_raw_responses,
        )
        print("[IMPORT_DEEPSEEK_JSONL]", result)
        return

    if args.stage == "train_subagent":
        if not args.agent_kind:
            sys.exit("train_subagent requires --agent_kind")
        kind = args.agent_kind
        result = run_train_subagent(
            ctx=ctx, agent_kind=kind,
            train_jsonl=(args.sft_train_jsonl or None),
            dev_jsonl=(args.sft_dev_jsonl or None),
            epochs=args.sft_epochs, lr=args.sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        print("[TRAIN_SUBAGENT]", result)
        return

    if args.stage == "build_marginal_sft":
        data = _load_benchmark_splits(args)
        train_rows = _exclude_sft_rows(data["train"], args.exclude_sft_example_ids)
        result = run_build_marginal_sft(
            ctx=ctx,
            rows=train_rows,
            manager_dir=(args.mv_manager_dir or None),
            n_samples=args.mv_n_samples,
            max_depth=args.mv_max_depth,
            max_new_tokens=args.mv_max_new_tokens,
            temperature=args.mv_temperature,
            max_commit_rescue_ratio=args.mv_max_commit_rescue_ratio,
            task_description=args.task_description,
            output_dir=(args.mv_output_dir or None),
            subagent_server_url=(args.subagent_server_url or None),
        )
        print("[BUILD_MARGINAL_SFT]", result)
        return

    if args.stage == "train_manager_sft":
        result = run_train_manager_sft(
            ctx=ctx,
            train_jsonl=(args.manager_sft_train_jsonl or None),
            init_model_or_adapter=(args.manager_sft_init_adapter or None),
            output_dir=(args.manager_sft_output_dir or None),
            epochs=args.manager_sft_epochs,
            lr=args.manager_sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        print("[TRAIN_MGR_SFT]", result)
        return

    if args.stage == "eval_subagents":
        data = _load_benchmark_splits(args)
        kinds = [k.strip() for k in args.eval_kinds.split(",") if k.strip()]
        result = run_eval_subagents(
            ctx=ctx, rows=data["dev"] or data["test"], agent_kinds=kinds,
            n_samples=args.eval_n_samples,
        )
        print("[EVAL_SUBAGENTS]", result["by_agent"])
        return

    if args.stage == "eval_manager":
        result = run_eval_manager(
            ctx=ctx, rows=_load_eval_rows(args),
            manager_dir=(args.eval_manager_dir or None),
            n_samples=args.eval_n_samples,
            temperature=args.eval_temperature,
            max_new_tokens=args.eval_max_new_tokens,
            task_description=args.task_description,
            sc_k=args.eval_sc_k,
            sc_temperature=args.eval_sc_temperature,
        )
        print("[EVAL_MANAGER]", result)
        return

    if args.stage == "eval_manager_forced":
        forced = [t.strip() for t in args.eval_forced_tools.split(",") if t.strip()]
        if forced == ["none"]:
            forced = []
        result = run_eval_manager_forced(
            ctx=ctx, rows=_load_eval_rows(args),
            manager_dir=(args.eval_manager_dir or None),
            forced_tools=forced,
            n_samples=args.eval_n_samples,
            temperature=args.eval_temperature,
            max_new_tokens=args.eval_max_new_tokens,
            task_description=args.task_description,
            out_tag=args.eval_out_tag,
            subagent_server_url=(args.subagent_server_url or None),
        )
        print("[EVAL_MANAGER_FORCED]", result)
        return

    if args.stage == "eval_manager_tools":
        result = run_eval_manager_tools(
            ctx=ctx, rows=_load_eval_rows(args),
            manager_dir=(args.eval_manager_dir or None),
            n_samples=args.eval_n_samples,
            temperature=args.eval_temperature,
            max_new_tokens=args.eval_max_new_tokens,
            max_tool_calls=args.eval_max_tool_calls,
            task_description=args.task_description,
            subagent_server_url=(args.subagent_server_url or None),
        )
        print("[EVAL_MANAGER_TOOLS]", result)
        return

    sys.exit(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
