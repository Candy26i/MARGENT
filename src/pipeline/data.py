"""Data-loading stages: benchmark loaders plus the train/dev/test split rule.

Each loader returns normalized StandardRow objects and optionally writes them
to a normalized JSONL cache that the CLI reuses on later runs.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ..benchmarks.aqua_rat import load_aqua_rat
from ..benchmarks.base import StandardRow
from ..benchmarks.gpqa import load_gpqa
from ..benchmarks.medqa import load_medqa
from ..benchmarks.mmlu_pro import load_mmlu_pro
from ..utils.io import write_jsonl


def _split_rows(
    rows: List[StandardRow],
    train_size: int,
    dev_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[StandardRow], List[StandardRow], List[StandardRow]]:
    """Honor existing splits when present; otherwise random-split."""
    by_split: Dict[str, List[StandardRow]] = {"train": [], "dev": [], "test": [], "": []}
    unknown_labels: Dict[str, int] = {}
    for r in rows:
        s = (r.split or "").lower().strip()
        if s == "validation":
            s = "dev"
        if s not in by_split:
            # Unknown labels would otherwise be appended under their own key
            # and silently dropped by every branch below.
            unknown_labels[s] = unknown_labels.get(s, 0) + 1
            s = ""
        by_split[s].append(r)
    if unknown_labels:
        print(f"[SPLIT] WARNING: unrecognized split labels folded into unlabeled pool: {unknown_labels}")

    # Any explicit train label wins: a train-only cache (e.g. the GPQA train
    # split built by scripts/build_gpqa_splits.py) must never have a phantom
    # test set carved out of its training rows by the random path below.
    have_explicit = bool(by_split["train"])
    if train_size == 0 and not by_split["train"]:
        # Eval-only pool (e.g. GPQA-Diamond or MMLU-Pro used as zero-shot
        # probes with --train_size 0): honor the loader's split labels.
        # The random path below would set n_test = min(test_size, n//4) and
        # silently shrink a 198-question probe to 49 rows.
        train = []
        dev = list(by_split["dev"])
        test = list(by_split["test"]) or list(by_split[""])
    elif have_explicit:
        train = by_split["train"]
        dev = by_split["dev"]
        # Never alias test to dev: any dev-driven decision (threshold picking,
        # early stopping, model selection) would leak straight into the test
        # report. Callers already fall back to dev when test is empty.
        test = by_split["test"]
        if not dev:
            # No explicit dev split: carve dev from the train tail rather than
            # aliasing test (dev ⊂ test would leak eval rows into any
            # dev-driven decision).
            n_dev = min(max(dev_size, 0), len(train) // 5)
            if n_dev > 0:
                dev = train[-n_dev:]
                train = train[:-n_dev]
            else:
                dev = []
    else:
        rng = random.Random(seed)
        all_rows = list(rows)
        rng.shuffle(all_rows)
        n = len(all_rows)
        n_test = min(test_size, n // 4)
        n_dev = min(dev_size, (n - n_test) // 4)
        test = all_rows[:n_test]
        dev = all_rows[n_test:n_test + n_dev]
        train = all_rows[n_test + n_dev:]

    if train_size > 0 and len(train) > train_size:
        train = train[:train_size]
    if dev_size > 0 and len(dev) > dev_size:
        dev = dev[:dev_size]
    if test_size > 0 and len(test) > test_size:
        test = test[:test_size]
    return train, dev, test


# --------------------- Stage: data loading ---------------------

def run_load_medqa(
    source: str = "hf",
    hf_dataset: str = "GBaker/MedQA-USMLE-4-options",
    local_path: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    cache_normalized_path: Optional[str] = None,
) -> List[StandardRow]:
    rows = load_medqa(
        source=source, hf_dataset=hf_dataset,
        local_path=local_path, hf_cache_dir=hf_cache_dir,
        max_examples=max_examples,
    )
    print(f"[LOAD_MEDQA] loaded {len(rows)} rows from {source}")
    if cache_normalized_path:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        print(f"[LOAD_MEDQA] cached normalized rows -> {cache_normalized_path}")
    return rows


# --------------------- Stage: GPQA loading ---------------------

def run_load_gpqa(
    dataset_name: str = "Idavidrein/gpqa",
    subsets: str = "gpqa_diamond",
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    answer_seed: int = 42,
    cache_normalized_path: Optional[str] = None,
    exclude_subsets: str = "",
) -> List[StandardRow]:
    rows = load_gpqa(
        dataset_name=dataset_name,
        subsets=subsets,
        hf_cache_dir=hf_cache_dir,
        max_examples=max_examples,
        answer_seed=answer_seed,
        exclude_subsets=exclude_subsets,
    )
    print(f"[LOAD_GPQA] loaded {len(rows)} rows  subsets={subsets}  exclude={exclude_subsets or 'none'}")
    if cache_normalized_path and rows:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        print(f"[LOAD_GPQA] cached normalized rows -> {cache_normalized_path}")
    elif cache_normalized_path:
        # Never cache an empty result (e.g. gated-dataset auth failure) — a
        # 0-row cache would be silently loaded by every subsequent run.
        print("[LOAD_GPQA] 0 rows loaded; NOT writing cache (fix HF auth and rerun).")
    return rows


# --------------------- Stage: MMLU-Pro loading ---------------------

def run_load_mmlu_pro(
    dataset_name: str = "TIGER-Lab/MMLU-Pro",
    categories: str = "",
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    splits: str = "test,validation",
    cache_normalized_path: Optional[str] = None,
) -> List[StandardRow]:
    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    rows = load_mmlu_pro(
        dataset_name=dataset_name,
        categories=categories,
        hf_cache_dir=hf_cache_dir,
        max_examples=max_examples,
        splits=split_list,
    )
    cat_desc = categories or "all"
    print(f"[LOAD_MMLU_PRO] loaded {len(rows)} rows  categories={cat_desc}")
    if cache_normalized_path:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        print(f"[LOAD_MMLU_PRO] cached normalized rows -> {cache_normalized_path}")
    return rows


# --------------------- Stage: AQuA-RAT loading ---------------------

def run_load_aqua_rat(
    dataset_name: str = "deepmind/aqua_rat",
    splits: str = "test",
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    cache_normalized_path: Optional[str] = None,
) -> List[StandardRow]:
    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    rows = load_aqua_rat(
        dataset_name=dataset_name,
        splits=split_list,
        hf_cache_dir=hf_cache_dir,
        max_examples=max_examples,
    )
    print(f"[LOAD_AQUA_RAT] loaded {len(rows)} rows  splits={splits}")
    if cache_normalized_path:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        print(f"[LOAD_AQUA_RAT] cached normalized rows -> {cache_normalized_path}")
    return rows
