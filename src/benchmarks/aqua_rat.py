"""AQuA-RAT loader.

Loads the deepmind/aqua_rat dataset from HuggingFace (public, no gating).

AQuA-RAT is a set of algebraic word problems with 5 options per question
(A–E) and a free-text rationale. Options arrive as "A)3/4" strings (letter,
")", text); the loader strips the prefix and keys the choices by letter.

Config: "raw". Splits: "train" (~97k questions), "validation" (254) and
"test" (254). The paper evaluates on the 254-question test split.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .base import StandardRow

HF_DEFAULT_DATASET = "deepmind/aqua_rat"
HF_DEFAULT_CONFIG = "raw"

# "A)3/4" -> letter "A", text "3/4" (tolerates spaces around the ")")
_OPTION_PREFIX = re.compile(r"^\s*([A-Za-z])\s*\)\s*")


def _parse_options(options: Sequence[Any]) -> Dict[str, str]:
    """Turn ["A)125", "B)150", ...] into {"A": "125", "B": "150", ...}.

    An option without the "X)" prefix falls back to its positional letter.
    """
    choices: Dict[str, str] = {}
    for i, opt in enumerate(options):
        if i >= 26:
            break
        s = str(opt).strip()
        m = _OPTION_PREFIX.match(s)
        if m:
            key = m.group(1).upper()
            text = s[m.end():].strip()
        else:
            key = chr(ord("A") + i)
            text = s
        choices[key] = text
    return choices


def _from_record(rec: Dict[str, Any], idx: int, split: str) -> Optional[StandardRow]:
    question = str(rec.get("question") or "").strip()
    options = rec.get("options") or []
    if not question or not options:
        return None

    choices = _parse_options(list(options))
    if len(choices) < 2:
        return None

    # correct is a letter like "A"
    gt = str(rec.get("correct") or "").strip().upper()
    if gt not in choices:
        return None

    rationale = str(rec.get("rationale") or "").strip()
    split = str(split or "").lower().strip()
    if split == "validation":
        split = "dev"
    if not split:
        split = "test"

    return StandardRow(
        example_id=idx,
        benchmark_name="aqua_rat",
        task_subtype="aqua_rat",
        question=question,
        choices=choices,
        ground_truth=gt,
        context="",
        metadata={"rationale": rationale, "n_options": len(choices)},
        split=split,
    )


def load_aqua_rat(
    dataset_name: str = HF_DEFAULT_DATASET,
    splits: "Sequence[str] | str" = ("test",),
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
) -> List[StandardRow]:
    """Load AQuA-RAT into a list of StandardRow.

    Args:
        dataset_name: HuggingFace dataset id.
        splits: which HF splits to load (train / validation / test).
        hf_cache_dir: optional HuggingFace cache directory.
        max_examples: cap total examples; 0 means no cap.
    """
    from datasets import load_dataset

    if isinstance(splits, str):
        split_list = [splits]
    else:
        split_list = list(splits)

    ds = load_dataset(dataset_name, HF_DEFAULT_CONFIG, cache_dir=hf_cache_dir)

    rows: List[StandardRow] = []
    for split_name in split_list:
        if split_name not in ds:
            # Also try "validation" as alias for "dev"
            alt = "validation" if split_name == "dev" else ("dev" if split_name == "validation" else None)
            if alt and alt in ds:
                split_name = alt
            else:
                continue

        for rec in ds[split_name]:
            sr = _from_record(dict(rec), len(rows), split_name)
            if sr is not None:
                rows.append(sr)
            if max_examples > 0 and len(rows) >= max_examples:
                break
        if max_examples > 0 and len(rows) >= max_examples:
            break

    # Reassign contiguous example_ids
    for new_id, r in enumerate(rows):
        r.example_id = new_id

    return rows
