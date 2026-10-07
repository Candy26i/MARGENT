#!/usr/bin/env python3
"""Recompute every reported number from the saved result artifacts.

Standard library only (no GPU, no model weights). Reads:

  <run_dir>/counterfactual_records.jsonl   (build_marginal_sft output)
  <run_dir>/eval_*.jsonl                   (eval_manager_tools outputs)

and prints a Markdown summary with 95% Wilson intervals and exact McNemar
tests. ``--json`` additionally writes the raw numbers.

Usage:
  python scripts/analyze_results.py results/medqa_marginal_v1
  python scripts/analyze_results.py results/medqa_marginal_v1 --json summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

ADVISORS = ("extractor", "reasoner", "verifier")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci_str(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{pct(k / n)} [{pct(lo)}, {pct(hi)}]"


def analyze_counterfactual(path: str) -> Dict[str, Any]:
    records = read_jsonl(path)
    n = len(records)
    direct = [bool(r["direct_correct"]) for r in records]
    out: Dict[str, Any] = {"n": n, "direct_correct": sum(direct), "advisors": {}}

    one_step: Dict[str, List[bool]] = {a: [] for a in ADVISORS}
    for r in records:
        seen = set()
        for br in r["branches"]:
            seq = tuple(br["sequence"])
            if len(seq) == 1 and seq[0] in one_step and seq[0] not in seen:
                one_step[seq[0]].append(bool(br["correct"]))
                seen.add(seq[0])

    for a in ADVISORS:
        forced = one_step[a]
        if len(forced) != n:
            continue
        rescue = sum(1 for d, f in zip(direct, forced) if not d and f)
        corrupt = sum(1 for d, f in zip(direct, forced) if d and not f)
        out["advisors"][a] = {
            "forced_correct": sum(forced),
            "rescue": rescue,
            "corruption": corrupt,
            "mcnemar_p": mcnemar_exact(rescue, corrupt),
        }

    any_correct = [
        d or any(one_step[a][i] for a in ADVISORS if len(one_step[a]) == n)
        for i, d in enumerate(direct)
    ]
    out["oracle_correct"] = sum(any_correct)
    depth = Counter(
        "unsolved" if r["preferred_sequence"] is None else len(r["preferred_sequence"])
        for r in records
    )
    out["preferred_depth"] = {str(k): v for k, v in depth.items()}
    return out


def analyze_eval(path: str) -> Dict[str, Any]:
    rows = read_jsonl(path)
    n = len(rows)
    final = [bool(r["correct"]) for r in rows]
    draft = [bool(r["initial_draft_correct"]) for r in rows]
    corrected = sum(1 for d, f in zip(draft, final) if not d and f)
    corrupted = sum(1 for d, f in zip(draft, final) if d and not f)
    calls = Counter(int(r["tool_calls"]) for r in rows)
    tools = Counter(t for r in rows for t in r.get("tool_names_called", []))

    def call_rate(subset: List[Dict[str, Any]]) -> float:
        return sum(1 for r in subset if r["tool_calls"] > 0) / max(1, len(subset))

    wrong = [r for r in rows if not r["initial_draft_correct"]]
    right = [r for r in rows if r["initial_draft_correct"]]
    ids = [r["example_id"] for r in rows]
    return {
        "n": n,
        "example_id_range": [min(ids), max(ids)],
        "final_correct": sum(final),
        "draft_correct": sum(draft),
        "corrected": corrected,
        "corrupted": corrupted,
        "mcnemar_p": mcnemar_exact(corrected, corrupted),
        "avg_calls": sum(int(r["tool_calls"]) for r in rows) / n,
        "call_distribution": {str(k): calls[k] for k in sorted(calls)},
        "tool_counts": dict(tools),
        "call_rate_given_draft_wrong": call_rate(wrong),
        "call_rate_given_draft_correct": call_rate(right),
        "valid_answer_rate": sum(1 for r in rows if r.get("valid_answer")) / n,
        "final_equals_draft": sum(1 for r in rows if r.get("pred") == r.get("initial_draft")),
    }


def render(cf: Dict[str, Any] | None, evals: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    if cf:
        n = cf["n"]
        lines += [
            f"## Counterfactual one-step branches (n = {n} training questions)",
            "",
            "| Policy | Accuracy % [95% CI] | Rescued | Corrupted | McNemar p vs. direct |",
            "|---|---:|---:|---:|---:|",
            f"| Direct (commit immediately) | {ci_str(cf['direct_correct'], n)} | – | – | – |",
        ]
        for a, s in cf["advisors"].items():
            lines.append(
                f"| Always call {a} | {ci_str(s['forced_correct'], n)} | {s['rescue']} | "
                f"{s['corruption']} | {s['mcnemar_p']:.2g} |"
            )
        lines += [
            f"| Hindsight best one-step branch | {ci_str(cf['oracle_correct'], n)} | – | – | – |",
            "",
            f"Preferred depth of selected SFT targets: {cf['preferred_depth']}",
            "",
        ]
    for name, e in evals.items():
        n = e["n"]
        lines += [
            f"## {name} (n = {n}, example_id {e['example_id_range'][0]}–{e['example_id_range'][1]})",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Initial-draft accuracy % [95% CI] | {ci_str(e['draft_correct'], n)} |",
            f"| Final accuracy % [95% CI] | {ci_str(e['final_correct'], n)} |",
            f"| Corrected (draft wrong → final right) | {e['corrected']} |",
            f"| Corrupted (draft right → final wrong) | {e['corrupted']} |",
            f"| Exact McNemar p (draft vs. final) | {e['mcnemar_p']:.2g} |",
            f"| Avg. advisor calls | {e['avg_calls']:.3f} |",
            f"| Call-count distribution | {e['call_distribution']} |",
            f"| Call rate given draft wrong / correct | {pct(e['call_rate_given_draft_wrong'])} / "
            f"{pct(e['call_rate_given_draft_correct'])} |",
            f"| Final answer equals draft | {e['final_equals_draft']} / {n} |",
            f"| Valid answer rate | {pct(e['valid_answer_rate'])} |",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--json", default="", help="optional path for the raw numbers")
    args = ap.parse_args()

    cf_path = os.path.join(args.run_dir, "counterfactual_records.jsonl")
    cf = analyze_counterfactual(cf_path) if os.path.exists(cf_path) else None
    evals = {
        os.path.basename(p)[: -len(".jsonl")]: analyze_eval(p)
        for p in sorted(glob.glob(os.path.join(args.run_dir, "eval_*.jsonl")))
    }
    print(render(cf, evals))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"counterfactual": cf, "eval": evals}, f, indent=2)


if __name__ == "__main__":
    main()
