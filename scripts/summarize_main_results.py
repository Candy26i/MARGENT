#!/usr/bin/env python3
"""Render the predecessor scaling diagnostic (paper Appendix B, Table 6) from
results/predecessor_scaling/main_results.json.

Prints, for every (benchmark, manager size): direct-answer baseline, cold-start
SFT, and every GRPO cell, with the gain over baseline and over cold start in
percentage points. The standard error column uses the conservative p = 0.5
bound, so a difference smaller than about 2 * sqrt(2) * SE between two
independent scores should not be read as a real change.

Usage:
  python scripts/summarize_main_results.py [results/predecessor_scaling/main_results.json]
"""
from __future__ import annotations

import json
import math
import sys


def se_pp(n: int, p: float = 0.5) -> float:
    return 100 * math.sqrt(p * (1 - p) / n)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/predecessor_scaling/main_results.json"
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    results.pop("_note", None)

    print("| Benchmark | n | SE (pp) | Manager | Baseline | Cold start | GRPO cell | GRPO | Δ vs base | Δ vs cold |")
    print("|---|---:|---:|---|---:|---:|---|---:|---:|---:|")
    for bench, d in results.items():
        n = d["n"]
        for size in ("4B", "8B", "9B"):
            if size not in d["base"]:
                continue
            base, cold = d["base"][size], d["cold"][size]
            for family in ("Generic", "Specific"):
                for gen, score in d[family].get(size, {}).items():
                    print(
                        f"| {bench} | {n} | {se_pp(n):.1f} | {size} | {100 * base:.1f} | {100 * cold:.1f} | "
                        f"{family} {gen} | {100 * score:.1f} | {100 * (score - base):+.1f} | "
                        f"{100 * (score - cold):+.1f} |"
                    )


if __name__ == "__main__":
    main()
