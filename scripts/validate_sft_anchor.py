#!/usr/bin/env python3
"""Validate SFT-anchor tokenization without loading manager model weights."""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

try:
    from trl.chat_template_utils import add_response_schema
except Exception:
    add_response_schema = None

from src.manager.marginal_value import _tool_schemas
from src.manager.routing_anchor import ANCHOR_MODES, build_anchor_features
from src.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="SFT checkpoint or base model tokenizer path")
    parser.add_argument("--anchor_jsonl", required=True)
    parser.add_argument("--mode", choices=ANCHOR_MODES, required=True)
    parser.add_argument("--binding_mode", choices=["environment", "argument"], default="environment")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    args = parser.parse_args()

    rows = read_jsonl(args.anchor_jsonl)
    if not rows:
        raise SystemExit(f"No rows in {args.anchor_jsonl}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if add_response_schema is not None:
        try:
            tokenizer = add_response_schema(tokenizer)
        except Exception:
            pass

    _, stats = build_anchor_features(
        rows=rows,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        mode=args.mode,
        tools=_tool_schemas(args.binding_mode),
    )
    print(json.dumps({"mode": args.mode, **stats}, indent=2, sort_keys=True))
    if int(stats["n_kept"]) <= 0:
        raise SystemExit("ERROR: no supervised anchor targets remain after tokenization")
    if int(stats["n_dropped_no_target"]) > 0:
        print(
            "WARNING: some rows lost every supervised token after truncation; "
            "inspect them or increase --max_seq_len before the expensive run."
        )


if __name__ == "__main__":
    main()
