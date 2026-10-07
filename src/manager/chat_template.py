"""Chat-template helpers shared by manager SFT, marginal-value collection and eval.

Qwen3.5's chat template iterates ``tool_call.arguments`` with ``|items``, so the
arguments must be a mapping. The vLLM API and the saved SFT JSONL carry them as
a JSON string, so we convert on a deep copy and leave the caller's messages
untouched. This module is stdlib-only so tests can import it without torch.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional


def normalize_tool_call_arguments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Decode JSON-string ``tool_calls[].function.arguments`` into dicts.

    Works on a deep copy; the caller's messages are never modified. Invalid
    JSON, or a payload that is not a mapping, becomes ``{}``. Only messages go
    through here — the ``tools`` schema list is never touched.
    """
    out = copy.deepcopy(messages)
    for m in out:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    fn["arguments"] = json.loads(a)
                except Exception:
                    fn["arguments"] = {}
            if not isinstance(fn.get("arguments"), dict):
                fn["arguments"] = {}
            if fn is not tc:
                tc.setdefault("name", fn.get("name"))
                tc["arguments"] = fn["arguments"]
    return out


def render_chat(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render ``messages`` with ``apply_chat_template(enable_thinking=False)``.

    Templates that do not accept ``enable_thinking`` raise TypeError; we then
    retry without it. ``tools`` is forwarded as-is when given.
    """
    messages = normalize_tool_call_arguments(messages)
    extra = {"tools": tools} if tools else {}
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False, **extra,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            **extra,
        )
