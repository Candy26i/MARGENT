"""Chat-template helpers shared by manager SFT, marginal-value collection, eval
and the sub-agent SFT/runtime.

Qwen3.5's chat template iterates ``tool_call.arguments`` with ``|items``, so the
arguments must be a mapping. The vLLM API and the saved SFT JSONL carry them as
a JSON string, so we convert on a deep copy and leave the caller's messages
untouched. ``tool_call_message`` builds such a turn, ``mask_prefix_len`` finds
the prompt/response boundary for SFT label masking. This module is stdlib-only
so tests can import it without torch.
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


def mask_prefix_len(prompt_ids: List[int], full_ids: List[int]) -> int:
    """Length of the common token prefix between the prompt-only render and the
    full (prompt+response) render.

    Using len(prompt_ids) directly is WRONG for templates where the generation
    prompt is not a strict prefix of the full render — e.g. Qwen3 with
    enable_thinking=False appends an empty <think></think> block to the
    generation prompt that does not appear before the assistant content in the
    full render. That off-by-N would mask the first response tokens.
    """
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    return i


def tool_call_message(
    tool_name: str, args: Dict[str, Any], call_id: str, content: str = ""
) -> Dict[str, Any]:
    """An assistant turn that calls ``tool_name`` with ``args`` (JSON-encoded,
    as the saved SFT JSONL carries them). ``content`` keeps the assistant's own
    text (the DRAFT_ANSWER_ line) in the history, as build_marginal_sft writes it."""
    return {
        "role": "assistant",
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
