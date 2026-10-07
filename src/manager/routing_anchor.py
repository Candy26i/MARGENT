"""Tokenization helpers for SFT-anchored manager GRPO experiments.

Two auxiliary objectives are supported without changing the manager's runtime
protocol:

``full``
    Replay the complete marginal-SFT assistant turn.  This anchors both the
    current draft and the subsequent routing/protocol behavior.

``route_only``
    Mask the prompt *and* the current ``DRAFT_ANSWER_*``.  Loss starts only at
    the suffix that realizes the routing action: a native tool call for CALL,
    or ``ANSWER_*`` for COMMIT.  The answer draft itself therefore receives no
    auxiliary SFT gradient.

The latter works because the existing manager protocol makes a draft before
deciding whether to call a tool or commit; no new ROUTE_* tokens are needed.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


ANCHOR_MODES = ("full", "route_only")
_DRAFT_RE = re.compile(r"DRAFT_ANSWER_[A-Za-z0-9_]+")


def _normalize_tool_call_arguments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Decode JSON-string tool-call arguments into dicts.

    Qwen3/Qwen3.5 chat templates iterate over ``arguments`` as a mapping, while
    OpenAI-style messages store them as a JSON string.
    """
    normalized = []
    for message in messages:
        new_message = dict(message)
        calls = message.get("tool_calls")
        if calls:
            new_calls = []
            for call in calls:
                new_call = dict(call)
                function = dict(new_call.get("function", {}))
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments or "{}")
                new_call["function"] = function
                new_calls.append(new_call)
            new_message["tool_calls"] = new_calls
        normalized.append(new_message)
    return normalized


def _render_chat(tokenizer: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool, tools=None) -> str:
    messages = _normalize_tool_call_arguments(messages)
    extra = {"tools": tools} if tools else {}
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
            **extra,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **extra,
        )


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def _normalize_response(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, dict):
        return [response]
    if isinstance(response, str):
        return [{"role": "assistant", "content": response}]
    if isinstance(response, list):
        return list(response)
    raise TypeError(f"Unsupported manager SFT response type: {type(response).__name__}")


def _draft_prefix_message(response_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return an assistant message containing only the already-made draft.

    The returned message intentionally omits ``tool_calls``.  Rendering this
    prefix and comparing it with the full target locates the token boundary at
    which CALL vs COMMIT begins in the native chat template.
    """
    if len(response_messages) != 1 or response_messages[0].get("role") != "assistant":
        raise ValueError("route_only anchor expects exactly one assistant response message")
    content = str(response_messages[0].get("content") or "")
    match = _DRAFT_RE.search(content)
    if match is None:
        raise ValueError(
            "route_only anchor requires DRAFT_ANSWER_* in every target response; "
            f"got content={content[:120]!r}"
        )
    return {"role": "assistant", "content": match.group(0)}


def _boundary_after_text_with_offsets(
    tokenizer: Any,
    full_text: str,
    needle: str,
) -> Optional[int]:
    """Find a conservative token boundary after ``needle`` when offsets exist.

    Fast tokenizers can merge the last draft character with following
    whitespace/markup.  Masking every token whose span touches the draft makes
    the route-only claim stronger: no token overlapping DRAFT_ANSWER_* is ever
    supervised.  Slow/custom tokenizers fall back to rendered-prefix matching.
    """
    char_start = full_text.rfind(needle)
    if char_start < 0:
        return None
    char_end = char_start + len(needle)
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        return None
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    for index, pair in enumerate(offsets):
        if pair is None or len(pair) != 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        if start >= char_end and end > start:
            return index
    return len(offsets)


def tokenize_anchor_row(
    row: Dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
    mode: str,
    tools=None,
) -> Tuple[Optional[Dict[str, List[int]]], Dict[str, int]]:
    """Tokenize one marginal-SFT row and build the requested label mask.

    Returns ``(features, stats)``.  ``features`` is ``None`` when truncation
    removes every supervised token, allowing callers to drop the row rather
    than silently train on an all--100 label tensor.
    """
    if mode not in ANCHOR_MODES:
        raise ValueError(f"Unknown SFT anchor mode {mode!r}; expected one of {ANCHOR_MODES}")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    prompt_messages = list(row["prompt"])
    response_messages = _normalize_response(row["response"])
    full_text = _render_chat(
        tokenizer,
        prompt_messages + response_messages,
        add_generation_prompt=False,
        tools=tools,
    )
    eos = tokenizer.eos_token or ""
    if eos and not full_text.rstrip().endswith(eos):
        full_text += eos

    full = tokenizer(full_text, add_special_tokens=False)
    full_ids = list(full["input_ids"])
    input_ids = full_ids[:max_seq_len]
    attention_mask = list(full["attention_mask"][:max_seq_len])

    prompt_text = _render_chat(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        tools=tools,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_boundary = _common_prefix_len(prompt_ids, full_ids)

    if mode == "full":
        label_boundary = prompt_boundary
    else:
        decision_type = str(row.get("decision_type") or "")
        if decision_type not in {"call", "commit", "commit_after_call"}:
            raise ValueError(
                "route_only anchor requires decision_type in "
                "{call, commit, commit_after_call}; "
                f"got {decision_type!r}"
            )
        draft_message = _draft_prefix_message(response_messages)
        draft_prefix_text = _render_chat(
            tokenizer,
            prompt_messages + [draft_message],
            add_generation_prompt=False,
            tools=tools,
        )
        draft_prefix_ids = tokenizer(draft_prefix_text, add_special_tokens=False)["input_ids"]
        label_boundary = _common_prefix_len(draft_prefix_ids, full_ids)
        offset_boundary = _boundary_after_text_with_offsets(
            tokenizer,
            full_text,
            str(draft_message["content"]),
        )
        if offset_boundary is not None:
            label_boundary = max(label_boundary, offset_boundary)
        if label_boundary <= prompt_boundary:
            raise ValueError(
                "The active chat template does not preserve the assistant draft before the "
                "routing suffix, so route_only masking would be invalid. Use mode=full or "
                "a prefix-preserving tool template."
            )

    label_boundary = min(label_boundary, max_seq_len)
    labels = ([-100] * label_boundary) + input_ids[label_boundary:]
    labels = labels[: len(input_ids)]
    supervised_tokens = sum(int(x != -100) for x in labels)
    stats = {
        "total_tokens": len(input_ids),
        "prompt_boundary": min(prompt_boundary, max_seq_len),
        "label_boundary": label_boundary,
        "supervised_tokens": supervised_tokens,
    }
    if supervised_tokens == 0:
        return None, stats
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }, stats


def build_anchor_features(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    mode: str,
    tools=None,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, float]]:
    """Tokenize anchor rows and summarize masking/truncation diagnostics."""
    features: List[Dict[str, List[int]]] = []
    total_supervised = 0
    total_tokens = 0
    dropped = 0
    for row in rows:
        feature, stats = tokenize_anchor_row(
            row=row,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            tools=tools,
        )
        total_tokens += stats["total_tokens"]
        total_supervised += stats["supervised_tokens"]
        if feature is None:
            dropped += 1
        else:
            features.append(feature)

    n_kept = len(features)
    return features, {
        "n_rows": float(len(rows)),
        "n_kept": float(n_kept),
        "n_dropped_no_target": float(dropped),
        "mean_tokens": total_tokens / max(1, len(rows)),
        "mean_supervised_tokens": total_supervised / max(1, n_kept),
    }
