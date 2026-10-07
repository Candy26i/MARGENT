"""Manager system prompt + final-answer parsing.

The manager is a deliberation orchestrator with three cognitive specialist tools:
  - extractor_tool: information extraction
  - reasoner_tool: structured reasoning
  - verifier_tool: domain audit and error detection
``manager_tool_schemas`` gives their native tool-calling schemas.

Draft-conditioned routing policy:
  - 0 to 3 tool calls allowed. Each tool may be called at most once.
  - Before every routing or stopping decision, output DRAFT_ANSWER_<TOKEN>.
  - Stop calling tools when further help is unlikely to improve the draft answer.
  - Final answer ends with exactly one line: ANSWER_<TOKEN>
  - Manager MUST NOT emit tool-call JSON or XML in plain text content.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


# Regex for final ANSWER_ on last line
ANSWER_LASTLINE_RE_FOR_KEYS = re.compile(
    r"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_([A-Za-z0-9_]+)\b[^\w]*$",
    re.IGNORECASE,
)

# Regex for DRAFT_ANSWER_ anywhere in text (intermediate candidate answers)
DRAFT_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*DRAFT_ANSWER_([A-Za-z0-9_]+)\b",
    re.IGNORECASE,
)


def build_manager_system_prompt(
    label_keys: List[str],
    task_description: str = "",
) -> str:
    """Build the manager's system prompt with draft-conditioned routing.

    Args:
        label_keys: choice keys for the current task (e.g. ["A","B","C","D"]).
        task_description: optional one-liner describing the task domain.
    """
    answer_lines = "\n".join(f"  ANSWER_{_label_to_token(k)}" for k in label_keys)
    draft_lines  = "\n".join(f"  DRAFT_ANSWER_{_label_to_token(k)}" for k in label_keys)
    desc = task_description or "You are a manager agent solving a multiple-choice question."
    return (
        desc + "\n\n"
        "You have THREE cognitive specialist tools:\n"
        "  - extractor_tool: extracts key signals and structures relevant facts from the question.\n"
        "  - reasoner_tool: produces a structured reasoning scaffold (sub-questions, per-choice analysis).\n"
        "  - verifier_tool: identifies domain principles and audits reasoning for errors. "
        "Pass your current draft answer key via the current_draft argument "
        "(e.g. current_draft=\"B\") so it audits that specific hypothesis.\n\n"
        "Deliberation policy:\n"
        "  - You may call 0 to 3 tools total. Each tool may be used at most once.\n"
        "  - Before every call-or-stop decision, state your current best answer on a new line:\n"
        + draft_lines + "\n"
        "  - Then decide: only call another tool if it might change your draft answer.\n"
        "  - Stop when additional tools are unlikely to improve your answer.\n"
        "  - Reserve all three tools for genuinely hard cases where each adds new signal.\n\n"
        "Output rules:\n"
        "  - Use the native tool-calling interface. Do NOT write tool calls as text, XML, or JSON.\n"
        "  - In a turn where you call a tool, output DRAFT_ANSWER_ but NOT the final ANSWER_.\n"
        "  - If you answer without a tool, output DRAFT_ANSWER_ immediately before the final ANSWER_.\n"
        "  - When you are ready to submit your final answer (no more tools), end with exactly:\n"
        + answer_lines + "\n"
        "  - Brief reasoning above the ANSWER_ line is allowed; nothing after it.\n"
        "  - Do not output <think> tags.\n"
    )


def build_manager_user_message(
    example_id: int,
    question: str,
    context: str,
    choices: Dict[str, str],
    binding_mode: str = "environment",
) -> str:
    lines = [f"Example ID: {example_id}", "", f"Question:\n{question}", ""]
    if choices:
        choices_block = "Choices:\n" + "\n".join(f"  {k}. {v}" for k, v in choices.items())
        lines.append(choices_block)
        lines.append("")
    if context:
        lines.append(f"Context:\n{context}")
        lines.append("")
    if binding_mode == "argument":
        lines.append(
            "If you call a tool, pass the current Example ID as the example_id argument. "
            "For verifier_tool, also pass your current draft answer key as current_draft."
        )
    else:
        lines.append(
            "If you call a tool, the current example is already bound — no example_id is needed. "
            "For verifier_tool, pass your current draft answer key as current_draft."
        )
    lines.append("")
    lines.append("If you do not call any tool, answer directly.")
    return "\n".join(lines)


# Tool descriptions. "training" is the wording rendered by build_marginal_sft and
# train_manager_sft; "evaluation" is the longer wording the eval stages used in the
# released runs. Both are kept so evaluations stay comparable with the paper.
_TOOL_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "training": {
        "example_id": "The current example ID.",
        "current_draft": "The current draft answer key.",
        "extractor": "Extract decision-relevant factual signals.",
        "reasoner": "Produce a structured reasoning scaffold.",
        "verifier": "Audit the current draft for errors.",
    },
    "evaluation": {
        "example_id": "The current example ID from the user message.",
        "current_draft": 'Your current draft answer key (e.g. "B") to audit.',
        "extractor": "Extract decision-relevant factual signals from the question and context.",
        "reasoner": "Produce a structured reasoning scaffold for the choices.",
        "verifier": (
            "Identify relevant domain principles and audit the reasoning for logical or "
            "computational errors. Pass your current draft answer via current_draft."
        ),
    },
}


def manager_tool_schemas(binding_mode: str, descriptions: str = "training") -> List[Dict[str, Any]]:
    """Native tool-calling schemas for the three tools.

    ``binding_mode`` "argument" adds a required ``example_id``; the verifier
    always takes ``current_draft``. ``descriptions`` selects the wording:
    "training" for build_marginal_sft / train_manager_sft, "evaluation" for the
    eval stages (see ``_TOOL_DESCRIPTIONS``).
    """
    text = _TOOL_DESCRIPTIONS[descriptions]
    required = ["example_id"] if binding_mode == "argument" else []
    properties: Dict[str, Any] = {}
    if binding_mode == "argument":
        properties["example_id"] = {"type": "integer", "description": text["example_id"]}
    verifier_properties = dict(properties)
    verifier_properties["current_draft"] = {"type": "string", "description": text["current_draft"]}

    def tool(name: str, description: str, props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    return [
        tool("extractor_tool", text["extractor"], properties),
        tool("reasoner_tool", text["reasoner"], properties),
        tool("verifier_tool", text["verifier"], verifier_properties),
    ]


def parse_final_answer(text: str, choice_keys: List[str]) -> Optional[str]:
    """Parse the final ANSWER_<TOKEN> line and map to a canonical choice key."""
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return None
    m = ANSWER_LASTLINE_RE_FOR_KEYS.match(lines[-1])
    if not m:
        return None
    token = m.group(1).upper()
    for k in choice_keys:
        if _label_to_token(k) == token:
            return k
    return None


def parse_draft_answer(text: str, choice_keys: List[str]) -> Optional[str]:
    """Parse the LAST DRAFT_ANSWER_<TOKEN> from an assistant turn.

    Returns the most recent draft choice key, or None if not present.
    Used by eval_manager_tools and build_marginal_sft to track candidate transitions.
    """
    if not text:
        return None
    matches = DRAFT_ANSWER_RE.findall(str(text))
    if not matches:
        return None
    token = matches[-1].upper()
    for k in choice_keys:
        if _label_to_token(k) == token:
            return k
    return None
