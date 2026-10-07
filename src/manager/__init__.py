"""Manager package. Only the prompt protocol is re-exported so ``import
src.manager`` stays torch-free; import marginal_value / sft / loading explicitly."""
from .prompt import (
    build_manager_system_prompt,
    build_manager_user_message,
    manager_tool_schemas,
    parse_final_answer,
    ANSWER_LASTLINE_RE_FOR_KEYS,
)

__all__ = [
    "build_manager_system_prompt",
    "build_manager_user_message",
    "manager_tool_schemas",
    "parse_final_answer",
    "ANSWER_LASTLINE_RE_FOR_KEYS",
]
