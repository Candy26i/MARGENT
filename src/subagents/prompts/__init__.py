"""Sub-agent prompts: teacher synthesis prompts and the runtime system prompts."""
from .extractor import build_extractor_synth_prompt
from .reasoner import build_reasoner_synth_prompt
from .verifier import build_verifier_synth_prompt
from .runtime_prompts import (
    EXTRACTOR_RUNTIME_SYSTEM,
    REASONER_RUNTIME_SYSTEM,
    VERIFIER_RUNTIME_SYSTEM,
    build_extractor_runtime_user,
    build_reasoner_runtime_user,
    build_verifier_runtime_user,
)

__all__ = [
    "build_extractor_synth_prompt",
    "build_reasoner_synth_prompt",
    "build_verifier_synth_prompt",
    "EXTRACTOR_RUNTIME_SYSTEM",
    "REASONER_RUNTIME_SYSTEM",
    "VERIFIER_RUNTIME_SYSTEM",
    "build_extractor_runtime_user",
    "build_reasoner_runtime_user",
    "build_verifier_runtime_user",
]
