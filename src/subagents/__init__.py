"""Subagent package.

``SUBAGENT_KINDS`` names the three frozen sub-agents (the paper's Extractor,
Reasoner and Verifier; "advisor" in older docs and report keys). Schema objects
are loaded lazily so lightweight utilities such as JSONL prompt export do not
require pydantic at import time.
"""
from typing import Tuple

SUBAGENT_KINDS: Tuple[str, ...] = ("extractor", "reasoner", "verifier")

_SCHEMA_EXPORTS = {
    "AgentKind",
    "ExtractorOutput",
    "ReasonerOutput",
    "VerifierOutput",
    "SCHEMA_REGISTRY",
}


def __getattr__(name):
    if name in _SCHEMA_EXPORTS:
        from . import schemas
        return getattr(schemas, name)
    raise AttributeError(name)


__all__ = ["SUBAGENT_KINDS", *sorted(_SCHEMA_EXPORTS)]
