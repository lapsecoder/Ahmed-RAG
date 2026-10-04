"""Fixed, deterministic answers for every path that must not reach the LLM.

Keeping them in one module makes the behaviour auditable: no randomness, no
model call, no dependence on retrieved content.
"""

from __future__ import annotations

from typing import Final

INJECTION_REFUSAL: Final[str] = (
    "I can't help with that. I only answer questions about Ahmed's portfolio using "
    "information from my knowledge base, and I don't change my instructions on request."
)

OFF_TOPIC_RESPONSE: Final[str] = (
    "I can only answer questions about Ahmed's portfolio, experience, projects and skills. "
    "Ask me something about those and I'll answer from my knowledge base."
)

SMALLTALK_RESPONSE: Final[str] = (
    "Hello. I'm Ahmed's portfolio assistant. Ask me about his experience, projects or skills."
)

NO_CONTEXT_RESPONSE: Final[str] = (
    "I don't have that information in my knowledge base. "
    "Try asking about Ahmed's experience, projects, skills or education."
)

LLM_UNAVAILABLE_RESPONSE: Final[str] = (
    "The local language model is unavailable right now, so I can't generate an answer. "
    "The service is free and local: start Ollama with `ollama serve` and pull "
    "the configured model, then try again."
)


def off_topic_response(*, is_smalltalk: bool) -> str:
    """Deterministic reply for the out-of-scope path."""
    return SMALLTALK_RESPONSE if is_smalltalk else OFF_TOPIC_RESPONSE
