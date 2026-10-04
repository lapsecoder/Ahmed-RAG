"""Enumerations describing how a query was routed."""

from __future__ import annotations

from enum import StrEnum


class QueryClassification(StrEnum):
    """Security/scope verdict for an inbound user message.

    The classifier deliberately has exactly three outcomes, because each one has
    a different and fully deterministic response policy.
    """

    IN_SCOPE = "in_scope"
    """A legitimate question about Ahmed / the portfolio knowledge base."""

    OFF_TOPIC = "off_topic"
    """Harmless but unrelated: answered with a fixed message, never sent to the LLM."""

    INJECTION = "injection"
    """Prompt-injection or system-prompt-extraction attempt: refused, never executed."""


class ChatOutcome(StrEnum):
    """What the chat pipeline actually did, for observability and tests."""

    ANSWERED = "answered"
    """Retrieval succeeded and the local LLM produced a validated answer."""

    NO_CONTEXT = "no_context"
    """Nothing scored above the similarity threshold: answered without the LLM."""

    OFF_TOPIC = "off_topic"
    """Out-of-scope question: answered deterministically."""

    BLOCKED_INJECTION = "blocked_injection"
    """Security attempt: refused deterministically."""

    BLOCKED_OUTPUT = "blocked_output"
    """The model tried to leak protected material: response replaced."""
