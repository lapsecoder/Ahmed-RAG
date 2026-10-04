"""Output validation: catch answers that leak protected material.

Retrieval and generation are both untrusted-input surfaces. Even when the model
behaves, a final inspection keeps three guarantees:

* the prompt scaffolding is never echoed back,
* hidden/system instructions are never disclosed, and
* internal implementation details are never narrated.

A response that trips any rule is discarded and replaced with a fixed, safe
message -- the original text is never forwarded to the client.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.security.prompt_builder import PROMPT_MARKERS

DEFAULT_MAX_ANSWER_CHARS: Final[int] = 8000

#: Phrases that indicate the model is narrating its own instructions.
_LEAK_PATTERNS: Final[tuple[tuple[str, str, re.Pattern[str]], ...]] = (
    (
        "prompt_scaffolding",
        "Echoes the prompt section markers or context delimiters",
        re.compile(
            r"|".join(re.escape(marker) for marker in PROMPT_MARKERS)
            + r"|<<<\s*END_CONTEXT_CHUNK"
            + r"|\[\s*/?\s*(?:SYSTEM|ASSISTANT|USER|INSTRUCTIONS?)\s*\]"
            + r"|<\|im_(?:start|end)\|>"
            + r"|<</?SYS>>",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_disclosure",
        "Discloses the system/developer prompt",
        re.compile(
            r"\b(?:here\s+(?:is|are)|these\s+are)\s+(?:my|the)\s+"
            r"(?:\w+\s+){0,3}?(?:system|developer|hidden|initial|original)?\s*"
            r"(?:prompt|instructions?|rules?|directives?)\b"
            r"|\bmy\s+(?:system|developer|hidden|initial|original|full|exact)\s+"
            r"(?:prompt|instructions?|message)\s+(?:is|are|was|says?|reads?)\b"
            r"|\b(?:my|the)\s+(?:system\s+)?(?:prompt|instructions?)\s*(?:say|says|reads?|states?)\b"
            r"|\bi\s+(?:was|am|have\s+been)\s+(?:told|instructed|programmed|configured)\s+to\b"
            r"|\bi\s+(?:must|have\s+to)\s+(?:follow|obey|comply\s+with)\s+the\s+following\b"
            r"|\bmy\s+(?:instructions|rules|guidelines|directives|programming)\s+(?:are|is)\b"
            r"|\bmy\s+training\s+(?:data\s+)?(?:prompt|data)\b"
            r"|\bprompt\s+engineering\s+note\s*:",
            re.IGNORECASE,
        ),
    ),
    (
        "context_disclosure",
        "Reveals the protected prompt context or scaffolding",
        re.compile(
            r"\bthe\s+(?:untrusted\s+retrieved\s+context|retrieved\s+context|user\s+query)\s+"
            r"(?:is|are|was|reads?|contains?|says?)\b"
            r"|\bcontext\s+window\s+(?:is|contains|includes)\b"
            r"|\b(?:above|below|preceding)\s+(?:the\s+)?(?:user\s+query|system\s+rules|"
            r"retrieved\s+context)\b"
            r"|\bI\s+was\s+given\s+the\s+following\s+(?:context|instructions|document)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal_implementation_disclosure",
        "Narrates internal implementation details",
        re.compile(
            r"\bmy\s+(?:implementation|source\s+code|codebase|pipeline|architecture)\s+"
            r"(?:is|uses|includes|contains)\b"
            r"|\bthe\s+source\s+code\s+(?:of|for)\s+this\s+(?:assistant|service|app\w*)\b"
            r"|\bi\s+(?:am|'m)\s+implemented\s+(?:in|using)\b"
            r"|\bthe\s+(?:prompt|system\s+prompt)\s+template\s+(?:is|reads?)\b"
            r"|\bI\s+(?:was\s+)?(?:built|configured)\s+with\s+the\s+following\s+"
            r"(?:rules|instructions|config\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override_echo",
        "Echoes an attempt to override instructions",
        re.compile(
            r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b"
            r"|\byou\s+are\s+now\s+in\s+(?:developer|jailbreak|dan|god)\s+mode\b"
            r"|\bdeveloper\s+mode\s+(?:enabled|activated)\b"
            r"|\bguardrails?\s+(?:are|have\s+been)\s+(?:disabled|removed|off)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Violation:
    """One reason a response was rejected."""

    rule_id: str
    description: str
    matched_text: str


@dataclass(frozen=True, slots=True)
class OutputValidationResult:
    """Verdict of the output inspection."""

    is_valid: bool
    violations: tuple[Violation, ...] = ()
    text: str = ""
    was_truncated: bool = False

    @property
    def rule_ids(self) -> tuple[str, ...]:
        """Ids of every rule that fired."""
        return tuple(violation.rule_id for violation in self.violations)


class OutputValidator:
    """Deterministic post-generation inspection."""

    def __init__(
        self,
        *,
        fallback_message: str = (
            "I can't share that. I only answer questions about Ahmed's portfolio "
            "using information from my knowledge base."
        ),
        max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
        patterns: Sequence[tuple[str, str, re.Pattern[str]]] = _LEAK_PATTERNS,
    ) -> None:
        self._fallback_message = fallback_message
        self._max_answer_chars = max_answer_chars
        self._patterns = tuple(patterns)

    @property
    def fallback_message(self) -> str:
        """The fixed answer substituted for a rejected response."""
        return self._fallback_message

    def validate(self, text: str | None) -> OutputValidationResult:
        """Inspect a model response.

        Args:
            text: Raw model output, possibly ``None`` or empty.

        Returns:
            :class:`OutputValidationResult`. ``text`` always holds something safe
            to send to the client.
        """
        if text is None or not text.strip():
            return OutputValidationResult(
                is_valid=False,
                violations=(Violation("empty_response", "The model returned no usable text", ""),),
                text=self._fallback_message,
            )

        candidate = text.strip()
        violations: list[Violation] = []
        for rule_id, description, pattern in self._patterns:
            match = pattern.search(candidate)
            if match:
                violations.append(
                    Violation(
                        rule_id=rule_id,
                        description=description,
                        matched_text=match.group(0),
                    )
                )

        truncated = len(candidate) > self._max_answer_chars
        if truncated:
            violations.append(
                Violation(
                    "answer_too_long",
                    f"Response exceeds {self._max_answer_chars} characters",
                    candidate[self._max_answer_chars : self._max_answer_chars + 20],
                )
            )

        if violations:
            return OutputValidationResult(
                is_valid=False,
                violations=tuple(violations),
                text=self._fallback_message,
            )

        if truncated:
            return OutputValidationResult(
                is_valid=True,
                text=candidate[: self._max_answer_chars].rstrip(),
                was_truncated=True,
            )
        return OutputValidationResult(is_valid=True, text=candidate)
