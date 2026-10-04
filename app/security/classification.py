"""Query classification: the routing decision that happens *before* retrieval.

Backend-only, deterministic, and ordered by security priority:

1. prompt-injection / system-prompt-extraction attempt -> ``INJECTION``
2. legitimate Ahmed/portfolio question                 -> ``IN_SCOPE``
3. harmless anything else                              -> ``OFF_TOPIC``

Steps 2 and 3 never reach the LLM; they are answered with fixed text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from app.config import DEFAULT_TOPIC_KEYWORDS
from app.models.enums import QueryClassification
from app.security.injection import InjectionDetector, InjectionMatch, InjectionVerdict
from app.services.answering.intents import QUESTION_FORMS, SUBJECT_REFERENCES

#: Words that can name the subject or shape a question. Matched as a set of
#: candidate words and then looked up, rather than as one alternation, so the
#: result can name *which* word decided the routing -- an in-scope decision is
#: worth being able to explain after the fact.
_SUBJECT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")
_FORM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z]+")

#: Greetings and pleasantries: harmless, but not portfolio questions.
_SMALLTALK: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:hi|hey|hello|yo|sup|good\s+(?:morning|afternoon|evening|day)|"
    r"how\s+are\s+you|thanks|thank\s+you|thx|ok|okay|cool|nice|bye|goodbye|"
    r"see\s+ya|good\s+night|cheers|👍|🙏)\b"
    r"(?:\s+(?:there|ahmed|everyone|all|man|friend))?"
    r"[\s!.,?🙂👋]*$",
    re.IGNORECASE,
)


def _compile_keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str] | None:
    """Build one alternation regex; longest keywords first for determinism."""
    cleaned = sorted({kw.strip().lower() for kw in keywords if kw.strip()}, key=len, reverse=True)
    if not cleaned:
        return None
    alternation = "|".join(re.escape(kw) for kw in cleaned)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """The routing decision plus the evidence behind it."""

    classification: QueryClassification
    reason: str
    matches: tuple[InjectionMatch, ...] = ()
    suppressed: tuple[InjectionMatch, ...] = ()
    topics: tuple[str, ...] = ()
    is_smalltalk: bool = False

    @property
    def is_injection(self) -> bool:
        """True when the message was classified as a security attempt."""
        return self.classification is QueryClassification.INJECTION

    @property
    def is_in_scope(self) -> bool:
        """True when the message is a legitimate portfolio question."""
        return self.classification is QueryClassification.IN_SCOPE

    @property
    def is_off_topic(self) -> bool:
        """True when the message is harmless but out of scope."""
        return self.classification is QueryClassification.OFF_TOPIC


class QueryClassifier:
    """Security-first classifier for inbound messages."""

    def __init__(
        self,
        detector: InjectionDetector | None = None,
        topic_keywords: Sequence[str] | None = None,
    ) -> None:
        self._detector = detector or InjectionDetector()
        keywords = tuple(topic_keywords) if topic_keywords is not None else DEFAULT_TOPIC_KEYWORDS
        self._keywords: tuple[str, ...] = tuple(
            sorted({kw.strip().lower() for kw in keywords if kw.strip()})
        )
        self._pattern = _compile_keyword_pattern(self._keywords)

    @property
    def topic_keywords(self) -> tuple[str, ...]:
        """The configured in-scope vocabulary."""
        return self._keywords

    @property
    def detector(self) -> InjectionDetector:
        """The injection detector shared with the rest of the pipeline."""
        return self._detector

    def match_topics(self, text: str) -> tuple[str, ...]:
        """Return the in-scope keywords present in ``text``, in match order."""
        if self._pattern is None:
            return ()
        return tuple(match.group(0).lower() for match in self._pattern.finditer(text))

    def classify(self, text: str) -> ClassificationResult:
        """Classify ``text`` and return the routing decision with its evidence."""
        cleaned = (text or "").strip()
        if not cleaned:
            return ClassificationResult(
                classification=QueryClassification.OFF_TOPIC,
                reason="empty message",
            )

        verdict: InjectionVerdict = self._detector.detect(cleaned)
        if verdict.is_injection:
            return ClassificationResult(
                classification=QueryClassification.INJECTION,
                reason=verdict.reason,
                matches=verdict.matches,
                suppressed=verdict.suppressed,
            )

        if _SMALLTALK.match(cleaned):
            return ClassificationResult(
                classification=QueryClassification.OFF_TOPIC,
                reason="greeting/small talk",
                suppressed=verdict.suppressed,
                is_smalltalk=True,
            )

        topics = self.match_topics(cleaned)
        if topics:
            return ClassificationResult(
                classification=QueryClassification.IN_SCOPE,
                reason=f"matched in-scope keyword(s): {', '.join(topics)}",
                suppressed=verdict.suppressed,
                topics=topics,
            )

        # Second gate, second chance. The keyword list is a closed vocabulary, and
        # a closed vocabulary cannot answer "Who r u?" -- "who are you" is listed,
        # "who r u" is not, and neither does "Where does he study?" nor "What
        # diploma is he pursuing?". Those questions were answered "I can only
        # answer questions about Ahmed's portfolio" by an assistant whose entire
        # subject is that portfolio.
        #
        # The test is *structural* rather than lexical: the message must be a
        # question or request that names its subject. "Who r u?" names you and
        # asks a question; "Recommend a restaurant near me" names nobody and is
        # therefore still off-topic. No word list is consulted, which is what keeps
        # this from becoming the keyword blacklist the design forbids.
        #
        # Note the asymmetry with injection: this runs strictly *after* the
        # detector has had its say, so widening what counts as in-scope cannot
        # widen what counts as an attack.
        subject, form = self._subject_and_form(cleaned)
        if subject and form:
            return ClassificationResult(
                classification=QueryClassification.IN_SCOPE,
                reason=f"question about the subject (subject={subject}, form={form})",
                suppressed=verdict.suppressed,
                topics=(),
            )

        return ClassificationResult(
            classification=QueryClassification.OFF_TOPIC,
            reason="no in-scope topic keyword matched",
            suppressed=verdict.suppressed,
        )

    @staticmethod
    def _subject_and_form(text: str) -> tuple[str | None, str | None]:
        """The subject the message is about, and its question form.

        Returns:
            ``(subject, form)``, either of which may be ``None``. Both are needed
            for a message to be in scope: naming the subject without asking
            anything ("Ahmed" alone) is not a question, and asking something
            without naming the subject ("What is the capital of France?") is a
            question about the world.
        """
        lowered = text.lower()
        subject = next(
            (word for word in _SUBJECT_PATTERN.findall(lowered) if word in SUBJECT_REFERENCES),
            None,
        )
        form = next(
            (word for word in _FORM_PATTERN.findall(lowered) if word in QUESTION_FORMS),
            None,
        )
        return subject, form
