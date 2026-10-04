"""The RAG orchestration pipeline.

    validate -> classify -> [injection | off-topic | retrieve] -> compose -> vet

The pipeline is unchanged in shape from the generated original, but the final
generation step is gone. Answers are assembled by
:class:`~app.services.answering.composer.AnswerComposer`, which only ever copies
sentences out of retrieved, detector-screened knowledge-base chunks. Two
consequences follow, and both are enforced here:

* an injection attempt never reaches retrieval, and
* an out-of-scope question, or one with no supporting evidence, never produces an
  answer -- it produces the standard no-information response.

There is no model call to make hallucination possible, and no prompt to
misinterpret. ``llm_used`` is therefore always ``False``; the field is retained
because it is part of the public response schema.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.logging import get_logger
from app.models.chat import ChatResult, SourceRef
from app.models.enums import ChatOutcome, QueryClassification
from app.models.retrieval import RetrievalResult
from app.security.classification import QueryClassifier
from app.security.context import build_context
from app.security.injection import InjectionDetector
from app.security.output_validator import OutputValidator
from app.services.answering.composer import AnswerComposer
from app.services.responses import (
    INJECTION_REFUSAL,
    NO_CONTEXT_RESPONSE,
    off_topic_response,
)
from app.services.retriever import Retriever
from app.services.text import content_terms

logger = get_logger("services.chat")


class ChatService:
    """Coordinates classification, retrieval, composition and output vetting."""

    def __init__(
        self,
        retriever: Retriever,
        composer: AnswerComposer,
        classifier: QueryClassifier,
        output_validator: OutputValidator,
        *,
        detector: InjectionDetector | None = None,
        no_context_message: str = NO_CONTEXT_RESPONSE,
        injection_message: str = INJECTION_REFUSAL,
    ) -> None:
        """
        Args:
            retriever: Dense+lexical retriever producing candidate chunks.
            composer: Assembles a grounded answer from screened context.
            classifier: Decides injection, off-topic or in-scope.
            output_validator: Final sanity net over the composed text.
            detector: Injection detector; the classifier's own instance by default.
            no_context_message: Answer used when no evidence supports a reply.
            injection_message: Answer used when an injection attempt is blocked.
        """
        self._retriever = retriever
        self._composer = composer
        self._classifier = classifier
        self._output_validator = output_validator
        self._detector = detector or classifier.detector
        self._no_context_message = no_context_message
        self._injection_message = injection_message

    @property
    def retriever(self) -> Retriever:
        """The configured retriever."""
        return self._retriever

    @property
    def composer(self) -> AnswerComposer:
        """The deterministic answer composer."""
        return self._composer

    def chat(self, message: str) -> ChatResult:
        """Run the full pipeline for one user message.

        Args:
            message: The raw user message (already length-validated by the API).

        Returns:
            A :class:`ChatResult`. ``llm_used`` is always ``False`` because no
            language model participates in answering.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        decision = self._classifier.classify(message)

        if decision.is_injection:
            logger.warning("blocked injection attempt (rules=%s)", decision.matches[0].rule_id)
            return ChatResult(
                response=self._injection_message,
                classification=QueryClassification.INJECTION,
                outcome=ChatOutcome.BLOCKED_INJECTION,
                injection_rule_ids=tuple(match.rule_id for match in decision.matches),
            )

        if decision.is_off_topic:
            logger.debug("routing off-topic message to a deterministic answer")
            return ChatResult(
                response=off_topic_response(is_smalltalk=decision.is_smalltalk),
                classification=QueryClassification.OFF_TOPIC,
                outcome=ChatOutcome.OFF_TOPIC,
            )

        # Retrieval is confined to the routed documents when the question's answer
        # is one of them whole -- a list question, or an overview request. The
        # composer can only ever quote those documents, so nothing outside the
        # scope was quotable anyway, and confining it keeps the context compact.
        results = self._retriever.retrieve(
            message, restrict_to=self._composer.document_scope(message)
        )

        if not results:
            results = self._scoped_fallback(message)
        elif not self._composer.answered_documents_retrieved(
            message, (result.chunk.source_file for result in results)
        ):
            # Retrieval returned chunks, just not from the document routing says
            # should answer this. That is a quieter failure than returning
            # nothing: everything looks healthy, and the question is declined
            # because the one relevant document was never a candidate. Lexical
            # search could not have reached it either -- it does not contain the
            # question's words -- so retrying within the routed documents is the
            # only thing that can, and routing is the evidence for that scope.
            logger.info(
                "retrieval missed the routed document for %r; retrying within the "
                "documents routing named",
                message[:60],
            )
            results = self._scoped_fallback(message) or results

        if not results:
            logger.info("no context above threshold for in-scope query")
            return ChatResult(
                response=self._no_context_message,
                classification=QueryClassification.IN_SCOPE,
                outcome=ChatOutcome.NO_CONTEXT,
            )

        blocks = build_context(
            [(result.chunk, result.similarity) for result in results],
            detector=self._detector,
        )
        composed = self._composer.compose(message, blocks)

        if composed is None:
            # Evidence was retrieved but none of it speaks to the question. Say so
            # rather than quoting the nearest unrelated chunk.
            logger.info("retrieved %d block(s) but none supported an answer", len(blocks))
            return ChatResult(
                response=self._no_context_message,
                classification=QueryClassification.IN_SCOPE,
                outcome=ChatOutcome.NO_CONTEXT,
            )

        # The composer only copies screened context, so this should never fire.
        # It stays as a last line of defence against a regression that would
        # otherwise let unvetted text reach the caller.
        validation = self._output_validator.validate(composed.text)
        if not validation.is_valid:
            logger.warning("rejected composed output (rules=%s)", ", ".join(validation.rule_ids))
            return ChatResult(
                response=validation.text,
                classification=QueryClassification.IN_SCOPE,
                outcome=ChatOutcome.BLOCKED_OUTPUT,
                sources=self._sources(results),
                retrieval_scores=self._scores(results),
            )

        # Cite what the answer actually used. Returning all twelve retrieved
        # chunks would claim evidence for statements that are not in the reply,
        # which is the citation equivalent of padding an answer.
        used = self._used(results, composed.citations)
        return ChatResult(
            response=validation.text,
            classification=QueryClassification.IN_SCOPE,
            outcome=ChatOutcome.ANSWERED,
            sources=self._sources(used),
            retrieval_scores=self._scores(used),
        )

    def _scoped_fallback(self, message: str) -> list[RetrievalResult]:
        """Retry a question the retrievers cannot key on, within the documents that own it.

        "Who are you?" is admitted as in-scope and then retrieves nothing, because
        every one of its words is a function word: BM25 has no query terms left
        to match and the dense score for the correct chunk (``About > Identity``,
        ranked first at 0.22) sits far below the 0.35 threshold. The user got
        "I don't have that information" for the one question whose answer the
        knowledge base states outright.

        "What's Ahmed's background?" fails the same way for a different reason:
        it contributes the single content word "background", which no document
        contains. Routing named the right four documents, but with nothing in the
        corpus to match, neither retriever could reach them.

        Both are one condition -- the retrievers have nothing to key on -- and
        when it holds, routing is the only relevance signal available, so it is
        used as the filter: search again within the routed documents. There the
        threshold is no longer what guarantees relevance, the document
        restriction does, so it is relaxed to admit the routed documents' best
        dense match.

        Widening this to "anything that retrieved nothing" was tried and reverted.
        It defeats a guarantee two tests exist to hold: an orthogonal query
        vector must still produce a refusal. The distinction between "the corpus
        has no word for this question" and "the corpus does not discuss this
        subject" is exactly the distinction that matters, and it is visible in
        the vocabulary -- see
        :meth:`~app.services.answering.composer.AnswerComposer.has_lexical_support`.

        Guarded four ways: it runs only when normal retrieval found nothing, only
        when the question has neither content terms nor a content term the corpus
        contains, and only when routing named documents. A question the corpus has
        words for keeps its existing behaviour, including the refusal, so this
        cannot become a way to smuggle in weak evidence. The composer still
        applies its own on-topic and attestation checks to whatever comes back.
        """
        termless = not content_terms(message)
        if not termless and self._composer.has_lexical_support(message):
            return []
        scoped = self._composer.allowed_files_for(message)
        if not scoped:
            return []
        logger.info(
            "no lexical or dense evidence for a %s query; retrying within %s",
            "termless" if termless else "corpus-unknown-word",
            ", ".join(scoped),
        )
        return self._retriever.retrieve(message, restrict_to=scoped, relax_threshold=True)

    @staticmethod
    def _used(
        results: Sequence[RetrievalResult], citations: Sequence[str]
    ) -> list[RetrievalResult]:
        """Narrow retrieved results to those the composer actually quoted."""
        if not citations:
            return list(results)
        cited = set(citations)
        matched = [
            result
            for result in results
            if f"{result.chunk.source_file} :: {result.chunk.section}" in cited
        ]
        # If the citation strings cannot be reconciled with the results, fall back
        # to the full set rather than reporting an answer with no evidence at all.
        return matched or list(results)

    @staticmethod
    def _sources(results: Sequence[RetrievalResult]) -> tuple[SourceRef, ...]:
        return tuple(
            SourceRef(
                chunk_id=result.chunk.chunk_id,
                source_file=result.chunk.source_file,
                section=result.chunk.section,
                similarity=result.similarity,
            )
            for result in results
        )

    @staticmethod
    def _scores(results: Sequence[RetrievalResult]) -> tuple[float, ...]:
        return tuple(result.similarity for result in results)
