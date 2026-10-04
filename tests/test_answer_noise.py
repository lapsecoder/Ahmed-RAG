"""Regression tests for the two answer-quality defects, R1 and R2.

* **R1** -- "What is resume forge?" led with the live URL instead of the project
  description, because the word "forge" inside the name matched the word inside
  ``resume-forge-one-zeta.vercel.app`` and nothing else in the document.
* **R2** -- answers carried noise: a duplicate restatement of a fact already
  given, a sentence that only made sense after the one before it, and rows of a
  certification's instructor's own credentials.
"""

from __future__ import annotations

import pytest
from app.services.answering.composer import (
    REDUNDANT_MIN_TERMS,
    REDUNDANT_OVERLAP,
    AnswerComposer,
)

from .conftest import make_chunk, make_profile
from .test_regressions import (
    CATEGORIES,
    CORPUS_PADDING,
    PADDING_SOURCE,
    corpus_blocks,
)

RESUME = (
    "ResumeForge is an AI-powered resume and career platform that Ahmed designed "
    "and built, engineered specifically to run at zero cost.\n\n"
    "- **Live URL:** https://resume-forge-one-zeta.vercel.app\n"
    "- **Role:** Design and build"
)


RESUME_PROJECTS = {"ResumeForge": "projects/resumeforge.md"}


def _composer_for(chunks: list, categories: dict[str, str]) -> AnswerComposer:
    """A composer over ``chunks`` with a real BM25 vocabulary, as production has."""
    from app.services.bm25 import Bm25Index

    index = Bm25Index(chunks)
    return AnswerComposer(
        make_profile(categories, RESUME_PROJECTS),
        idf=index.idf,
        vocabulary=index.vocabulary,
    )


def _plain(text: str) -> str:
    """Markdown-stripped, whitespace-collapsed text, for verbatim comparison."""
    import re

    stripped = re.sub(r"(\*{1,3}|_{1,3}|`+)", "", text)
    stripped = re.sub(r"^\s*[-*+]\s+", "", stripped, flags=re.MULTILINE)
    return " ".join(stripped.split()).lower()


def resume_corpus() -> list:
    """Padding plus the ResumeForge summary that R1 was about."""
    padding = [
        make_chunk(text, source_file=PADDING_SOURCE, section="Notes", ordinal=index)
        for index, text in enumerate(CORPUS_PADDING)
    ]
    project = make_chunk(
        RESUME, source_file="projects/resumeforge.md", section="ResumeForge > Summary"
    )
    return [*padding, project]


# --------------------------------------------------------------------------- #
# R1: a project name must route to its overview, not to a matching URL
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    ["What is ResumeForge?", "What is resume forge?", "What is resumeforge?"],
)
def test_a_spaced_or_plain_project_name_answers_with_the_overview(query: str) -> None:
    """R1: every spelling of the name gets the same answer.

    The live service answered "What is resume forge?" starting with "Live URL:
    https://resume-forge-one-zeta.vercel.app". The reason was mechanical: "forge"
    matched only the word inside that URL, so the URL outranked the summary.
    """
    composer = _composer_for(resume_corpus(), {**CATEGORIES})

    composed = composer.compose(query, corpus_blocks(resume_corpus()))

    assert composed is not None
    assert composed.text.splitlines()[0].endswith("run at zero cost.")
    assert "ResumeForge is an AI-powered resume and career platform" in composed.text


def test_the_overview_is_the_first_line_regardless_of_how_the_name_is_spelled() -> None:
    """All spellings must produce byte-identical answers, not merely correct ones."""
    answers = set()
    for query in ("What is ResumeForge?", "What is resume forge?", "What is RESUMEFORGE?"):
        composer = _composer_for(resume_corpus(), {**CATEGORIES})
        composed = composer.compose(query, corpus_blocks(resume_corpus()))
        assert composed is not None
        answers.add(composed.text)

    assert len(answers) == 1


def test_a_project_name_still_matches_the_document_for_relevance() -> None:
    """Dropping the fragments from *matching* must not drop them from *routing*.

    The name is what proves the question is about this document, so the on-topic
    gate keeps it even though statement ranking ignores it.
    """
    composer = _composer_for(resume_corpus(), {**CATEGORIES})

    composed = composer.compose("Summarise ResumeForge", corpus_blocks(resume_corpus()))

    assert composed is not None
    assert "ResumeForge is an AI-powered resume" in composed.text


# --------------------------------------------------------------------------- #
# R2: less noise, no loss of evidence
# --------------------------------------------------------------------------- #

CERTIFICATIONS = (
    "Ahmed has exactly one recorded certification, and it is this course.\n\n"
    "This is the only certification on record.\n\n"
    "- **Issuing platform:** Infosys Springboard\n"
    "- **Completed:** 6 September 2026\n"
    "- **Instructor:** Neal Davis, AWS Certified Solutions Architect and Developer\n"
    "- **Co-instructor:** Eric E. Huerta, AWS Certified Cloud Practitioner\n"
    "- **Reference number:** 0004\n"
    "- **Certificate ID:** `UC-fd24728f-6fe7-4349-b78c-00cafb23cac5`\n"
    "- **Verification URL:** https://ude.my/UC-fd24728f-6fe7-4349-b78c-00cafb23cac5\n"
)


def _certifications_corpus() -> list:
    """Padding plus the certifications chunk R2 was measured on."""
    padding = [
        make_chunk(text, source_file=PADDING_SOURCE, section="Notes", ordinal=index)
        for index, text in enumerate(CORPUS_PADDING)
    ]
    chunk = make_chunk(
        CERTIFICATIONS,
        source_file="certifications.md",
        section="Certifications > Recorded certifications",
    )
    return [*padding, chunk]


def test_a_restatement_of_a_fact_already_given_is_dropped() -> None:
    """R2: "This is the only certification on record" repeats the sentence above it.

    Both were quoted. Ordinary token overlap is too low for the Jaccard rule --
    the second says three content words and two of them recur -- so an
    IDF-weighted comparison is needed to see that they are one fact.
    """
    composer = _composer_for(
        _certifications_corpus(), {**CATEGORIES, "certifications.md": "certifications"}
    )

    composed = composer.compose(
        "What certifications does Ahmed have?", corpus_blocks(_certifications_corpus())
    )

    assert composed is not None
    assert "exactly one recorded certification" in composed.text
    assert "This is the only certification on record." not in composed.text


def test_the_redundancy_rule_needs_enough_evidence_to_fire() -> None:
    """A short statement must not be judged redundant against a long one.

    "Build meaningful projects" is entirely contained in "Projects that are worth
    building and that move him forward" by the numbers, but it is one of the
    recorded goals, and dropping it would lose evidence rather than noise.
    """
    goals = make_chunk(
        "Build meaningful projects.\n\nProjects that are worth building and that "
        "move him forward, rather than filler work.",
        source_file="goals.md",
        section="Goals",
    )
    padding = [
        make_chunk(text, source_file=PADDING_SOURCE, section="Notes", ordinal=index)
        for index, text in enumerate(CORPUS_PADDING)
    ]
    chunks = [*padding, goals]
    composer = _composer_for(chunks, {**CATEGORIES, "goals.md": "goals"})

    composed = composer.compose("What are Ahmed's goals?", corpus_blocks(chunks))

    assert composed is not None
    assert "Build meaningful projects." in composed.text
    assert REDUNDANT_MIN_TERMS >= 3
    assert 0.0 < REDUNDANT_OVERLAP <= 1.0


@pytest.mark.parametrize(
    "row",
    [
        "Instructor: Neal Davis, AWS Certified Solutions Architect and Developer",
        "Co-instructor: Eric E. Huerta, AWS Certified Cloud Practitioner",
        "Reference number: 0004",
    ],
)
def test_a_certification_row_about_somebody_else_is_not_quoted(row: str) -> None:
    """R2: an instructor's own credentials are not Ahmed's certification."""
    composer = _composer_for(
        _certifications_corpus(), {**CATEGORIES, "certifications.md": "certifications"}
    )

    composed = composer.compose(
        "What certifications does Ahmed have?", corpus_blocks(_certifications_corpus())
    )

    assert composed is not None
    assert row not in composed.text
    assert "Neal Davis" not in composed.text
    assert "Eric E. Huerta" not in composed.text


@pytest.mark.parametrize(
    "row",
    ["Issuing platform: Infosys Springboard", "Completed: 6 September 2026"],
)
def test_rows_about_ahmeds_own_certification_are_kept(row: str) -> None:
    """The rule must target third parties only, not metadata as such."""
    composer = _composer_for(
        _certifications_corpus(), {**CATEGORIES, "certifications.md": "certifications"}
    )

    composed = composer.compose(
        "What certifications does Ahmed have?", corpus_blocks(_certifications_corpus())
    )

    assert composed is not None
    assert row in composed.text


def test_a_sentence_that_only_works_after_the_previous_one_is_dropped() -> None:
    """R2: "It is the way to reach him for project inquiries" is a continuation.

    Quoted alone it refers to a subject the reader cannot see. The rule is
    pronoun-plus-copula, so "It uses only free, open-source and locally hosted
    tooling" -- which has a verb of its own -- is untouched.
    """
    contact = make_chunk(
        "Ahmed's email address is ayaanmsrit@gmail.com.\n\n"
        "It is the way to reach him for project inquiries and initial contact.\n\n"
        "It uses only free, open-source and locally hosted tooling.",
        source_file="contact.md",
        section="Contact",
    )
    padding = [
        make_chunk(text, source_file=PADDING_SOURCE, section="Notes", ordinal=index)
        for index, text in enumerate(CORPUS_PADDING)
    ]
    chunks = [*padding, contact]
    composer = _composer_for(chunks, {**CATEGORIES, "contact.md": "contact"})

    composed = composer.compose("How can I contact Ahmed?", corpus_blocks(chunks))

    assert composed is not None
    assert "It is the way to reach him" not in composed.text
    # A pronoun-initial sentence with its own predicate survives.
    assert "It uses only free, open-source and locally hosted tooling." in composed.text


# --------------------------------------------------------------------------- #
# Opaque identifier rows are suppressed unless the question asks for them
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "row",
    [
        "Certificate ID: UC-fd24728f-6fe7-4349-b78c-00cafb23cac5",
        "Verification URL: https://ude.my/UC-fd24728f-6fe7-4349-b78c-00cafb23cac5",
    ],
)
def test_an_opaque_identifier_is_not_padded_into_a_broad_answer(row: str) -> None:
    """A UUID is unreadable filler in "What certifications does Ahmed have?".

    It is a real attribute of his own certification, so it is not third-party
    metadata and the rule that removes instructor credentials does not apply. It
    is dropped because the question did not ask for it -- and only for that
    reason.
    """
    composer = _composer_for(
        _certifications_corpus(), {**CATEGORIES, "certifications.md": "certifications"}
    )

    composed = composer.compose(
        "What certifications does Ahmed have?", corpus_blocks(_certifications_corpus())
    )

    assert composed is not None
    assert "UC-fd24728f" not in composed.text
    assert row not in composed.text


@pytest.mark.parametrize(
    "query",
    [
        "What is Ahmed's certificate ID?",
        "What is the ID of Ahmed's certificate?",
        "What is Ahmed's certificate number?",
    ],
)
def test_an_explicitly_requested_identifier_is_still_answered(query: str) -> None:
    """Suppressing the row must not suppress the fact it carries.

    The value stays in the knowledge base and is quoted whenever the question
    names the thing the label identifies.
    """
    composer = _composer_for(
        _certifications_corpus(), {**CATEGORIES, "certifications.md": "certifications"}
    )

    composed = composer.compose(query, corpus_blocks(_certifications_corpus()))

    assert composed is not None
    assert "UC-fd24728f-6fe7-4349-b78c-00cafb23cac5" in composed.text


def test_the_certificate_information_stays_in_the_knowledge_base() -> None:
    """The fix is about what gets quoted, never about what is recorded."""
    from pathlib import Path

    kb = (Path(__file__).resolve().parents[1] / "knowledge_base" / "certifications.md").read_text(
        encoding="utf-8"
    )

    assert "UC-fd24728f-6fe7-4349-b78c-00cafb23cac5" in kb
    assert "Certificate ID" in kb


def test_the_broad_certification_answer_still_carries_its_readable_facts() -> None:
    """Tidying must not hollow the answer out.

    Suppressing identifiers is only acceptable while the platform, the course and
    the completion date are still all there.
    """
    composer = _composer_for(
        _certifications_corpus(), {**CATEGORIES, "certifications.md": "certifications"}
    )

    composed = composer.compose(
        "What certifications does Ahmed have?", corpus_blocks(_certifications_corpus())
    )

    assert composed is not None
    assert "Issuing platform: Infosys Springboard" in composed.text
    assert "Completed: 6 September 2026" in composed.text


def test_a_recorded_absence_is_not_mistaken_for_a_continuation() -> None:
    """ "That is not recorded in this knowledge base" is an answer, not a fragment.

    It looks like a continuation of the question above it, and the anaphoric rule
    would drop it -- turning a question with a recorded answer into a refusal.
    """
    faq = make_chunk(
        "**Who is Ahmed?**\nMohammed Ayaan Ahmed, usually called Ahmed.\n\n"
        "**What is his nickname?**\nThat is not recorded in this knowledge base.",
        source_file="faq.md",
        section="Frequently Asked Questions > About Ahmed",
    )
    padding = [
        make_chunk(text, source_file=PADDING_SOURCE, section="Notes", ordinal=index)
        for index, text in enumerate(CORPUS_PADDING)
    ]
    chunks = [*padding, faq]
    composer = _composer_for(chunks, {**CATEGORIES})

    composed = composer.compose("What is Ahmed's nickname?", corpus_blocks(chunks))

    assert composed is not None
    assert "not recorded" in composed.text


def test_noise_reduction_does_not_drop_evidence() -> None:
    """Every statement that survives must still be present in the corpus verbatim.

    Tidying an answer is only acceptable while the answer is still entirely made
    of things the knowledge base says. This is the guard on that.
    """
    chunks = _certifications_corpus()
    composer = _composer_for(chunks, {**CATEGORIES, "certifications.md": "certifications"})

    composed = composer.compose("What certifications does Ahmed have?", corpus_blocks(chunks))

    assert composed is not None
    corpus = _plain(" ".join(chunk.text for chunk in chunks))
    for statement in composed.statements:
        assert _plain(statement) in corpus, statement
    # Tidying must not empty the answer out.
    assert len(composed.statements) >= 2
