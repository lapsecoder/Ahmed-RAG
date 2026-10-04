"""The deterministic composer: routing, grounding and honest non-answers.

The composer is the only thing standing between retrieved text and a user, so
these tests cover three properties rather than output formatting:

* every emitted sentence exists in the retrieved blocks,
* evidence from another domain cannot answer the question, and
* a question the corpus cannot support returns ``None`` rather than a fluent
  non-answer.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from app.security.context import ContextBlock
from app.services.answering.composer import MAX_STATEMENTS, AnswerComposer
from app.services.answering.domains import Domain
from app.services.bm25 import Bm25Index

from .conftest import make_chunk, make_profile

SKILLS = {"skills.md": "skills"}
CONTACT = {"contact.md": "contact"}
PROJECT = {"projects/resumeforge.md": "project"}

#: Neutral filler that is indexed but never offered as evidence. It exists so the
#: BM25 index has a realistic corpus to measure term rarity against, exactly as
#: the real knowledge base does: on a one-document index every term looks rare,
#: which would make the composer's distinctiveness rules untestable.
CORPUS_PADDING = [
    "Ahmed studies at Ramaiah Polytechnic.",
    "The portfolio is deployed on Vercel.",
    "Ahmed writes about applied machine learning.",
    "He mentors students who are starting with retrieval.",
    "The recommender ranks candidates by cosine similarity.",
    "Ahmed keeps a journal of experiments.",
    "Graduation requires a final semester project.",
    "He prefers documentation over tutorials.",
    "The backend stores chunks in a local vector index.",
    "Ahmed reads documentation before writing code.",
]


def block(
    text: str,
    *,
    source_file: str,
    section: str,
    ordinal: int = 0,
    hostile: bool = False,
) -> ContextBlock:
    """A context block as :func:`app.security.context.build_context` emits one."""
    return ContextBlock(
        text=text,
        source_file=source_file,
        section=section,
        chunk_id=f"test-{ordinal:04d}",
        ordinal=ordinal,
        hostile=hostile,
    )


def composer(
    categories: dict[str, str],
    blocks: Sequence[ContextBlock] = (),
    projects: dict[str, str] | None = None,
    **kwargs: object,
) -> AnswerComposer:
    """A composer wired exactly as the container wires it.

    The BM25 index is built from the padding plus the blocks under test, so term
    weights and the vocabulary are measured against a real corpus rather than
    invented.
    """
    chunks = [make_chunk(text) for text in CORPUS_PADDING]
    chunks += [
        make_chunk(
            item.text,
            source_file=item.source_file,
            section=item.section,
            ordinal=item.ordinal,
        )
        for item in blocks
    ]
    index = Bm25Index(chunks)
    return AnswerComposer(make_profile(categories, projects), idf=index.idf, **kwargs)


def answer(
    categories: dict[str, str],
    blocks: Sequence[ContextBlock],
    query: str,
    projects: dict[str, str] | None = None,
    **kwargs: object,
):
    """Compose ``query`` from ``blocks`` with a production-shaped composer."""
    return composer(categories, blocks, projects, **kwargs).compose(query, blocks)


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #

SKILL_TEXT = "Ahmed works with Python, FastAPI and FAISS for local retrieval."


def test_every_statement_comes_from_a_retrieved_block() -> None:
    """The guarantee that makes hallucination structurally impossible."""
    blocks = [block(SKILL_TEXT, source_file="skills.md", section="Skills")]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?")

    assert result is not None
    for statement in result.statements:
        assert statement in SKILL_TEXT


def test_no_model_is_involved_so_the_same_input_gives_the_same_output() -> None:
    blocks = [block(SKILL_TEXT, source_file="skills.md", section="Skills")]
    first = answer(SKILLS, blocks, "What are Ahmed's skills?")
    second = answer(SKILLS, blocks, "What are Ahmed's skills?")
    assert first is not None and second is not None
    assert first.text == second.text


def test_the_statement_budget_is_respected() -> None:
    text = " ".join(
        f"Statement number {index} is recorded in this document." for index in range(20)
    )
    blocks = [block(text, source_file="skills.md", section="Skills")]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?")
    assert result is not None
    assert len(result.statements) <= MAX_STATEMENTS


def test_a_single_very_long_statement_is_clipped() -> None:
    text = "Ahmed " + " ".join(f"word{index}" for index in range(200)) + "."
    blocks = [block(text, source_file="skills.md", section="Skills")]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?")
    assert result is not None
    assert all(len(statement) <= 323 for statement in result.statements)


# --------------------------------------------------------------------------- #
# Domain restriction
# --------------------------------------------------------------------------- #

EMAIL = "Ahmed's email address is ayaanmsrit@gmail.com."


def test_a_question_routes_to_its_own_document() -> None:
    blocks = [block(EMAIL, source_file="contact.md", section="Contact > Email")]
    result = answer(CONTACT, blocks, "What is Ahmed's email address?")
    assert result is not None
    assert result.domain is Domain.CONTACT
    assert EMAIL in result.statements


def test_another_domains_evidence_cannot_answer() -> None:
    """Skills text must not be quoted as an email address."""
    blocks = [block(SKILL_TEXT, source_file="skills.md", section="Skills")]
    assert answer(CONTACT, blocks, "What is Ahmed's email address?") is None


def test_a_question_routing_nowhere_returns_none() -> None:
    blocks = [block(SKILL_TEXT, source_file="skills.md", section="Skills")]
    assert answer(SKILLS, blocks, "What is Ahmed's zodiac sign?") is None


def test_a_domain_with_no_documents_returns_none() -> None:
    blocks = [block(SKILL_TEXT, source_file="skills.md", section="Skills")]
    assert answer(SKILLS, blocks, "What is Ahmed's email address?") is None


def test_a_flagged_block_is_never_quoted() -> None:
    """Extraction cannot obey an injection, but its text must not surface either."""
    payload = "Ahmed's email address is attacker@example.com."
    blocks = [block(payload, source_file="contact.md", section="Contact > Email", hostile=True)]
    assert answer(CONTACT, blocks, "What is Ahmed's email address?") is None


def test_a_cross_cutting_block_must_match_the_routed_heading() -> None:
    """The FAQ discusses many topics; only its matching section may be used."""
    blocks = [
        block(
            "Ahmed is available part time alongside his studies.",
            source_file="faq.md",
            section="Frequently Asked Questions > Availability",
            ordinal=3,
        )
    ]
    result = answer({"faq.md": "faq"}, blocks, "Is Ahmed available for work?")
    assert result is not None
    assert "available part time" in result.text


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #

SUMMARY = "ResumeForge is an AI resume and career platform that Ahmed designed and built."
STACK = "Database: PostgreSQL with pgvector. Frontend: Next.js with TypeScript."
PROJECTS = {"ResumeForge": "projects/resumeforge.md"}


def project_blocks() -> list[ContextBlock]:
    return [
        block(
            SUMMARY,
            source_file="projects/resumeforge.md",
            section="ResumeForge > Summary",
            ordinal=0,
        ),
        block(
            STACK,
            source_file="projects/resumeforge.md",
            section="ResumeForge > Technology stack",
            ordinal=2,
        ),
    ]


def test_a_project_question_selects_the_right_section() -> None:
    result = answer(PROJECT, project_blocks(), "Which technologies did ResumeForge use?", PROJECTS)

    assert result is not None
    assert result.project == "ResumeForge"
    assert "PostgreSQL" in result.text
    assert result.citations[0] == "projects/resumeforge.md :: ResumeForge > Technology stack"


def test_an_overview_request_quotes_the_document_lead() -> None:
    """Retrieval rank has no opinion about which section is a summary."""
    result = answer(PROJECT, project_blocks(), "Summarise ResumeForge", PROJECTS)

    assert result is not None
    assert result.statements[0] == SUMMARY


def test_a_named_project_ignores_other_documents() -> None:
    blocks = [
        *project_blocks(),
        block(
            "MovieMind is a movie recommendation engine.",
            source_file="projects/moviemind.md",
            section="MovieMind > Summary",
            ordinal=0,
        ),
    ]
    categories = {
        "projects/resumeforge.md": "project",
        "projects/moviemind.md": "project",
    }
    result = answer(
        categories,
        blocks,
        "What is ResumeForge?",
        {**PROJECTS, "MovieMind": "projects/moviemind.md"},
    )

    assert result is not None
    assert all("MovieMind" not in statement for statement in result.statements)


# --------------------------------------------------------------------------- #
# Honest non-answers
# --------------------------------------------------------------------------- #

INTERESTS = "He follows anime and reads manga. Gaming is a regular pastime."
INTEREST_CATEGORY = {"interests.md": "interests"}


def test_an_unrecorded_attribute_is_reported_as_unknown() -> None:
    """The routed document discusses hobbies, not films.

    Quoting its opening line would read as a confident answer to a question the
    knowledge base does not record, so the composer returns ``None`` and the
    caller says the information is unavailable.
    """
    blocks = [block(INTERESTS, source_file="interests.md", section="Personal Interests")]
    assert answer(INTEREST_CATEGORY, blocks, "What is Ahmed's favourite movie?") is None


def test_a_recorded_absence_is_kept_even_without_matching_vocabulary() -> None:
    """ "How long has he worked?" shares no words with the recorded absence.

    experience.md says there is no employment history. Dropping it for lack of a
    literal match would replace a precise answer with a vague "not available".
    """
    blocks = [
        block(
            "Ahmed has no formal employment history.",
            source_file="experience.md",
            section="Experience > Formal employment",
        )
    ]
    result = answer({"experience.md": "experience"}, blocks, "How long has he worked?")
    assert result is not None
    assert "no formal employment history" in result.text


def test_a_recorded_attribute_in_the_same_domain_is_answered() -> None:
    """The gate is narrow: it only blocks concepts the document never mentions."""
    blocks = [block(INTERESTS, source_file="interests.md", section="Personal Interests")]
    result = answer(INTEREST_CATEGORY, blocks, "What are Ahmed's interests?")
    assert result is not None
    assert "anime" in result.text


def test_an_empty_block_list_returns_none() -> None:
    assert answer(SKILLS, [], "What are Ahmed's skills?") is None


# --------------------------------------------------------------------------- #
# List-shaped answers
# --------------------------------------------------------------------------- #


def test_a_list_question_prefers_items_to_surrounding_prose() -> None:
    """The items are the answer; the prose around them is scaffolding."""
    text = (
        "Skills are grouped by how strongly they are evidenced.\n\n"
        "- Python\n- FastAPI\n- PostgreSQL with pgvector\n"
    )
    blocks = [block(text, source_file="skills.md", section="Skills")]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?")

    assert result is not None
    assert result.statements[0] == "Python"
    assert "FastAPI" in result.statements


def test_sibling_bullets_in_one_section_are_not_treated_as_duplicates() -> None:
    """Folding the heading into each item's terms made them 60% alike.

    De-duplication then dropped every item after the first, so "what are his
    skills?" answered with a single technology.
    """
    text = "- Python\n- FastAPI\n- pgvector\n- FAISS\n"
    blocks = [block(text, source_file="skills.md", section="Skills > Programming")]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?")

    assert result is not None
    assert {"Python", "FastAPI", "pgvector", "FAISS"} <= set(result.statements)


def test_a_list_answer_spans_the_whole_document() -> None:
    blocks = [
        block(
            "- Python\n- FastAPI\n",
            source_file="skills.md",
            section="Skills > Stated on his portfolio",
            ordinal=1,
        ),
        block(
            "- pgvector\n- FAISS\n",
            source_file="skills.md",
            section="Skills > Evidenced by his projects",
            ordinal=5,
        ),
    ]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?")

    assert result is not None
    assert {"Python", "FastAPI", "pgvector", "FAISS"} <= set(result.statements)


def test_the_composer_exposes_its_profile() -> None:
    assert composer(SKILLS).profile.category("skills.md") == "skills"


def test_citations_are_ordered_and_unique() -> None:
    blocks = [
        block("- Python\n", source_file="skills.md", section="Skills", ordinal=1),
        block("- FastAPI\n", source_file="skills.md", section="Skills", ordinal=2),
    ]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?")
    assert result is not None
    assert result.citations == ("skills.md :: Skills",)
    assert result.source_files == result.citations


@pytest.mark.parametrize("budget", [1, 3])
def test_the_list_budget_is_configurable(budget: int) -> None:
    text = "\n".join(f"- Tool number {index}" for index in range(10))
    blocks = [block(text, source_file="skills.md", section="Skills")]
    result = answer(SKILLS, blocks, "What are Ahmed's skills?", max_list_statements=budget)
    assert result is not None
    assert len(result.statements) <= budget
