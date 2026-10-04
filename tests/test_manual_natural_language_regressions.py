from __future__ import annotations

from pathlib import Path

import pytest
from app.security.context import build_context
from app.services.answering.composer import AnswerComposer
from app.services.answering.corpus import build_profile as build_kb_profile
from app.services.answering.intents import Domain, Intent, route
from app.services.bm25 import Bm25Index
from app.services.chunker import chunk_markdown

from .conftest import make_chunk, make_profile

DOCS = [
    (
        "Ahmed is a Diploma in Computer Science student at Ramaiah Polytechnic. "
        "He is entering the 5th semester and is expected to graduate in 2027. "
        "His field of study is Computer Science, with a focus on Artificial Intelligence and Machine Learning.",
        "education.md",
        "Education > Diploma in Computer Science",
    ),
    (
        "Ahmed has exactly three recorded projects: ResumeForge, MovieMind and Ahmed-RAG.",
        "faq.md",
        "Frequently Asked Questions > Projects",
    ),
    (
        "Ahmed is into anime, manga, gaming and comic books. He enjoys these interests outside his studies.",
        "interests.md",
        "Interests > Personal",
    ),
    (
        "Ahmed can be reached at ayaanmsrit@gmail.com or on LinkedIn.",
        "contact.md",
        "Contact > Contact",
    ),
    (
        "Ahmed is a Diploma in Computer Science student at Ramaiah Polytechnic.",
        "about.md",
        "About > Identity",
    ),
]
CATEGORIES = {s: ("project" if s.startswith("projects/") else "faq" if s == "faq.md" else s.removesuffix(".md")) for _, s, _ in DOCS}
PROJECTS = {}


def make_composer() -> AnswerComposer:
    chunks = [make_chunk(text, source_file=source, section=section, ordinal=i) for i, (text, source, section) in enumerate(DOCS)]
    index = Bm25Index(chunks)
    return AnswerComposer(make_profile(CATEGORIES, PROJECTS), idf=index.idf, vocabulary=index.vocabulary, scoped_vocabulary=index.vocabulary_for)


def blocks():
    return build_context([make_chunk(text, source_file=source, section=section, ordinal=i) for i, (text, source, section) in enumerate(DOCS)])


def test_project_paraphrase_routes_and_answers():
    query = "What kind of stuff has he made?"
    routing = route(query)
    assert routing.intent is Intent.PROJECTS
    assert routing.primary is Domain.PROJECTS
    answer = make_composer().compose(query, blocks())
    assert answer is not None
    assert "ResumeForge" in answer.text
    assert "MovieMind" in answer.text


def test_academic_history_is_education_and_uses_history_as_overview():
    query = "Any chance you could walk me through his academic history?"
    routing = route(query)
    assert routing.intent is Intent.EDUCATION
    assert routing.domains == (Domain.EDUCATION,)
    answer = make_composer().compose(query, blocks())
    assert answer is not None
    assert "Ramaiah Polytechnic" in answer.text
    assert "5th semester" in answer.text
    assert "2027" in answer.text


def test_common_framing_words_do_not_block_known_routes():
    cases = {
        "What sort of things does he get up to in his spare time?": Domain.INTERESTS,
        "How do you get in touch with him?": Domain.CONTACT,
        "What can Ahmed actually do?": Domain.SKILLS,
    }
    for query, domain in cases.items():
        routing = route(query)
        assert domain in routing.domains, (query, routing.trace)


# ------------------------------------------- an academic history is a timeline
#
# The composer answered the QA query with the FAQ's individual question-and-
# answer fragments, which is a transcript of the FAQ rather than a history: the
# stages lost their order and the recorded measures -- the CGPA, a year -- were
# dropped entirely. The fixtures below are shaped like the real education
# document: an earlier stage first, the current one second, and the measures
# recorded as rows and bullets.


JOURNEY_QUERY = "Any chance you could walk me through his academic history?"

#: A two-stage education record, in the shape a knowledge base keeps one: the
#: school that came first, then the diploma currently being studied. Each stage
#: states its institution, its qualification, the year it ended or is expected
#: to end, and -- where one exists -- the mark that was recorded for it.
JOURNEY_DOCS = [
    (
        "## Secondary School\n\n"
        "- **Institution:** Mc Nay Doons Public School\n"
        "- **Board:** ICSE\n"
        "- **Percentage:** 83%\n"
        "- **Year of passing:** 2024\n\n"
        "## Diploma in Computer Science\n\n"
        "- **Institution:** Ramaiah Polytechnic\n"
        "- **Qualification:** Diploma in Computer Science\n"
        "- **Current stage:** Entering the 5th semester\n"
        "- **Expected graduation:** 2027\n\n"
        "## Recorded marks\n\n"
        "- The recorded CGPA is 9.11.\n\n"
        "## Relevant coursework\n\n"
        "- Python programming\n"
        "- Web fundamentals\n",
        "education.md",
        "Education > Secondary School",
    ),
    (
        "**Where does he study?**\nHe studies at Ramaiah Polytechnic, for a Diploma in "
        "Computer Science.\n\n"
        "**When does he graduate?**\n2027 is the expected graduation year.\n\n"
        "**What is his CGPA?**\n9.11. The grading scale is not recorded.\n",
        "faq.md",
        "Frequently Asked Questions > Education",
    ),
]

#: The rest of the corpus. A two-document index gives every term an inverse
#: document frequency the composer rightly treats as uninformative, so a lookup
#: could not be distinguished from a generic lead. These are here to make the
#: fixture behave like a real corpus, not to be answered.
FILLER_DOCS = [
    ("Ahmed is into anime, manga, gaming and comic books.", "interests.md", "Interests > Personal"),
    ("Ahmed can be reached at ayaanmsrit@gmail.com.", "contact.md", "Contact > Contact"),
    ("Ahmed builds small, focused software projects.", "goals.md", "Goals > Projects"),
    ("Ahmed has no formal employment history.", "experience.md", "Experience > Formal employment"),
    ("He publicly lists Python, JavaScript basics and Git.", "skills.md", "Skills > Programming"),
    (
        "Ahmed completed Python Programming for AWS on 6 September 2026.",
        "certifications.md",
        "Certifications > Recorded",
    ),
]
JOURNEY_CATEGORIES = {
    "education.md": "education",
    "faq.md": "faq",
    "interests.md": "interests",
    "contact.md": "contact",
    "goals.md": "goals",
    "experience.md": "experience",
    "skills.md": "skills",
    "certifications.md": "certifications",
}
JOURNEY_CORPUS = JOURNEY_DOCS + FILLER_DOCS


def journey_blocks():
    return build_context(
        [
            make_chunk(text, source_file=source, section=section, ordinal=index)
            for index, (text, source, section) in enumerate(JOURNEY_CORPUS)
        ]
    )


def journey_composer() -> AnswerComposer:
    index = Bm25Index(
        [
            make_chunk(text, source_file=source, section=section, ordinal=position)
            for position, (text, source, section) in enumerate(JOURNEY_CORPUS)
        ]
    )
    return AnswerComposer(
        make_profile(JOURNEY_CATEGORIES, {}),
        idf=index.idf,
        vocabulary=index.vocabulary,
        scoped_vocabulary=index.vocabulary_for,
    )


def test_an_academic_history_is_answered_as_a_timeline() -> None:
    """Every stage of the record, in the order the document keeps them."""
    answer = journey_composer().compose(JOURNEY_QUERY, journey_blocks())

    assert answer is not None
    assert answer is not None and "Mc Nay Doons Public School" in answer.text
    assert answer is not None and "Ramaiah Polytechnic" in answer.text
    # The school came first, so it is quoted before the diploma.
    assert answer.text.index("Mc Nay Doons Public School") < answer.text.index("Ramaiah")
    # Coursework is context for a stage, not a stage of the record.
    assert answer is not None and "Python programming" not in answer.text


def test_an_academic_history_surfaces_the_recorded_measures() -> None:
    """The percentage, the year and the CGPA are the point of the journey."""
    answer = journey_composer().compose(JOURNEY_QUERY, journey_blocks())

    assert answer is not None
    for recorded in ("83%", "2024", "2027", "9.11"):
        assert recorded in answer.text, (recorded, answer.text)


def test_an_academic_history_does_not_invent_an_unrecorded_stage() -> None:
    """Every line quoted is a line the document records, verbatim.

    Checked against the document with its Markdown emphasis and bullet markers
    removed, because the emphasis is presentation and the statement is not.
    """
    answer = journey_composer().compose(JOURNEY_QUERY, journey_blocks())

    assert answer is not None
    recorded = "\n".join(
        block.text.replace("**", "").replace("- ", "").replace("## ", "")
        for block in journey_blocks()
    )
    for statement in answer.statements:
        assert statement in recorded, (statement, recorded)


def test_only_an_overview_of_the_record_reaches_the_journey() -> None:
    """The gate itself, so a lookup can never be answered as a timeline.

    Asserted on the gate rather than on the rendered text because a lookup's
    answer depends on the corpus it is asked of -- and whether "cgpa" counts as a
    distinctive term is a function of corpus size. That the real knowledge base
    still answers its education lookups as single facts is pinned separately,
    against the real documents.
    """
    for query in (
        "Where does Ahmed study?",
        "What is Ahmed's CGPA?",
        "When does he graduate?",
        "What is his favourite movie?",
    ):
        routing = route(query)
        assert not AnswerComposer._wants_a_journey(query, routing), query

    for query in (
        JOURNEY_QUERY,
        "Walk me through his academic history.",
        "Tell me about his academic history.",
        "Walk me through his education journey.",
    ):
        routing = route(query)
        assert AnswerComposer._wants_a_journey(query, routing), (query, routing.trace)


# ------------------------------------------------- against the real knowledge base
#
# The fixture above proves the rule; this pins the answer a visitor actually got,
# so a change to the real education.md cannot quietly stop being a history.


@pytest.fixture(scope="module")
def real_kb() -> tuple[AnswerComposer, list]:
    kb_dir = Path(__file__).resolve().parents[1] / "knowledge_base"
    profile = build_kb_profile(kb_dir, ("_*",))
    chunks = []
    for path in sorted(kb_dir.rglob("*.md")):
        if path.name.startswith("_"):
            continue
        chunks.extend(
            chunk_markdown(
                path.read_text(encoding="utf-8-sig"), path.relative_to(kb_dir).as_posix()
            )
        )
    index = Bm25Index(chunks)
    composer = AnswerComposer(
        profile,
        idf=index.idf,
        vocabulary=index.vocabulary,
        scoped_vocabulary=index.vocabulary_for,
    )
    return composer, build_context(chunks)


def test_the_real_academic_history_is_a_timeline(real_kb) -> None:
    """The QA query, against the documents a visitor actually gets."""
    composer, blocks = real_kb
    answer = composer.compose(JOURNEY_QUERY, blocks)

    assert answer is not None
    # The school record comes first, with the board, the mark and the year.
    for recorded in ("Mc Nay Doons Public School", "ICSE", "83%", "2024"):
        assert recorded in answer.text, (recorded, answer.text)
    # The diploma record follows, including the CGPA that "Not on record" holds.
    for recorded in (
        "Ramaiah Polytechnic",
        "Diploma in Computer Science",
        "5th semester",
        "2027",
        "9.11",
    ):
        assert recorded in answer.text, (recorded, answer.text)
    # Chronological: the school he left is quoted before the college he is at.
    assert answer.text.index("Mc Nay Doons Public School") < answer.text.index("Ramaiah"), (
        answer.text
    )


def test_the_real_education_lookups_are_unchanged(real_kb) -> None:
    composer, blocks = real_kb

    study = composer.compose("Where does Ahmed study?", blocks)
    assert study is not None and "Ramaiah Polytechnic" in study.text
    assert study is not None and "9.11" not in study.text, study.text

    cgpa = composer.compose("What is Ahmed's CGPA?", blocks)
    assert cgpa is not None and "9.11" in cgpa.text
    assert cgpa is not None and "Ramaiah" not in cgpa.text, cgpa.text


def test_a_prose_only_education_document_still_answers_in_prose() -> None:
    """No recorded rows means no timeline, and the ordinary answer is unchanged.

    The journey path is for documents written as a record. A knowledge base that
    keeps its education in paragraphs has no milestones to line up, and turning
    its sentences into a bullet list would drop the sentence that names the
    institution -- which is exactly the regression this work replaced.
    """
    answer = make_composer().compose(JOURNEY_QUERY, blocks())

    assert answer is not None
    assert "Ramaiah Polytechnic" in answer.text
    assert "5th semester" in answer.text
    assert "2027" in answer.text
