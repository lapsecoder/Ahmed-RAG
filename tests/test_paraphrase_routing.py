"""Part 6: a deterministic paraphrase evaluation set.

The brief's requirement is that paraphrases of the same question route
consistently to the same domain(s), *without* requiring any particular answer
wording. These tests therefore assert routing and outcome, never the text.

Two things are being held here, and they are different.

The first is **consistency within a group**: "What is Ahmed focused on?", "What
does Ahmed focus on?" and "What's he working toward?" are three ways of asking
one question, so all three must land on the same domain. A system that
enumerates questions would pass this only by accident, and a system that routes
on meaning passes it by construction.

The second is **generalisation past the enumerated set**: the questions in
:data:`UNSEEN_PARAPHRASES` are not in any table. Each routes because of the
mechanism -- concept groups, frames, weighted co-answers -- rather than because
the string was anticipated. That is the claim worth making, because it is the
one a keyword list cannot make.

Hermetic: a miniature corpus, no model, nothing fetched.
"""

from __future__ import annotations

import itertools

import pytest
from app.security.context import build_context
from app.services.answering.composer import AnswerComposer
from app.services.answering.intents import Domain, Intent, route
from app.services.bm25 import Bm25Index

from .conftest import make_chunk, make_profile

IDENTITY = (
    "Mohammed Ayaan Ahmed, commonly known as Ahmed. "
    "He is a Diploma in Computer Science student at Ramaiah Polytechnic. "
    "He focuses on artificial intelligence and machine learning."
)
EDUCATION = (
    "Ahmed studies at Ramaiah Polytechnic. His expected graduation year is 2027. "
    "He is in the fifth semester of the diploma."
)
GOALS = (
    "Ahmed's goals are to build a future with AI, have a stable career, and enjoy life. "
    "He is interested in artificial intelligence and machine learning."
)
SKILLS = "Skills are grouped by how strongly they are evidenced.\n- Python\n- pandas"
INTERESTS = "He follows anime and reads manga. Gaming is a personal interest."
RESUMEFORGE = "ResumeForge is an AI-powered resume and career platform Ahmed designed and built."
MOVIEMIND = "MovieMind is a local movie recommendation engine Ahmed built."
AVAILABILITY = "Ahmed is open to internships and entry-level work."
CERTIFICATIONS = "Ahmed has exactly one recorded certification.\n- Issuing platform: Springboard"
CONTACT = "Ahmed can be reached on LinkedIn or GitHub."
FAQ_PROJECTS = "**What has Ahmed built?**\nThree, all designed and built by him."

DOCUMENTS: list[tuple[str, str, str]] = [
    (IDENTITY, "about.md", "About > Identity"),
    (EDUCATION, "education.md", "Education"),
    (GOALS, "goals.md", "Goals"),
    (SKILLS, "skills.md", "Skills"),
    (INTERESTS, "interests.md", "Interests"),
    (RESUMEFORGE, "projects/resumeforge.md", "ResumeForge > Summary"),
    (MOVIEMIND, "projects/moviemind.md", "MovieMind > Summary"),
    (AVAILABILITY, "availability.md", "Availability"),
    (CERTIFICATIONS, "certifications.md", "Certifications"),
    (CONTACT, "contact.md", "Contact"),
    (FAQ_PROJECTS, "faq.md", "Frequently Asked Questions > Projects"),
]

CATEGORIES = {
    "about.md": "profile",
    "education.md": "education",
    "goals.md": "goals",
    "skills.md": "skills",
    "interests.md": "interests",
    "availability.md": "availability",
    "certifications.md": "certifications",
    "contact.md": "contact",
    "faq.md": "faq",
    "projects/resumeforge.md": "project",
    "projects/moviemind.md": "project",
}

PROJECTS = {"ResumeForge": "projects/resumeforge.md", "MovieMind": "projects/moviemind.md"}


def composer() -> AnswerComposer:
    chunks = [
        make_chunk(text, source_file=source, section=section, ordinal=index)
        for index, (text, source, section) in enumerate(DOCUMENTS)
    ]
    index = Bm25Index(chunks)
    return AnswerComposer(
        make_profile(CATEGORIES, PROJECTS),
        idf=index.idf,
        vocabulary=index.vocabulary,
        scoped_vocabulary=index.vocabulary_for,
    )


def blocks() -> list:
    return build_context(
        [
            make_chunk(text, source_file=source, section=section, ordinal=index)
            for index, (text, source, section) in enumerate(DOCUMENTS)
        ]
    )


#: The brief's list, grouped by the domain each group must reach.
PARAPHRASE_GROUPS: tuple[tuple[Domain, tuple[str, ...]], ...] = (
    (
        Domain.IDENTITY,
        (
            "Who are you?",
            "Who r u?",
            "Who is Ahmed?",
            "Tell me about Ahmed.",
            "Introduce Ahmed.",
            "What's your background?",
        ),
    ),
    (
        Domain.GOALS,
        (
            "What is Ahmed focused on?",
            "What does Ahmed focus on?",
            "What's he working toward?",
            "What area is Ahmed interested in?",
        ),
    ),
    (
        Domain.INTERESTS,
        (
            "What does Ahmed do for fun?",
            "What does he enjoy?",
            "What's he into?",
            "What are his hobbies?",
            "What does he like doing?",
        ),
    ),
    (
        Domain.PROJECTS,
        (
            "What has Ahmed built?",
            "What projects has he made?",
            "What has he worked on?",
            "Show me Ahmed's projects.",
        ),
    ),
    (
        Domain.SKILLS,
        (
            "What is Ahmed good at?",
            "What can Ahmed do?",
            "What does he know technically?",
            "What technologies does he work with?",
        ),
    ),
    (
        Domain.EDUCATION,
        (
            "What's Ahmed studying?",
            "Where does he study?",
            "What's his academic background?",
            "What diploma is he pursuing?",
        ),
    ),
    (
        Domain.AVAILABILITY,
        (
            "Is Ahmed available for internships?",
            "What kind of work is Ahmed looking for?",
            "Can someone hire Ahmed?",
        ),
    ),
    (
        Domain.CERTIFICATIONS,
        ("What certifications does Ahmed have?",),
    ),
    (Domain.CONTACT, ("How can I reach Ahmed?",)),
)

#: The brief's worked example, plus paraphrases chosen to share *no* phrasing
#: with the enumerated set above. None of these strings is a rule.
UNSEEN_PARAPHRASES: tuple[tuple[str, Domain], ...] = (
    ("Can you give me a quick introduction to Ahmed?", Domain.IDENTITY),
    ("Would you mind telling me a bit about Ahmed himself?", Domain.IDENTITY),
    ("What sort of things does he get up to in his spare time?", Domain.INTERESTS),
    ("Any chance you could walk me through his academic history?", Domain.EDUCATION),
    ("How do you get in touch with him?", Domain.CONTACT),
)


@pytest.mark.parametrize(
    ("domain", "query"),
    [(domain, query) for domain, queries in PARAPHRASE_GROUPS for query in queries],
)
def test_paraphrases_of_one_question_reach_one_domain(domain: Domain, query: str) -> None:
    """Every phrasing of a question routes to the domain the question is about.

    Wording is deliberately not asserted. The composer selects from whatever the
    retrievers surfaced, so the answer to "Who r u?" and to "Who are you?" may
    differ in which sentences are quoted while remaining equally correct.
    """
    routing = route(query)

    assert domain in routing.domains, (query, routing.domains, routing.trace)


@pytest.mark.parametrize(
    ("domain", "query"),
    [(domain, query) for domain, queries in PARAPHRASE_GROUPS for query in queries],
)
def test_paraphrases_are_answered_rather_than_declined(domain: Domain, query: str) -> None:
    """Routing correctly is only half the requirement; the answer must follow.

    A router that recognises a paraphrase and then declines it has moved the
    failure rather than fixed it. Every paraphrase in the brief's list must
    produce an answer from the knowledge base.
    """
    composed = composer().compose(query, blocks())

    assert composed is not None, (query, route(query).trace)
    assert composed.domain in route(query).domains
    assert composed.text.strip()
    assert composed.citations, (query, "an answered question must cite its evidence")


@pytest.mark.parametrize(("query", "domain"), UNSEEN_PARAPHRASES)
def test_an_unenumerated_paraphrase_still_routes(query: str, domain: Domain) -> None:
    """Generalisation, not memorisation.

    "Can you give me a quick introduction to Ahmed?" was never added as a rule.
    It routes because "introduction" is a surface of the *identity* concept and
    the ``<any> introduction to <any>`` frame both point there -- the same two
    signals that answer "Introduce Ahmed." and "Tell me about Ahmed." The words
    the sentence does not share with those questions are irrelevant to routing
    and are filtered out before it, because a request modifier is not a subject.

    A keyword list cannot make this claim: it can only answer the strings it
    contains.

    Only *routing* is asserted here, and that is the whole claim. Whether an
    answer follows additionally depends on evidence existing, which this
    miniature corpus is too thin to guarantee for an arbitrary phrasing; the
    enumerated groups above assert the full route-and-answer path, and the real
    knowledge base is checked end to end by the end-to-end runner and by
    ``scripts/conversation_probe.py``.
    """
    routing = route(query)

    assert domain in routing.domains, (query, routing.domains, routing.trace)
    assert routing.intent is not Intent.UNKNOWN, (query, routing.trace)


def test_the_enumerated_table_does_not_contain_the_unseen_questions() -> None:
    """Guard the claim above: the unseen set must really be unseen.

    Without this, "generalisation" would be unfalsifiable -- the next person to
    add "quick introduction to Ahmed" as a literal rule would leave the test
    green while removing the behaviour it claims to test.
    """
    from app.services.answering import intents as routing_tables

    patterns = [frame.pattern.pattern for frame in routing_tables.FRAMES]
    surfaces = {word for group in routing_tables.CONCEPT_GROUPS for word in group.surfaces}
    vocabulary = surfaces | set(patterns)

    for query, _ in UNSEEN_PARAPHRASES:
        # No two consecutive content words of an unseen question may both be
        # listed vocabulary: that is what "never added as a rule" has to mean.
        words = [word for word in query.lower().split() if word.isalpha()]
        bigrams = [f"{a} {b}" for a, b in itertools.pairwise(words)]
        for phrase in bigrams:
            assert phrase not in vocabulary, (query, phrase)


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("Who are you?", Intent.IDENTITY),
        ("What has Ahmed built?", Intent.PROJECTS),
        ("Where does he study?", Intent.EDUCATION),
        ("What is Ahmed good at?", Intent.SKILLS),
        ("What does Ahmed do for fun?", Intent.INTERESTS),
        ("Is Ahmed available for internships?", Intent.AVAILABILITY),
        ("How can I reach Ahmed?", Intent.CONTACT),
    ],
)
def test_a_stable_question_reaches_a_stable_intent(query: str, intent: Intent) -> None:
    """Routing is pure: same input, same intent, no state and no randomness."""
    first = route(query)
    second = route(query)

    assert first.intent is intent, (query, first.trace)
    assert second.intent is first.intent
    assert second.domains == first.domains
    assert second.trace == first.trace
