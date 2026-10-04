"""Regression tests for the two wrong-domain routing defects.

Both are the same failure wearing different clothes, and both are what the
polysemy rule in :data:`app.services.answering.intents.AMBIGUOUS_SURFACES`
exists to prevent.

**Defect 1 -- a generic noun manufactured a domain.** "What is Ahmed's bank
balance?" routed to interests, because "balance" was a hobby surface, and was
then answered with his hobbies. The answer was grounded, which is what made it
dangerous: nothing downstream could tell that the document had been chosen by a
word that never referred to it. The corpus even uses the same root -- "he
deliberately balances his projects" -- so no amount of evidence-checking would
have caught it. Only the routing decision was ever wrong.

The first version of the fix deleted "balance" from the vocabulary. That works
and is exactly what this file exists to forbid: it leaves the mechanism intact,
so the next generic noun someone adds restores the bug, and it cannot be
distinguished from a special case written for "bank balance".

**Defect 2 -- a generic noun manufactured a co-answer.** "Any chance you could
walk me through his academic history?" routed to education *and* experience,
because "history" is an employment surface. "Academic" had already claimed the
noun for education and the experience document was dragged in anyway, then won
the answer outright because it is the only document that uses the word
"history" at all.

The rule these tests hold is one sentence: **a word whose referent is fixed by
a modifier may not begin a route, only join one.** "employment history" is
unaffected -- "employment" is in the same group, so it corroborates "history"
-- while "academic history" and "bank balance" have nothing to corroborate them
and route nowhere.

Hermetic: a miniature corpus, no model, nothing fetched.
"""

from __future__ import annotations

import pytest
from app.security.context import build_context
from app.services.answering.composer import AnswerComposer
from app.services.answering.intents import AMBIGUOUS_SURFACES, CONCEPT_GROUPS, Domain, Intent, route
from app.services.bm25 import Bm25Index

from .conftest import make_chunk, make_profile

IDENTITY = (
    "Mohammed Ayaan Ahmed, commonly known as Ahmed. "
    "He is a Diploma in Computer Science student at Ramaiah Polytechnic."
)
#: interests.md deliberately uses "balances" -- the corpus root that let the old
#: routing pass the on-topic test while pointing at the wrong document.
INTERESTS = "He follows anime and reads manga. He deliberately balances his projects."
EDUCATION = (
    "Ahmed studies at Ramaiah Polytechnic. His expected graduation year is 2027. "
    "He is in the fifth semester of the diploma."
)
#: experience.md is what "his academic history" was answered from before.
EXPERIENCE = (
    "Ahmed has no formal employment history. He has never held a full-time role "
    "or a freelance client engagement."
)
#: The real knowledge base's FAQ restates the diploma as prose. Without it the
#: miniature corpus has no document that shares a single word with "his academic
#: history", and the composer correctly declines a question nothing can answer --
#: which would make this file's positive assertion about education vacuous.
FAQ_EDUCATION = (
    "**What is his academic history?**\n"
    "His academic history is one ongoing qualification: a Diploma in Computer "
    "Science at Ramaiah Polytechnic, expected in 2027."
)

DOCUMENTS: list[tuple[str, str, str]] = [
    (IDENTITY, "about.md", "About > Identity"),
    (INTERESTS, "interests.md", "Interests"),
    (EDUCATION, "education.md", "Education"),
    (EXPERIENCE, "experience.md", "Experience"),
    (FAQ_EDUCATION, "faq.md", "Frequently Asked Questions > Education"),
]

CATEGORIES = {
    "about.md": "profile",
    "interests.md": "interests",
    "education.md": "education",
    "experience.md": "experience",
    "faq.md": "faq",
}


def composer() -> AnswerComposer:
    chunks = [
        make_chunk(text, source_file=source, section=section, ordinal=index)
        for index, (text, source, section) in enumerate(DOCUMENTS)
    ]
    index = Bm25Index(chunks)
    return AnswerComposer(
        make_profile(CATEGORIES, {}),
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


# ---------------------------------------------------------------- defect 1


@pytest.mark.parametrize(
    "query",
    [
        "What is Ahmed's bank balance?",
        "What is his account balance?",
        "How much is in his bank balance?",
        "What's his balance?",
    ],
)
def test_a_generic_noun_does_not_manufacture_a_domain(query: str) -> None:
    """An uncorroborated polysemous noun routes nowhere at all.

    Not "to the least-bad domain" -- nowhere. A bank balance is a real question
    with a real answer the knowledge base does not have, and the honest outcome
    is to say so rather than to substitute the nearest document that happens to
    contain the same letters.
    """
    routing = route(query)

    assert routing.intent is Intent.UNKNOWN, (query, routing.trace)
    assert Domain.INTERESTS not in routing.domains, (query, routing.domains)
    assert Domain.INTERESTS not in routing.spanned, (query, routing.spanned)


def test_the_bank_balance_question_is_declined_rather_than_answered() -> None:
    """End to end: no interests document is quoted.

    Asserting only on routing would miss the failure that mattered. The old
    answer was *grounded* -- "he deliberately balances his projects" really is in
    interests.md -- so the guarantee worth holding is about the response, not
    about the intent alone.
    """
    answer = composer().compose("What is Ahmed's bank balance?", blocks())

    assert answer is None


def test_a_generic_noun_is_absent_from_the_trace_as_a_route() -> None:
    """The noun is still seen -- it just may not start anything.

    Silently dropping the word would be the wrong fix for the wrong reason: the
    composer needs to know the router recognised it, or a document that does
    discuss balances cannot be admitted when a real question arrives. The trace
    records it as uncorroborated, which is both explainable and inert.
    """
    trace = route("What is Ahmed's bank balance?").trace

    assert any(entry.startswith("uncorroborated:hobby") for entry in trace), trace
    assert not any(entry.startswith("concept:hobby") for entry in trace), trace


# ---------------------------------------------------------------- defect 2


@pytest.mark.parametrize(
    "query",
    [
        "Any chance you could walk me through his academic history?",
        "Could you walk me through his academic history?",
        "Tell me about his academic history.",
        "Walk me through his academic history.",
    ],
)
def test_a_generic_noun_does_not_manufacture_a_co_answer(query: str) -> None:
    """ "Academic" fixes what "history" refers to, so only education answers.

    The defect was not that experience was ranked highly; it was that it was
    admitted at all. "Academic" names the subject matter, and a document the
    question did not ask about must not be able to add itself on the strength of
    a noun another group has already claimed.
    """
    routing = route(query)

    assert routing.intent is Intent.EDUCATION, (query, routing.trace)
    assert routing.domains == (Domain.EDUCATION,), (query, routing.domains)
    assert Domain.EXPERIENCE not in routing.spanned, (query, routing.spanned)


def test_the_academic_history_question_answers_from_education_only() -> None:
    """End to end: the employment document is neither used nor cited."""
    answer = composer().compose(
        "Any chance you could walk me through his academic history?", blocks()
    )

    assert answer is not None
    assert all("experience.md" not in citation for citation in answer.citations), answer.citations
    assert all("formal employment history" not in statement for statement in answer.statements), (
        answer.statements
    )
    assert any("Ramaiah Polytechnic" in statement for statement in answer.statements), (
        answer.statements
    )


# --------------------------------------------------- the mechanism, not a rule


@pytest.mark.parametrize(
    "query",
    [
        "What is Ahmed's employment history?",
        "What's his work history?",
        "Tell me about his professional history.",
    ],
)
def test_corroboration_within_the_group_still_routes(query: str) -> None:
    """The fix must not cost the legitimate question it looks like it breaks.

    "employment history", "work history" and "professional history" are three
    ways of asking one question about the same document, and they were all
    answerable before. Each pairs an ambiguous noun with an unambiguous word
    *from the same group*, which is precisely the case the rule permits.
    """
    routing = route(query)

    assert routing.intent is Intent.EXPERIENCE, (query, routing.trace)
    assert Domain.EXPERIENCE in routing.domains, (query, routing.domains)


def test_the_ambiguous_set_is_a_property_of_words_not_of_questions() -> None:
    """Guard against the defect being reintroduced as a special case.

    The obvious way to "fix" defect 1 is a rule matching "bank balance", and the
    obvious way to "fix" defect 2 is one matching "academic history". Both would
    pass everything above while leaving the mechanism absent, and both would be
    defeated by the next paraphrase. So: every ambiguous word must still be a
    *listed surface* of its group -- removed from the vocabulary is not an
    acceptable substitute -- and no group's ambiguous words may all be ambiguous.
    """
    for word in AMBIGUOUS_SURFACES:
        holders = [group for group in CONCEPT_GROUPS if word in group.surfaces]
        assert holders, (word, "listed nowhere: deleted rather than made ambiguous")


def test_no_group_is_made_entirely_of_ambiguous_words() -> None:
    """A group that cannot corroborate itself could never route.

    This is the failure mode of the rule itself: adding one ambiguous word to a
    one-word group silences the group permanently, which no test above would
    catch because each of them uses a group with a second, unambiguous word.
    """
    for group in CONCEPT_GROUPS:
        assert group.surfaces > group.ambiguous, (group.name, sorted(group.ambiguous))
