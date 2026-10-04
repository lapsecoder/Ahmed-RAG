"""Which part of the corpus a question is about, and which project it names.

Routing itself lives in :mod:`app.services.answering.intents`. This module keeps
the project-name resolution that intents deliberately does not own -- it needs
the corpus's own project names, which are not known until the knowledge base has
been read -- and re-exports the routing vocabulary so existing imports and tests
keep working against one definition.
"""

from __future__ import annotations

import re
from typing import Final

from app.services.answering.intents import (
    CROSS_CUTTING_CATEGORY,
    DOMAIN_CATEGORIES,
    DOMAIN_PRIORITY,
    OVERVIEW_TERMS,
    Domain,
    detect_domain,
    is_overview_query,
    normalised_query,
    query_words,
    route,
    routing_terms,
)
from app.services.text import normalise_key

__all__ = [
    "CROSS_CUTTING_CATEGORY",
    "DOMAIN_CATEGORIES",
    "DOMAIN_PRIORITY",
    "MAX_PROJECT_NAME_WORDS",
    "OVERVIEW_TERMS",
    "Domain",
    "detect_domain",
    "detect_project",
    "detect_unrecognised_project",
    "is_overview_query",
    "is_part_of_project_name",
    "normalised_query",
    "query_words",
    "route",
    "routing_terms",
]


#: Longest run of adjacent words a project name may span. "Resume Forge" and
#: "Movie Mind" are two words each; three leaves headroom without letting a name
#: match across a clause boundary.
MAX_PROJECT_NAME_WORDS: int = 3


def detect_project(query: str, project_names: dict[str, str]) -> str | None:
    """Return the canonical project name mentioned in ``query``.

    Matching is done over every run of up to :data:`MAX_PROJECT_NAME_WORDS`
    adjacent words, with the spaces removed. That folds the natural ways a name
    is written -- "ResumeForge", "resume forge", "RESUMEFORGE", "resume-forge"
    -- into one comparison, which a padded whole-string search could not:
    "resume forge" normalised to "resume forge" and never contained the token
    "resumeforge", so a question asked with the name spaced out silently stopped
    being recognised as being about that project.

    Args:
        query: The user's question.
        project_names: Normalised name -> display name, derived from the corpus
            (see :class:`~app.services.answering.corpus.CorpusProfile`). Names
            come from the knowledge base itself rather than a hardcoded list, so
            adding a project file is enough to make its questions route.

    Returns:
        The display name of the project, or ``None`` when the question does not
        name one.
    """
    words = normalise_key(query).split()
    if not words:
        return None
    # Longest names first, so a two-word name is never shadowed by a one-word
    # name that happens to be its prefix.
    ordered = sorted(
        project_names.items(),
        key=lambda item: (-len(item[0].replace(" ", "")), item[0]),
    )
    for key, display in ordered:
        compact_key = key.replace(" ", "")
        if not compact_key:
            continue
        for start in range(len(words)):
            for size in range(1, MAX_PROJECT_NAME_WORDS + 1):
                end = start + size
                if end > len(words):
                    break
                if "".join(words[start:end]) == compact_key:
                    return display
    return None


#: The subject's own name, as tokens. A capitalised run containing one of these
#: is a sentence that happens to open with a capital rather than a product name;
#: see :func:`detect_unrecognised_project`, which is the only place this matters.
_SUBJECT_NAME_TOKENS: Final[frozenset[str]] = frozenset({"ahmed", "mohammed", "ayaan"})

#: Capitalised words that open a sentence or name a person, an employer or a
#: platform, and so are never a project name. Without this, "Who is Ahmed?" would
#: look like a reference to a project called "Ahmed".
_NOT_A_PROJECT: Final[frozenset[str]] = _SUBJECT_NAME_TOKENS | frozenset(
    {
        # sentence openers and question words
        "what",
        "who",
        "whom",
        "whose",
        "which",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        # modal and auxiliary verbs. "Can someone hire Ahmed?" opens with a
        # capital C, and the unknown-project guard read "Can" as a product name
        # and declined a question availability.md answers outright.
        "can",
        "could",
        "may",
        "might",
        "must",
        "should",
        "would",
        "will",
        "shall",
        "do",
        "have",
        "has",
        "had",
        "does",
        "did",
        "tell",
        "describe",
        "explain",
        "summarize",
        "summarise",
        "give",
        "show",
        "list",
        "hi",
        "hello",
        "hey",
        "thanks",
        # the subject
        "i",
        "my",
        "me",
        # organisations and credentials the knowledge base names
        "infosys",
        "springboard",
        "aws",
        "vercel",
        "github",
        "linkedin",
        "instagram",
        "tmdb",
        "cgpa",
    }
)

#: A run of capitalised words, which is how a product or project name appears.
_NAME_LIKE_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ -][A-Z][A-Za-z0-9]*)*")


def _has_name_evidence(phrase: str) -> bool:
    """True when ``phrase`` is a name on evidence beyond its opening capital.

    Two independent signals, either sufficient:

    * an internal capital, which is what distinguishes a product name written as
      one token ("FakeProject") from an ordinary word ("Any"); or
    * a word after the first that is not ordinary vocabulary, which is what
      distinguishes a spaced name ("Movie Mind") from a capitalised ordinary
      word.

    Args:
        phrase: A capitalised run taken from the question.
    """
    if any(character.isupper() for word in phrase.split() for character in word[1:]):
        return True
    words = normalise_key(phrase).split()
    return any(word not in _NOT_A_PROJECT for word in words[1:])


def is_part_of_project_name(term: str, project: str) -> bool:
    """True when ``term`` is a fragment of ``project``'s name rather than a topic.

    "moviemind" is a single token to the corpus, but a question may write it as
    "movie mind", and then "movie" and "mind" arrive as free-standing concepts.
    They are not subject matter to be looked up -- they are the name of the
    document already chosen -- so the answer layer drops them. Without this,
    "Tell me about movie mind." was declined for want of a document that mentions
    the word "mind", while the MovieMind project sat right there.

    Fragments shorter than three characters are ignored, so a two-letter word
    cannot be swallowed by any project name.
    """
    if len(term) < 3:
        return False
    return term in normalise_key(project).replace(" ", "")


def detect_unrecognised_project(query: str, project_names: dict[str, str]) -> str | None:
    """Return a project-like name in ``query`` that the corpus does not contain.

    When a question names a specific project, the only honest answers are that
    project's evidence or no information at all. Falling back to whatever project
    document sorted first answered "Tell me about a project called FakeProject."
    with a confident, fully-cited description of ResumeForge -- the citations were
    true, the answer was not, and the user had no way to see the difference.

    Args:
        query: The user's question.
        project_names: The known names, as passed to :func:`detect_project`.

    Returns:
        The unmatched name as it was written, or ``None`` when every capitalised
        name in the question is either a known project or ordinary vocabulary.
    """
    known = {key.replace(" ", "") for key in project_names}
    for match in _NAME_LIKE_RE.finditer(query):
        phrase = match.group(0)
        if normalise_key(phrase).replace(" ", "") in known:
            continue
        # An all-capitals run is an acronym, not a product name. "What is Ahmed's
        # certificate ID?" contains "ID", and refusing it as an unknown project
        # suppressed the one row the question actually asked for. Product names
        # are CamelCase ("ResumeForge") or spaced ("Movie Mind"); abbreviations are
        # not, and the distinction is free of exceptions to enumerate.
        if phrase.isupper():
            continue
        words = normalise_key(phrase).split()
        if not words:
            continue
        # The first word of a sentence is capitalised by convention, so its
        # capital is not evidence of anything. A run that opens the question
        # therefore has to show evidence beyond that one letter: an internal
        # capital anywhere in it ("FakeProject", "Summarize MovieMind"), or a
        # later word that is not ordinary vocabulary ("Movie Mind").
        #
        # Without this, "Any chance you could walk me through his academic
        # history?" was read as a reference to a project called "Any" and
        # declined -- the second time this guard had been defeated by an
        # ordinary sentence opener, after "Introduce Ahmed". Enumerating openers
        # is the failure mode this rule exists to remove.
        if match.start() == 0 and not _has_name_evidence(phrase):
            continue
        # Capitalisation runs across an ordinary word boundary -- "Which AWS
        # services" is one capitalised run, but neither word names a project.
        # Only a run with at least one non-ordinary word can name one.
        if all(word in _NOT_A_PROJECT for word in words):
            continue
        # A capitalised run containing the subject's own name is not a product,
        # it is a sentence that happens to open with a capital. "Introduce
        # Ahmed." is the case: "Introduce" is capitalised because it starts the
        # sentence, and the run is therefore "Introduce Ahmed" -- a verb and a
        # man's name, which read as a project the corpus has never heard of, so
        # the guard declined a question about.md answers outright.
        #
        # The rule is structural rather than lexical: a product name does not
        # contain the person it is about, so any run naming the subject is
        # disqualified on that ground alone. That covers every opener of this
        # shape -- introduce, describe, summarise, present, walk me through --
        # without enumerating them, which is what the verb entries in
        # :data:`_NOT_A_PROJECT` were doing one word at a time. It costs
        # nothing in coverage: a real project named after him would already have
        # been returned by :func:`detect_project` before this scan runs, so the
        # only runs reaching here are ones the corpus has never declared.
        if _SUBJECT_NAME_TOKENS.intersection(words):
            continue
        return phrase
    return None
