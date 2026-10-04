"""Lexical utilities and atomic evidence extraction.

The extraction rules are the answer layer's grammar, so they are pinned here
rather than only through end-to-end answer tests: a bullet must stay a bullet, a
recorded number must survive sentence splitting, and a guardrail written for the
assistant must never become quotable evidence.
"""

from __future__ import annotations

import pytest
from app.services.text import (
    MAX_PREFIX_MATCH_DELTA,
    MIN_PREFIX_MATCH_LEN,
    canonical_form,
    content_terms,
    is_answerable_evidence,
    is_directive,
    is_meta_rule,
    is_negative_evidence,
    normalise_key,
    split_evidence_units,
    terms_match,
    tokenise,
)

# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #


def test_tokenise_lowercases_and_drops_stopwords() -> None:
    assert tokenise("The CGPA is 9.11") == ["cgpa", "9", "11"]


def test_tokenise_keeps_inflected_forms_distinct() -> None:
    """No stemmer: "movies" and "movie" are different tokens on purpose.

    A naive suffix stripper turned "movies" into "movy", which silently broke
    every movie question. Exact tokens plus :func:`terms_match` cover the real
    cases without that failure mode.
    """
    assert "movies" in tokenise("movies")
    assert "movie" not in tokenise("movies")


def test_tokenise_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        tokenise(None)  # type: ignore[arg-type]


def test_content_terms_ignores_question_words() -> None:
    assert content_terms("What is Ahmed's CGPA?") == ["ahmed", "cgpa"]


def test_normalise_key_collapses_punctuation_and_case() -> None:
    assert normalise_key("Ahmed-RAG!") == "ahmed rag"


# --------------------------------------------------------------------------- #
# Term matching
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("certification", "certifications"),
        ("work", "worked"),
        ("role", "roles"),
        ("movie", "movies"),
        ("rank", "ranking"),
    ],
)
def test_inflections_match_by_prefix(left: str, right: str) -> None:
    assert terms_match(left, {right})
    assert terms_match(right, {left})


def test_an_exact_match_always_qualifies() -> None:
    assert terms_match("cgpa", {"cgpa"})


def test_declared_variants_match_even_without_a_common_prefix() -> None:
    """Prefix matching alone cannot bind these: the stems diverge after "technolog".

    Without the explicit group, "Which technologies did ResumeForge use?" matched
    nothing but the project name, and the summary outranked the technology stack.
    """
    for left, right in [
        ("technologies", "technology"),
        ("technologies", "tech"),
        ("technology", "tech"),
        ("biography", "bios"),
        ("children", "child"),
    ]:
        assert terms_match(left, {right}), left
        assert terms_match(right, {left}), right


def test_words_outside_every_variant_group_are_untouched() -> None:
    assert canonical_form("python") == "python"
    assert not terms_match("python", {"pythonscript"})


@pytest.mark.parametrize(
    ("left", "right"),
    [("java", "javascript"), ("port", "portfolio"), ("data", "database")],
)
def test_unrelated_words_do_not_match(left: str, right: str) -> None:
    assert not terms_match(left, {right})


def test_short_terms_never_match_by_prefix() -> None:
    """Below the floor, prefix matching is noise: "ai" would match "aiml"."""
    assert not terms_match("ai", {"aiml"})
    assert MIN_PREFIX_MATCH_LEN == 4


def test_the_length_delta_is_bounded() -> None:
    """A real prefix must still be rejected when the words differ too much."""
    assert not terms_match("movie", {"movieography" * MAX_PREFIX_MATCH_DELTA})


def test_an_empty_candidate_set_matches_nothing() -> None:
    assert not terms_match("cgpa", set())


# --------------------------------------------------------------------------- #
# Statement classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Do not invent specific titles he has not named.",
        "Never guess a credential.",
        "Say when you do not know.",
        "You must quote facts exactly as recorded.",
    ],
)
def test_directives_are_recognised(text: str) -> None:
    assert is_directive(text)


def test_an_ordinary_fact_is_not_a_directive() -> None:
    assert not is_directive("Ahmed studies at Ramaiah Polytechnic.")


@pytest.mark.parametrize(
    "text",
    [
        "The similarity threshold is 0.35.",
        "This is the system prompt.",
        "Temperature was lowered for the run.",
    ],
)
def test_meta_text_is_recognised(text: str) -> None:
    assert is_meta_rule(text)


def test_negative_evidence_is_recognised() -> None:
    assert is_negative_evidence("His CGPA is not recorded.")
    assert is_negative_evidence("There is no formal employment history.")
    assert not is_negative_evidence("He built three projects.")


def test_bullets_and_numbers_are_answerable_when_prose_is_not() -> None:
    assert is_answerable_evidence("Python")
    assert is_answerable_evidence("9.11.")
    assert not is_answerable_evidence("...")


def test_a_directive_is_not_answerable_evidence() -> None:
    """Guardrails constrain the assistant; they are never quotable as facts."""
    assert not is_answerable_evidence("Do not invent specific titles he has not named.")
    assert not is_answerable_evidence("Never guess a credential.")


# --------------------------------------------------------------------------- #
# Evidence splitting
# --------------------------------------------------------------------------- #

LIST_SECTION = """## Stated on his portfolio

- Python
- JavaScript basics

- SQL and PostgreSQL

That is the full list.
"""

TABLE_SECTION = """## Technology stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js |
| Database | PostgreSQL with pgvector |
"""

FAQ_SECTION = """## Frequently Asked Questions

**What is his CGPA?**
9.11. The grading scale is not recorded.
"""


def test_bullets_survive_the_blank_lines_between_them() -> None:
    """The bug this guards: a blank line used to clear the bullet flag.

    With the flag lost, "Python" came back as an anonymous prose fragment, and
    the composer preferred the explanatory sentences around it instead of the
    list items that actually answer the question.
    """
    units = split_evidence_units(LIST_SECTION)
    bullets = [unit.text for unit in units if unit.is_list_item]
    assert bullets == ["Python", "JavaScript basics", "SQL and PostgreSQL"]


def test_prose_after_a_list_is_not_tagged_as_a_list_item() -> None:
    units = split_evidence_units(LIST_SECTION)
    trailing = [unit for unit in units if unit.text.startswith("That is")]
    assert trailing and not trailing[0].is_list_item


def test_a_bullet_keeps_its_lazy_continuation() -> None:
    units = split_evidence_units("- ResumeForge is an AI resume platform that\n  Ahmed built.\n")
    assert [unit.text for unit in units if unit.is_list_item] == [
        "ResumeForge is an AI resume platform that Ahmed built."
    ]


def test_a_table_header_is_dropped_and_rows_become_statements() -> None:
    units = split_evidence_units(TABLE_SECTION)
    texts = [unit.text for unit in units]
    assert "Database: PostgreSQL with pgvector" in texts
    assert not any("Layer" in text for text in texts)


def test_a_table_data_row_becomes_a_labelled_pair() -> None:
    units = split_evidence_units("| Layer | Choice |\n| --- | --- |\n| Database | PostgreSQL |\n")
    assert [unit.text for unit in units] == ["Database: PostgreSQL"]


def test_a_header_only_table_yields_no_statements() -> None:
    """The row above the separator is a header, so it is removed with it."""
    assert split_evidence_units("| Docker |\n| --- |\n") == []


def test_a_stored_faq_prompt_attaches_to_its_answer_only() -> None:
    units = split_evidence_units(FAQ_SECTION)
    first = units[0]
    assert first.prompt == "What is his CGPA?"
    assert "9.11" in first.text
    assert all(unit.prompt is None for unit in units[1:])


def test_headings_are_not_quoted_as_statements() -> None:
    units = split_evidence_units("## Recorded certifications\n\nOne course is on record.\n")
    assert [unit.text for unit in units] == ["One course is on record."]


def test_a_decimal_is_not_a_sentence_boundary() -> None:
    """Splitting after "9.11." would orphan the recorded value."""
    units = split_evidence_units("His recorded CGPA is 9.11. The scale is not stated.\n")
    assert any("9.11." in unit.text and "scale" in unit.text for unit in units)


def test_a_personal_initial_does_not_end_a_sentence() -> None:
    """A dangling "Huerta, Digital Cloud Training ..." became quotable on its own.

    It then outranked every real statement, because a fragment with no label
    looks like a bare list entry rather than an attribute of the course.
    """
    units = split_evidence_units(
        "- **Co-instructor:** Eric E. Huerta, Digital Cloud Training - AWS Certified\n"
        "  Practitioner and AWS Certified Solutions Architect\n"
    )
    assert [unit.text for unit in units] == [
        "Co-instructor: Eric E. Huerta, Digital Cloud Training - AWS Certified "
        "Practitioner and AWS Certified Solutions Architect"
    ]
    assert units[0].is_list_item


def test_a_real_sentence_after_no_is_still_split() -> None:
    """faq.md answers "No. It is proprietary..." -- two sentences, not one."""
    units = split_evidence_units("No. It is proprietary with the licence undecided.\n")
    assert [unit.text for unit in units] == [
        "No.",
        "It is proprietary with the licence undecided.",
    ]


def test_a_guardrail_in_a_list_never_becomes_a_fact() -> None:
    """A bullet can be a directive; :func:`is_answerable_evidence` rejects it."""
    units = split_evidence_units("- He built ResumeForge.\n- Do not invent projects.\n")
    quoted = [unit.text for unit in units if is_answerable_evidence(unit.text)]
    assert quoted == ["He built ResumeForge."]


def test_front_matter_is_dropped() -> None:
    units = split_evidence_units("---\ncategory: skills\n---\n\nHe writes Python.\n")
    assert [unit.text for unit in units] == ["He writes Python."]


def test_empty_text_yields_no_units() -> None:
    assert split_evidence_units("") == []
    assert split_evidence_units("\n\n   \n") == []
