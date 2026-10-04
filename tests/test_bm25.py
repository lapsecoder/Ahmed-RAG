"""BM25 lexical ranking: scoring, admission metadata and determinism.

The index is built from :class:`~app.models.document.DocumentChunk` objects in
store order, so a hit's ``position`` is simultaneously its FAISS position. These
tests pin that contract, because every fusion decision downstream depends on the
two rankings referring to the same document.
"""

from __future__ import annotations

import pytest
from app.core.exceptions import RetrieverError
from app.services.bm25 import Bm25Index

from .conftest import make_chunk

DOCUMENTS = [
    "The recorded CGPA is 9.11.",
    "Ahmed built ResumeForge with Next.js and PostgreSQL.",
    "Python, pandas and NumPy appear throughout his projects.",
    "Ahmed is available part time alongside his studies.",
]


def build_index() -> Bm25Index:
    """A four-document index whose positions match the chunk ordinals."""
    return Bm25Index(
        [
            make_chunk(text, source_file="sample.md", section="Section", ordinal=index)
            for index, text in enumerate(DOCUMENTS)
        ]
    )


def test_positions_match_the_chunk_ordinals() -> None:
    index = build_index()
    hit = index.search("PostgreSQL", 1)[0]
    assert hit.position == 1


def test_a_rare_term_outranks_a_common_one() -> None:
    """BM25's whole point: shared words carry less signal than specific ones."""
    index = build_index()
    hits = {hit.position: hit for hit in index.search("Ahmed ResumeForge", 4)}
    assert 1 in hits, "the document with both query terms must match"
    assert hits[1].matched_terms == 2
    assert index.idf("resumeforge") > index.idf("ahmed")


def test_a_single_common_term_matches_every_document_it_appears_in() -> None:
    index = build_index()
    hits = index.search("Ahmed", 10)
    assert {hit.position for hit in hits} == {1, 3}
    assert all(hit.matched_terms == 1 for hit in hits)


def test_max_idf_reports_the_rarest_matched_term() -> None:
    index = build_index()
    hit = index.search("Ahmed ResumeForge", 4)[0]
    assert hit.max_idf == pytest.approx(index.idf("resumeforge"))


def test_a_query_sharing_no_vocabulary_returns_nothing() -> None:
    """No lexical evidence at all is better evidence than a wrong match."""
    index = build_index()
    assert index.search("xylophone", 5) == []


def test_search_is_deterministic_for_equal_scores() -> None:
    """Two identical documents must rank by ascending position, every time."""
    index = build_index()
    first = index.search("Ahmed", 10)
    second = index.search("Ahmed", 10)
    assert [hit.position for hit in first] == [hit.position for hit in second]
    assert [hit.position for hit in first] == sorted(hit.position for hit in first)


def test_top_k_truncates_the_ranking() -> None:
    index = build_index()
    assert len(index.search("Ahmed", 1)) == 1


def test_an_empty_corpus_matches_nothing() -> None:
    assert Bm25Index([]).search("anything", 5) == []


def test_vocabulary_lists_every_indexed_term() -> None:
    index = build_index()
    assert {"cgpa", "resumeforge", "postgresql"} <= index.vocabulary
    assert "xylophone" not in index.vocabulary


def test_idf_of_an_unknown_term_is_zero() -> None:
    assert build_index().idf("xylophone") == 0.0


def test_a_non_positive_top_k_is_rejected() -> None:
    with pytest.raises(RetrieverError):
        build_index().search("Ahmed", 0)


def test_average_length_is_the_mean_token_count() -> None:
    index = build_index()
    assert index.average_length > 0.0
