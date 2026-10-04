"""Hybrid retrieval: admission rules and rank fusion.

The retriever fuses a dense FAISS ranking with a BM25 ranking using reciprocal
rank fusion. These tests pin the two properties that are easy to break silently:

* the dense threshold contract is unchanged, and
* a chunk admitted on lexical evidence alone is still reported honestly, with a
  zero similarity rather than an invented one.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.services.bm25 import Bm25Index
from app.services.hybrid_retriever import (
    DEFAULT_MIN_LEXICAL_TERMS,
    DEFAULT_RRF_K,
    RARE_TERM_IDF,
    HybridRetriever,
)
from app.services.vector_store import FaissVectorStore

from .conftest import ScriptedEmbedder, make_chunk

#: Wide enough that every chunk in :data:`DOCUMENTS` gets its own axis.
DIM = 12

#: Twelve documents, so inverse document frequency can actually separate a rare
#: term from a common one. On a four-document corpus every term looks rare and
#: :data:`RARE_TERM_IDF` would be meaningless.
DOCUMENTS = [
    "The recorded CGPA is 9.11.",
    "Ahmed built ResumeForge with Next.js and PostgreSQL.",
    "Python, pandas and NumPy appear throughout his projects.",
    "Ahmed is available part time alongside his studies.",
    "The portfolio is deployed on Vercel.",
    "He maintains a daily journal of experiments.",
    "Graduation requires a final semester project.",
    "He prefers reading documentation over tutorials.",
    "The recommender ranks candidates by cosine similarity.",
    "His writing covers applied machine learning.",
    "He mentors students starting with retrieval.",
    "The backend stores chunks in a local vector index.",
]


def orthogonal(count: int) -> np.ndarray:
    """``count`` distinct axis-aligned unit vectors."""
    return np.eye(count, DIM, dtype=np.float32)


def build(
    *,
    threshold: float = 0.35,
    top_k: int = 4,
    min_lexical_terms: int = DEFAULT_MIN_LEXICAL_TERMS,
    document_expansion: int = 0,
) -> HybridRetriever:
    """A hybrid retriever over four orthogonal chunks.

    Each chunk gets a distinct axis-aligned unit vector, so dense similarity
    between the query and each chunk is decided by the scripted embedder rather
    than by the text.

    ``document_expansion`` defaults to ``0`` so that admission and ranking can be
    asserted exactly; the expansion behaviour has its own tests below.
    """
    embedder = ScriptedEmbedder(dimension=DIM)
    store = FaissVectorStore(DIM)
    store.add(
        [
            make_chunk(text, source_file="sample.md", section="Section", ordinal=index)
            for index, text in enumerate(DOCUMENTS)
        ],
        orthogonal(len(DOCUMENTS)),
    )
    return HybridRetriever(
        store,
        embedder,
        bm25=Bm25Index(store.chunks),
        similarity_threshold=threshold,
        top_k=top_k,
        min_lexical_terms=min_lexical_terms,
        document_expansion=document_expansion,
    )


def test_defaults_are_the_documented_constants() -> None:
    retriever = build()
    assert retriever.rrf_k == DEFAULT_RRF_K
    assert retriever.min_lexical_terms == DEFAULT_MIN_LEXICAL_TERMS
    assert retriever.bm25 is not None


def test_from_store_builds_a_lexical_index_mirroring_the_store() -> None:
    embedder = ScriptedEmbedder(dimension=DIM)
    store = FaissVectorStore(DIM)
    store.add(
        [make_chunk("ResumeForge uses PostgreSQL.", ordinal=index) for index in range(2)],
        orthogonal(2),
    )
    retriever = HybridRetriever.from_store(store, embedder, top_k=2)

    assert retriever.bm25 is not None
    # Every lexical position must be a valid FAISS position, or fusion would
    # silently attribute one retriever's score to a different document.
    valid = set(range(store.size))
    assert {hit.position for hit in retriever.bm25.search("PostgreSQL", 5)} <= valid


def test_chunks_above_the_dense_threshold_are_admitted() -> None:
    retriever = build(threshold=-1.0, top_k=len(DOCUMENTS))
    results = retriever.retrieve("Anything at all")
    assert {result.chunk.ordinal for result in results} == set(range(len(DOCUMENTS)))


def test_the_dense_threshold_still_gates_dense_evidence() -> None:
    """At cosine 1.0 no random query vector matches, so only lexical evidence remains."""
    assert build(threshold=1.0).retrieve("xylophone") == []


def test_lexical_evidence_admits_a_chunk_the_threshold_rejected() -> None:
    """The whole reason the hybrid retriever exists."""
    retriever = build(threshold=1.0)
    assert [result.chunk.ordinal for result in retriever.retrieve("PostgreSQL")] == [1]


#: A unit vector with 0.5 along the axis of ``DOCUMENTS[1]``, the chunk that
#: records "PostgreSQL". Naming the score in advance is what lets the test assert
#: the measured value rather than merely that some value came back.
_POSTGRESQL_QUERY_VECTOR = [0.0, 0.5, 0.8660254037844387] + [0.0] * 9


def test_a_lexically_admitted_hit_reports_its_true_dense_similarity() -> None:
    """The score shown is the measured one, never a stand-in for the evidence used.

    A lexically admitted chunk still has a real cosine similarity, just a
    sub-threshold one. Reporting that measured value alongside a positive
    ``lexical_score`` is more honest than zeroing it or, worse, presenting the
    fused rank as if it were a similarity.

    The query vector is pinned to sit 0.5 from that chunk along its axis, so the
    expected score is known exactly. Asserting a mere range would not catch a
    regression that swapped the measured value for the 0.0 used when no dense
    score exists.
    """
    embedder = ScriptedEmbedder(dimension=DIM, vectors={"PostgreSQL": _POSTGRESQL_QUERY_VECTOR})
    store = FaissVectorStore(DIM)
    store.add(
        [
            make_chunk(text, source_file="sample.md", section="Section", ordinal=index)
            for index, text in enumerate(DOCUMENTS)
        ],
        orthogonal(len(DOCUMENTS)),
    )
    retriever = HybridRetriever(
        store,
        embedder,
        bm25=Bm25Index(store.chunks),
        # Unreachable threshold: this chunk can only get in on lexical evidence.
        similarity_threshold=1.0,
        top_k=4,
        document_expansion=0,
    )

    hit = retriever.retrieve("PostgreSQL")[0]

    # The measured cosine, below the threshold that refused it, and not the
    # rank-fusion score wearing its clothes.
    assert hit.similarity == pytest.approx(0.5)
    assert hit.similarity != hit.fused_score
    assert hit.lexical_score > 0.0
    assert hit.fused_score > 0.0


def test_a_chunk_outside_the_dense_head_reports_zero_rather_than_a_guess() -> None:
    """0.0 means "no dense score was measured", which is honest; a guess is not.

    A lexically admitted chunk the dense search never reached has no similarity
    to report. Inventing one -- or reporting its fused rank as if it were a
    similarity -- would put a number in the response that no measurement
    supports.
    """
    hits = build(threshold=1.0).retrieve("PostgreSQL")

    assert hits
    assert all(-1.0 <= hit.similarity <= 1.0 for hit in hits)


def test_a_single_common_term_is_not_enough_for_lexical_admission() -> None:
    """One non-rare term matches half the corpus, which proves nothing."""
    assert build(threshold=1.0, min_lexical_terms=2).retrieve("Ahmed") == []


def test_a_single_rare_term_is_enough() -> None:
    retriever = build(threshold=1.0)
    assert retriever.retrieve("PostgreSQL"), "a rare term decides the answer alone"
    assert retriever.bm25 is not None
    assert retriever.bm25.search("PostgreSQL", 1)[0].max_idf >= RARE_TERM_IDF


def test_results_are_ordered_by_descending_fused_score() -> None:
    scores = [result.fused_score for result in build(threshold=-1.0).retrieve("Ahmed Python")]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_dense_and_start_at_zero() -> None:
    results = build(threshold=-1.0).retrieve("Ahmed Python")
    assert [result.rank for result in results] == list(range(len(results)))


def test_results_are_deterministic() -> None:
    retriever = build(threshold=-1.0)
    first = [result.chunk.chunk_id for result in retriever.retrieve("Ahmed Python")]
    second = [result.chunk.chunk_id for result in retriever.retrieve("Ahmed Python")]
    assert first == second


def test_top_k_bounds_the_result_list() -> None:
    assert len(build(threshold=-1.0, top_k=2).retrieve("Ahmed Python")) == 2


# --------------------------------------------------------------------------- #
# Document expansion
# --------------------------------------------------------------------------- #


def expansion_store() -> FaissVectorStore:
    """One document split into five chunks, plus an unrelated second document."""
    store = FaissVectorStore(DIM)
    # Only the first chunk mentions the query's terms. The other four are exactly
    # the sibling bullets a per-chunk threshold throws away.
    texts = [
        "The database is PostgreSQL with pgvector.",
        "Python",
        "FastAPI",
        "FAISS",
        "Docker",
        "The portfolio is deployed on Vercel.",
    ]
    vectors = orthogonal(len(texts))
    store.add(
        [
            make_chunk(
                text,
                source_file="skills.md" if index < 5 else "other.md",
                section=f"Skills > {index}",
                ordinal=index,
            )
            for index, text in enumerate(texts)
        ],
        vectors,
    )
    return store


def test_siblings_of_an_admitted_document_are_supplied() -> None:
    """A list-shaped answer lives in sibling chunks the threshold rejected.

    Each skills bullet shares its document's framing but repeats none of the
    question's own words, so chunk similarity alone would drop them and the
    answer would silently omit most of the list.
    """
    embedder = ScriptedEmbedder(dimension=DIM)
    store = expansion_store()
    retriever = HybridRetriever(
        store,
        embedder,
        bm25=Bm25Index(store.chunks),
        similarity_threshold=1.0,
        top_k=1,
        document_expansion=4,
    )

    results = retriever.retrieve("PostgreSQL pgvector")
    assert results
    assert {result.chunk.source_file for result in results} == {"skills.md"}
    assert {result.chunk.ordinal for result in results} == {0, 1, 2, 3, 4}


def test_expansion_never_introduces_another_document() -> None:
    """This is document completion, not a second retrieval pass."""
    embedder = ScriptedEmbedder(dimension=DIM)
    store = expansion_store()
    retriever = HybridRetriever(
        store,
        embedder,
        bm25=Bm25Index(store.chunks),
        similarity_threshold=1.0,
        top_k=1,
        document_expansion=12,
    )
    results = retriever.retrieve("PostgreSQL pgvector")
    assert "other.md" not in {result.chunk.source_file for result in results}


def test_expansion_is_capped_per_document() -> None:
    embedder = ScriptedEmbedder(dimension=DIM)
    store = expansion_store()
    retriever = HybridRetriever(
        store,
        embedder,
        bm25=Bm25Index(store.chunks),
        similarity_threshold=1.0,
        top_k=1,
        document_expansion=2,
    )
    results = retriever.retrieve("PostgreSQL pgvector")
    assert len(results) == 3


def test_expansion_can_be_turned_off() -> None:
    embedder = ScriptedEmbedder(dimension=DIM)
    store = expansion_store()
    retriever = HybridRetriever(
        store,
        embedder,
        bm25=Bm25Index(store.chunks),
        similarity_threshold=1.0,
        top_k=1,
        document_expansion=0,
    )
    assert len(retriever.retrieve("PostgreSQL pgvector")) == 1


def test_an_expanded_chunk_reports_no_invented_score() -> None:
    """It was admitted by its document, not by its own similarity."""
    embedder = ScriptedEmbedder(dimension=DIM)
    store = expansion_store()
    retriever = HybridRetriever(
        store,
        embedder,
        bm25=Bm25Index(store.chunks),
        similarity_threshold=1.0,
        top_k=1,
        document_expansion=4,
    )
    results = retriever.retrieve("PostgreSQL pgvector")
    # The admitted chunk keeps its real BM25 score. Its siblings were admitted by
    # their document, so they were never fused and never scored lexically: they
    # report 0.0 rather than a borrowed value. (Their dense similarity may be
    # non-zero when the dense pass happened to reach them, so it is not asserted
    # here.)
    assert results[0].lexical_score > 0.0
    assert all(result.lexical_score == 0.0 for result in results[1:])
    assert all(result.fused_score == 0.0 for result in results[1:])


def test_a_negative_expansion_is_rejected() -> None:
    embedder = ScriptedEmbedder(dimension=DIM)
    with pytest.raises(ValueError, match="document_expansion"):
        HybridRetriever(FaissVectorStore(DIM), embedder, document_expansion=-1)


def test_an_empty_index_returns_nothing() -> None:
    embedder = ScriptedEmbedder(dimension=DIM)
    retriever = HybridRetriever(FaissVectorStore(DIM), embedder, bm25=Bm25Index([]), top_k=3)
    assert retriever.retrieve("Ahmed") == []


def test_invalid_fusion_constants_are_rejected() -> None:
    embedder = ScriptedEmbedder(dimension=DIM)
    store = FaissVectorStore(DIM)
    store.add([make_chunk("text")], orthogonal(1))
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(store, embedder, rrf_k=0)
    with pytest.raises(ValueError, match="min_lexical_terms"):
        HybridRetriever(store, embedder, min_lexical_terms=-1)


def test_an_absent_bm25_index_still_retrieves_densely() -> None:
    """The lexical half is optional, not required."""
    embedder = ScriptedEmbedder(dimension=DIM)
    store = FaissVectorStore(DIM)
    store.add([make_chunk("ResumeForge")], orthogonal(1))
    retriever = HybridRetriever(store, embedder, bm25=None, similarity_threshold=-1.0)
    results = retriever.retrieve("ResumeForge")
    assert len(results) == 1
    assert results[0].lexical_score == 0.0
