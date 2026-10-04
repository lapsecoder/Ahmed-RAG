"""Retriever: query embedding, threshold filtering and RetrievalResult shape."""

from __future__ import annotations

import numpy as np
import pytest
from app.core.exceptions import DimensionMismatchError, RetrieverError
from app.models.document import DocumentChunk
from app.models.retrieval import RetrievalResult
from app.services.vector_store import FaissVectorStore

from .conftest import ScriptedEmbedder, make_chunk, unit_vector

DIM = 4


def store_with(vectors: list[list[float]], texts: list[str]) -> FaissVectorStore:
    chunks = [make_chunk(text, ordinal=index) for index, text in enumerate(texts)]
    store = FaissVectorStore(DIM)
    store.add(chunks, np.vstack([unit_vector(vector) for vector in vectors]))
    return store


def make_retriever(
    store: FaissVectorStore,
    *,
    threshold: float = 0.5,
    top_k: int = 5,
) -> tuple[ScriptedEmbedder, object]:
    embedder = ScriptedEmbedder(
        dimension=DIM,
        vectors={"query": unit_vector([1.0, 0.0, 0.0, 0.0])},
    )
    from app.services.retriever import Retriever

    return embedder, Retriever(store, embedder, similarity_threshold=threshold, top_k=top_k)


# --------------------------------------------------------------------------- #
# Result shape
# --------------------------------------------------------------------------- #


def test_results_pair_a_document_chunk_with_its_similarity() -> None:
    store = store_with([[1, 0, 0, 0], [0, 1, 0, 0]], ["about", "skills"])
    _, retriever = make_retriever(store, threshold=-1.0)
    results = retriever.retrieve("query")

    assert results
    for result in results:
        assert isinstance(result, RetrievalResult)
        assert isinstance(result.chunk, DocumentChunk)
        assert -1.0 <= result.similarity <= 1.0
        assert result.position >= 0
        assert result.chunk.chunk_id == f"test-{result.position:04d}"


def test_results_are_ordered_by_descending_similarity() -> None:
    store = store_with([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], ["a", "b", "c"])
    _, retriever = make_retriever(store, threshold=-1.0)
    results = retriever.retrieve("query")
    scores = [result.similarity for result in results]
    assert scores == sorted(scores, reverse=True)
    assert [result.rank for result in results] == list(range(len(results)))


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #


def test_similarity_threshold_filters_weak_matches() -> None:
    store = store_with([[1, 0, 0, 0], [0, 0, 0, 1]], ["perfect", "orthogonal"])
    _, retriever = make_retriever(store, threshold=0.5)
    results = retriever.retrieve("query")
    assert [result.chunk.text for result in results] == ["perfect"]
    assert all(result.similarity >= 0.5 for result in results)


def test_a_high_threshold_returns_nothing() -> None:
    store = store_with([[1, 0, 0, 0]], ["only"])
    embedder = ScriptedEmbedder(dimension=DIM, vectors={"query": unit_vector([0.0, 1.0, 0.0, 0.0])})
    from app.services.retriever import Retriever

    retriever = Retriever(store, embedder, similarity_threshold=0.99, top_k=5)
    assert retriever.retrieve("query") == []


def test_threshold_is_configurable_and_exposed() -> None:
    store = store_with([[1, 0, 0, 0]], ["only"])
    _, retriever = make_retriever(store, threshold=0.42)
    assert retriever.similarity_threshold == 0.42
    assert retriever.top_k == 5


def test_top_k_limits_the_number_of_results() -> None:
    store = store_with(
        [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0]],
        ["a", "b", "c"],
    )
    _, retriever = make_retriever(store, threshold=-1.0, top_k=2)
    assert len(retriever.retrieve("query")) == 2


def test_invalid_retriever_configuration_is_rejected() -> None:
    store = store_with([[1, 0, 0, 0]], ["only"])
    embedder = ScriptedEmbedder(dimension=DIM)
    from app.services.retriever import Retriever

    with pytest.raises(RetrieverError):
        Retriever(store, embedder, similarity_threshold=1.5)
    with pytest.raises(RetrieverError):
        Retriever(store, embedder, top_k=0)


# --------------------------------------------------------------------------- #
# Query embedding behaviour
# --------------------------------------------------------------------------- #


def test_query_is_embedded_verbatim() -> None:
    store = store_with([[1, 0, 0, 0]], ["about"])
    embedder, retriever = make_retriever(store, threshold=-1.0)
    retriever.retrieve("Who is Ahmed?")
    assert embedder.queries == ["Who is Ahmed?"]
    assert embedder.query_calls == 1


def test_empty_index_skips_embedding_entirely() -> None:
    store = FaissVectorStore(DIM)
    embedder, retriever = make_retriever(store)
    assert retriever.retrieve("anything") == []
    assert embedder.queries == []


def test_a_mismatched_query_dimension_is_reported() -> None:
    store = store_with([[1, 0, 0, 0]], ["about"])
    embedder = ScriptedEmbedder(dimension=DIM, vectors={"query": unit_vector([1, 0])})
    from app.services.retriever import Retriever

    retriever = Retriever(store, embedder, similarity_threshold=0.0)
    with pytest.raises(DimensionMismatchError):
        retriever.retrieve("query")


def test_blank_queries_are_rejected() -> None:
    store = store_with([[1, 0, 0, 0]], ["about"])
    _, retriever = make_retriever(store)
    with pytest.raises(RetrieverError):
        retriever.retrieve("   ")


def test_retrieve_many_returns_one_result_list_per_query() -> None:
    store = store_with([[1, 0, 0, 0]], ["about"])
    _, retriever = make_retriever(store, threshold=-1.0)
    batches = retriever.retrieve_many(["query", "query"])
    assert len(batches) == 2
    assert all(batch for batch in batches)
