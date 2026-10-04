"""FAISS store: indexing, position -> DocumentChunk mapping, dimension checks."""

from __future__ import annotations

import json

import faiss
import numpy as np
import pytest
from app.core.exceptions import DimensionMismatchError, VectorStoreError
from app.models.document import DocumentChunk
from app.models.embedding import Embedding
from app.services.vector_store import CHUNKS_FILE, FaissVectorStore

from .conftest import ScriptedEmbedder, make_chunk, unit_vector

DIM = 6


def build_store(chunks: list[DocumentChunk], vectors: np.ndarray) -> FaissVectorStore:
    store = FaissVectorStore(DIM)
    store.add(chunks, vectors)
    return store


# --------------------------------------------------------------------------- #
# Index construction
# --------------------------------------------------------------------------- #


def test_index_is_created_with_the_requested_dimension() -> None:
    store = FaissVectorStore(384)
    assert store.dimension == 384
    assert isinstance(store._index, faiss.IndexFlatIP)
    assert store.is_empty
    assert len(store) == 0


def test_index_is_sized_from_the_real_embedding_dimension() -> None:
    embedder = ScriptedEmbedder(dimension=17)
    store = FaissVectorStore(embedder.dimension)
    vectors = embedder.embed_documents(["a", "b"])
    store.add([make_chunk("a", ordinal=0), make_chunk("b", ordinal=1)], vectors)
    assert store.dimension == 17
    assert store._index.d == 17


def test_invalid_dimension_is_rejected() -> None:
    with pytest.raises(VectorStoreError):
        FaissVectorStore(0)
    with pytest.raises(VectorStoreError):
        FaissVectorStore(-5)


def test_add_returns_the_new_size() -> None:
    store = FaissVectorStore(DIM)
    vectors = np.vstack([unit_vector([1, 0, 0, 0, 0, 0]), unit_vector([0, 1, 0, 0, 0, 0])])
    assert store.add([make_chunk("a"), make_chunk("b", ordinal=1)], vectors) == 2


def test_empty_add_is_a_noop() -> None:
    store = FaissVectorStore(DIM)
    assert store.add([], np.zeros((0, DIM), dtype=np.float32)) == 0
    assert store.is_empty


# --------------------------------------------------------------------------- #
# Metadata mapping
# --------------------------------------------------------------------------- #


def test_metadata_maps_faiss_positions_to_document_chunks() -> None:
    chunks = [make_chunk("alpha", ordinal=0), make_chunk("beta", ordinal=1)]
    vectors = np.vstack([unit_vector([1, 0, 0, 0, 0, 0]), unit_vector([0, 1, 0, 0, 0, 0])])
    store = build_store(chunks, vectors)

    for position, chunk in enumerate(chunks):
        stored = store.chunk_at(position)
        assert isinstance(stored, DocumentChunk)
        assert stored == chunk
        assert stored.chunk_id == chunk.chunk_id
    assert store.chunks == tuple(chunks)


def test_store_refuses_to_index_embeddings_as_metadata() -> None:
    """FAISS positions must never map to Embedding objects."""
    embeddings = [Embedding(vector=unit_vector([1, 0, 0, 0, 0, 0]), chunk_id="c0")]
    vectors = np.vstack([embedding.vector for embedding in embeddings])
    store = FaissVectorStore(DIM)
    with pytest.raises(VectorStoreError, match="DocumentChunk"):
        store.add(embeddings, vectors)  # type: ignore[arg-type]
    assert store.is_empty


def test_store_never_exposes_embeddings_from_its_metadata() -> None:
    store = build_store([make_chunk("alpha")], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])]))
    for position in range(store.size):
        assert not isinstance(store.chunk_at(position), Embedding)
        assert isinstance(store.chunk_at(position), DocumentChunk)


def test_chunk_at_rejects_out_of_range_positions() -> None:
    store = build_store([make_chunk("alpha")], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])]))
    with pytest.raises(VectorStoreError):
        store.chunk_at(1)
    with pytest.raises(VectorStoreError):
        store.chunk_at(-1)


def test_row_count_must_match_chunk_count() -> None:
    store = FaissVectorStore(DIM)
    with pytest.raises(VectorStoreError, match="embedding row"):
        store.add([make_chunk("a")], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])] * 2))


def test_embeddings_without_chunks_are_rejected() -> None:
    store = FaissVectorStore(DIM)
    with pytest.raises(VectorStoreError):
        store.add([], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])]))


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def test_search_returns_cosine_similarities_for_normalised_vectors() -> None:
    base = unit_vector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    near = unit_vector([0.99, 0.14, 0.0, 0.0, 0.0, 0.0])
    far = unit_vector([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    store = build_store(
        [make_chunk("near", ordinal=0), make_chunk("far", ordinal=1)],
        np.vstack([near, far]),
    )

    hits = store.search(base, top_k=2)
    assert [hit.position for hit in hits] == [0, 1]
    assert hits[0].similarity == pytest.approx(float(np.dot(base, near)), abs=1e-5)
    assert hits[0].similarity > hits[1].similarity
    assert store.chunk_at(hits[0].position).text == "near"


def test_search_on_an_empty_index_returns_nothing() -> None:
    store = FaissVectorStore(DIM)
    assert store.search(unit_vector([1, 0, 0, 0, 0, 0]), top_k=5) == []


def test_search_caps_k_at_the_index_size() -> None:
    store = build_store([make_chunk("only")], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])]))
    assert len(store.search(unit_vector([1, 0, 0, 0, 0, 0]), top_k=50)) == 1


def test_search_rejects_a_wrongly_sized_query() -> None:
    store = build_store([make_chunk("a")], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])]))
    with pytest.raises(DimensionMismatchError, match="query dimension"):
        store.search(unit_vector([1, 0, 0]), top_k=1)


def test_add_rejects_wrongly_sized_vectors() -> None:
    store = FaissVectorStore(DIM)
    with pytest.raises(DimensionMismatchError, match="index was built for dimension"):
        store.add([make_chunk("a")], np.vstack([unit_vector([1, 0, 0])]))


def test_search_requires_a_positive_top_k() -> None:
    store = build_store([make_chunk("a")], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])]))
    with pytest.raises(VectorStoreError):
        store.search(unit_vector([1, 0, 0, 0, 0, 0]), top_k=0)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_and_load_round_trip(tmp_path: object) -> None:
    chunks = [make_chunk("alpha", ordinal=0), make_chunk("beta", ordinal=1)]
    store = build_store(
        chunks,
        np.vstack([unit_vector([1, 0, 0, 0, 0, 0]), unit_vector([0, 1, 0, 0, 0, 0])]),
    )
    store.save(tmp_path)

    loaded = FaissVectorStore.load(tmp_path)
    assert loaded.size == store.size
    assert loaded.dimension == store.dimension
    assert loaded.chunks == tuple(chunks)
    hits = loaded.search(unit_vector([1, 0, 0, 0, 0, 0]), top_k=1)
    assert loaded.chunk_at(hits[0].position).text == "alpha"


def test_load_reports_a_missing_index(tmp_path: object) -> None:
    with pytest.raises(VectorStoreError, match="no saved index"):
        FaissVectorStore.load(tmp_path)


def test_load_rejects_inconsistent_metadata(tmp_path: object) -> None:
    store = build_store([make_chunk("alpha")], np.vstack([unit_vector([1, 0, 0, 0, 0, 0])]))
    store.save(tmp_path)
    (tmp_path / CHUNKS_FILE).write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(VectorStoreError, match="does not match the index size"):
        FaissVectorStore.load(tmp_path)
