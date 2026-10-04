"""Embeddings: normalisation, float32 output, zero vectors, mocked model loading."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from app.core.exceptions import DependencyMissingError, EmbeddingError
from app.services.embedder import (
    DEFAULT_MODEL,
    Embedder,
    SentenceTransformerEmbedder,
    is_normalised,
    load_sentence_transformer,
    normalize_embeddings,
)

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_normalize_produces_unit_rows() -> None:
    result = normalize_embeddings([[3.0, 4.0], [1.0, 0.0]])
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(result[1], [1.0, 0.0], atol=1e-6)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_normalize_accepts_a_single_vector() -> None:
    result = normalize_embeddings([3.0, 4.0])
    assert result.shape == (1, 2)
    assert result.dtype == np.float32


def test_normalize_handles_a_zero_vector_without_producing_nan() -> None:
    result = normalize_embeddings([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[0], np.zeros(3, dtype=np.float32))
    assert np.isclose(float(np.linalg.norm(result[1])), 1.0)


def test_normalize_scrubs_non_finite_values() -> None:
    result = normalize_embeddings([[np.nan, 1.0], [np.inf, 0.0], [-np.inf, 2.0]])
    assert np.isfinite(result).all()


def test_normalize_rejects_non_matrix_input() -> None:
    with pytest.raises(EmbeddingError):
        normalize_embeddings(np.zeros((2, 2, 2)))


def test_normalize_is_idempotent() -> None:
    once = normalize_embeddings([[2.0, 0.0, 0.0]])
    twice = normalize_embeddings(once)
    np.testing.assert_allclose(once, twice, atol=1e-7)


def test_is_normalised_helper() -> None:
    assert is_normalised(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert is_normalised(np.zeros(3, dtype=np.float32))
    assert not is_normalised(np.array([3.0, 4.0, 0.0], dtype=np.float32))
    assert not is_normalised(np.zeros(0, dtype=np.float32))


# --------------------------------------------------------------------------- #
# SentenceTransformerEmbedder with a mocked model
# --------------------------------------------------------------------------- #


def test_embedder_uses_the_mocked_sentence_transformer(
    fake_sentence_transformer: list[object],
) -> None:
    embedder = SentenceTransformerEmbedder()
    matrix = embedder.embed_documents(["alpha", "beta"])
    assert matrix.shape == (2, 4)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    assert len(fake_sentence_transformer) == 1
    assert fake_sentence_transformer[0].model_name == DEFAULT_MODEL


def test_embedder_asks_the_model_not_to_normalise(
    fake_sentence_transformer: list[object],
) -> None:
    embedder = SentenceTransformerEmbedder()
    embedder.embed_documents(["alpha"])
    call = fake_sentence_transformer[0].encode_calls[0]
    assert call["normalize_embeddings"] is False
    assert call["convert_to_numpy"] is True
    assert call["show_progress_bar"] is False


def test_embedder_query_is_normalised_and_float32(
    fake_sentence_transformer: list[object],
) -> None:
    embedder = SentenceTransformerEmbedder()
    vector = embedder.embed_query("what is ahmed's experience?")
    assert vector.shape == (embedder.dimension,)
    assert vector.dtype == np.float32
    assert is_normalised(vector)


def test_embedder_never_builds_a_fake_document_to_embed_a_query(
    fake_sentence_transformer: list[object],
) -> None:
    """The raw query string is passed straight through -- nothing is fabricated."""
    embedder = SentenceTransformerEmbedder()
    embedder.embed_query("Tell me about the projects")
    call = fake_sentence_transformer[0].encode_calls[0]
    assert call["texts"] == ["Tell me about the projects"]
    assert fake_sentence_transformer[0].model_name == DEFAULT_MODEL


def test_dimension_comes_from_the_real_model(
    fake_sentence_transformer: list[object],
) -> None:
    embedder = SentenceTransformerEmbedder()
    assert embedder.dimension == 4
    assert embedder.dimension == 4  # cached, no second load


def test_zero_vectors_from_the_model_survive_normalisation(
    fake_sentence_transformer: list[object],
) -> None:
    embedder = SentenceTransformerEmbedder()
    embedder.model.override = np.zeros((2, 4), dtype=np.float32)
    matrix = embedder.embed_documents(["a", "b"])
    assert np.isfinite(matrix).all()
    np.testing.assert_array_equal(matrix, np.zeros((2, 4), dtype=np.float32))


def test_empty_batch_returns_an_empty_matrix(
    fake_sentence_transformer: list[object],
) -> None:
    embedder = SentenceTransformerEmbedder()
    matrix = embedder.embed_documents([])
    assert matrix.shape == (0, embedder.dimension)
    assert matrix.dtype == np.float32


def test_embedder_rejects_an_empty_query(fake_sentence_transformer: list[object]) -> None:
    embedder = SentenceTransformerEmbedder()
    with pytest.raises(EmbeddingError):
        embedder.embed_query("   ")


def test_dimension_mismatch_between_model_and_query_is_an_error(
    fake_sentence_transformer: list[object],
) -> None:
    embedder = SentenceTransformerEmbedder()
    embedder.model.override = np.ones((1, 7), dtype=np.float32)
    with pytest.raises(EmbeddingError):
        embedder.embed_query("hello")


def test_embedder_satisfies_the_protocol(
    fake_sentence_transformer: list[object],
) -> None:
    assert isinstance(SentenceTransformerEmbedder(), Embedder)


def test_a_missing_dependency_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(DependencyMissingError):
        load_sentence_transformer(DEFAULT_MODEL)


def test_loader_passes_the_model_name_through(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class Recorder:
        def __init__(self, name: str) -> None:
            created.append(name)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = Recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    load_sentence_transformer("some/model")
    assert created == ["some/model"]


def test_injected_model_is_used_without_loading_another(
    fake_sentence_transformer: list[object],
    fake_model: object,
) -> None:
    embedder = SentenceTransformerEmbedder(model=fake_model)
    assert embedder.model is fake_model
    assert fake_sentence_transformer == []
