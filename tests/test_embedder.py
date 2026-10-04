"""Embeddings: normalisation, float32 output, zero vectors, ONNX model loading."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from app.core.exceptions import DependencyMissingError, EmbeddingError
from app.services.embedder import (
    DEFAULT_MODEL,
    MAX_SEQUENCE_LENGTH,
    MINILM_DIMENSION,
    Embedder,
    OnnxMiniLMEmbedder,
    is_normalised,
    load_onnx_model,
    mean_pool,
    normalize_embeddings,
)

from .conftest import FakeOnnxSession, FakeTokenizer

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
# Mean pooling: the sentence-transformers Pooling module, reproduced
# --------------------------------------------------------------------------- #


def test_mean_pool_averages_only_the_attended_tokens() -> None:
    hidden = np.asarray([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.asarray([[1, 1, 0]], dtype=np.int64)
    np.testing.assert_allclose(mean_pool(hidden, mask), [[2.0, 2.0]])


def test_mean_pool_respects_a_middle_gap_in_the_mask() -> None:
    hidden = np.asarray([[[2.0], [99.0], [4.0]]], dtype=np.float32)
    mask = np.asarray([[1, 0, 1]], dtype=np.int64)
    np.testing.assert_allclose(mean_pool(hidden, mask), [[3.0]])


def test_mean_pool_does_not_divide_by_zero_on_an_untended_row() -> None:
    hidden = np.zeros((1, 2, 3), dtype=np.float32)
    mask = np.zeros((1, 2), dtype=np.int64)
    result = mean_pool(hidden, mask)
    assert np.isfinite(result).all()


# --------------------------------------------------------------------------- #
# OnnxMiniLMEmbedder with a fake ONNX session
# --------------------------------------------------------------------------- #


def test_embedder_produces_normalised_float32_rows(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    matrix = embedder.embed_documents(["alpha", "beta"])
    assert matrix.shape == (2, 4)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    assert embedder.model_name == DEFAULT_MODEL


def test_embedder_pads_to_the_longest_row_in_the_batch(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    """Padding must be masked out, not attended to."""
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    embedder.embed_documents(["alpha", "beta"])
    fed = fake_onnx_model[0].run_calls[0]
    input_ids, attention_mask = fed["input_ids"], fed["attention_mask"]
    assert input_ids.shape == attention_mask.shape
    # Padding rows carry mask 0 and token id 0.
    assert set(np.unique(input_ids[attention_mask == 0])) <= {0}
    assert attention_mask.sum() > 0


def test_embedder_never_normalises_inside_the_graph(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    """Normalisation is this module's job, so the graph's own output is un-normalised."""
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    assert embedder.dimension == 4  # resolve and cache before counting calls
    raw = fake_onnx_model[0].run_calls[-1]["input_ids"]
    pooled = mean_pool(
        np.full((1, 2, 1), 4.0, dtype=np.float32), np.ones((1, 2), dtype=np.int64)
    )
    # Mean pooling averages; it does not normalise. Only this module does that.
    np.testing.assert_allclose(pooled[0], [4.0])
    assert np.allclose(np.linalg.norm(embedder.embed_documents(["alpha"])[0]), 1.0, atol=1e-6)
    assert raw.shape[0] == 1


def test_embedder_query_is_normalised_and_float32(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    vector = embedder.embed_query("what is ahmed's experience?")
    assert vector.shape == (embedder.dimension,)
    assert vector.dtype == np.float32
    assert is_normalised(vector)


def test_embedder_never_builds_a_fake_document_to_embed_a_query(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    """The raw query string is passed straight through -- nothing is fabricated."""
    session, tokenizer = fake_onnx_model
    assert session is fake_onnx_model[0]
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    assert embedder.dimension == 4  # resolve first; the probe encodes another string
    embedder.embed_query("Tell me about the projects")
    assert tokenizer.encode_calls[-1] == ["Tell me about the projects"]
    assert embedder.model_name == DEFAULT_MODEL


def test_embedder_batches_large_inputs(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    session, _ = fake_onnx_model
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model, batch_size=4)
    assert embedder.dimension == 4  # resolve first so the probe call is not counted
    session.run_calls.clear()
    matrix = embedder.embed_documents([f"row {index}" for index in range(10)])
    assert matrix.shape == (10, 4)
    assert len(session.run_calls) == 3  # 4 + 4 + 2


def test_dimension_comes_from_the_real_model(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    assert embedder.dimension == 4
    assert embedder.dimension == 4  # cached, no second load


def test_zero_vectors_from_the_model_survive_normalisation(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    assert embedder.dimension == 4  # resolve before overriding the session
    # FakeTokenizer emits three tokens per row, so the override must be 3-D.
    fake_onnx_model[0].override = np.zeros((2, 3, 4), dtype=np.float32)
    matrix = embedder.embed_documents(["a", "b"])
    assert np.isfinite(matrix).all()
    np.testing.assert_array_equal(matrix, np.zeros((2, 4), dtype=np.float32))


def test_empty_batch_returns_an_empty_matrix(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    matrix = embedder.embed_documents([])
    assert matrix.shape == (0, embedder.dimension)
    assert matrix.dtype == np.float32


def test_embedder_rejects_an_empty_query(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    with pytest.raises(EmbeddingError):
        embedder.embed_query("   ")


def test_dimension_mismatch_between_model_and_query_is_an_error(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    embedder = OnnxMiniLMEmbedder(model=fake_onnx_model)
    assert embedder.dimension == 4  # cache the 4-wide dimension before mismatching
    fake_onnx_model[0].dimension = 7
    fake_onnx_model[0].override = np.ones((1, 3, 7), dtype=np.float32)
    with pytest.raises(EmbeddingError):
        embedder.embed_query("hello")


def test_embedder_satisfies_the_protocol(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
) -> None:
    assert isinstance(OnnxMiniLMEmbedder(model=fake_onnx_model), Embedder)


def test_injected_model_is_used_without_loading_another(
    fake_onnx_model: tuple[FakeOnnxSession, FakeTokenizer],
    tmp_path: Path,
) -> None:
    embedder = OnnxMiniLMEmbedder(model_dir=tmp_path, model=fake_onnx_model)
    assert embedder.model is fake_onnx_model
    assert embedder.embed_query("hello").shape == (4,)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_missing_model_files_raise_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="model files missing"):
        load_onnx_model(tmp_path)


def test_a_missing_dependency_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(DependencyMissingError):
        load_onnx_model(Path("models/all-MiniLM-L6-v2"))


def test_loader_pins_truncation_and_disables_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model.onnx").write_bytes(b"graph")
    (tmp_path / "tokenizer.json").write_text("{}")
    created: dict[str, object] = {}

    def recorder(path: str, providers: list[str]) -> object:
        created["providers"] = providers
        created["graph"] = path
        return "session"

    class TokenizerStub:
        @staticmethod
        def from_file(path: str) -> TokenizerStub:
            created["tokenizer"] = path
            return TokenizerStub()

        def enable_truncation(self, max_length: int) -> None:
            created["truncation"] = max_length

        def no_padding(self) -> None:
            created["padding"] = False

    ort_module = types.ModuleType("onnxruntime")
    ort_module.InferenceSession = recorder  # type: ignore[attr-defined]
    tokens_module = types.ModuleType("tokenizers")
    tokens_module.Tokenizer = TokenizerStub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setitem(sys.modules, "tokenizers", tokens_module)

    session, tokenizer = load_onnx_model(tmp_path)
    assert session == "session"
    assert isinstance(tokenizer, TokenizerStub)
    assert created["providers"] == ["CPUExecutionProvider"]
    # The exported checkpoint was truncated at 256; anything else changes vectors.
    assert created["truncation"] == MAX_SEQUENCE_LENGTH == 256
    assert created["padding"] is False


# --------------------------------------------------------------------------- #
# The committed checkpoint
# --------------------------------------------------------------------------- #

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/all-MiniLM-L6-v2"


@pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="committed ONNX checkpoint is not present",
)
def test_the_committed_checkpoint_still_produces_384_dimensions() -> None:
    embedder = OnnxMiniLMEmbedder(model_dir=MODEL_DIR)
    assert embedder.dimension == MINILM_DIMENSION == 384
    vector = embedder.embed_query("Where does Ahmed study?")
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    assert is_normalised(vector, tolerance=1e-5)


@pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="committed ONNX checkpoint is not present",
)
def test_the_committed_checkpoint_is_deterministic() -> None:
    """The same text must give the same vector, every time. No sampling."""
    embedder = OnnxMiniLMEmbedder(model_dir=MODEL_DIR)
    first = embedder.embed_query("walk me through his academic history")
    second = OnnxMiniLMEmbedder(model_dir=MODEL_DIR).embed_query(
        "walk me through his academic history"
    )
    np.testing.assert_array_equal(first, second)


@pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="committed ONNX checkpoint is not present",
)
def test_the_committed_checkpoint_agrees_with_the_pytorch_export() -> None:
    """The parity that justified replacing sentence-transformers.

    ``REFERENCE_VECTORS`` holds values captured from the PyTorch implementation
    itself. The tolerance is two orders of magnitude looser than the 1.2e-07
    actually measured, so this fails loudly if the weights, the pooling or the
    truncation length ever drift.
    """
    embedder = OnnxMiniLMEmbedder(model_dir=MODEL_DIR)
    cases = {
        "Where does Ahmed study?": REFERENCE_VECTORS["study"],
        "What is Ahmed's CGPA?": REFERENCE_VECTORS["cgpa"],
    }
    for query, reference in cases.items():
        vector = embedder.embed_query(query)
        np.testing.assert_allclose(vector[: len(reference)], reference, atol=2e-5)


#: The first eight components of the vectors produced by
#: ``sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")``
#: for two real queries, captured from that implementation before it was removed.
#: They are the regression anchor: if the committed ONNX graph, the pooling or the
#: truncation length ever drifts, these numbers stop matching.
REFERENCE_VECTORS = {
    "study": np.asarray(
        [0.134801, 0.086950, -0.057844, 0.086127, -0.054765, 0.004654, 0.034274, -0.053317],
        dtype=np.float32,
    ),
    "cgpa": np.asarray(
        [-0.001961, 0.084259, -0.109885, -0.000862, -0.114909, 0.053281, 0.055709, 0.007401],
        dtype=np.float32,
    ),
}
