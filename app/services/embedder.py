"""Embeddings.

Vectors are produced as **float32** and L2-normalised here, in one place, so
that an inner-product search (FAISS ``IndexFlatIP``) is exactly cosine
similarity. Normalisation is explicit rather than delegated to the model so it
is testable and so zero vectors can be handled safely.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import DependencyMissingError, EmbeddingError
from app.core.logging import get_logger

logger = get_logger("services.embedder")

DEFAULT_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"


def load_sentence_transformer(model_name: str) -> Any:
    """Import and instantiate ``SentenceTransformer``.

    Imported lazily so the module (and the test suite) works without the
    heavyweight dependency installed.

    Raises:
        DependencyMissingError: If ``sentence-transformers`` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DependencyMissingError(
            "sentence-transformers is required for embedding generation. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def normalize_embeddings(matrix: Any) -> NDArray[np.float32]:
    """L2-normalise rows into a contiguous ``float32`` matrix.

    Zero rows are returned unchanged (as exact zero vectors) instead of
    producing NaN/Inf, which would corrupt the index. A zero vector scores 0
    against everything, so it simply never clears the similarity threshold.

    Raises:
        EmbeddingError: If the input cannot be interpreted as a 2-D matrix.
    """
    try:
        array = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(f"cannot coerce embeddings to float32: {exc}") from exc

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise EmbeddingError(f"expected a 2-D matrix of embeddings, got shape {array.shape}")
    if array.size == 0:
        return np.zeros((0, array.shape[-1] if array.ndim == 2 else 0), dtype=np.float32)

    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, np.float32(1.0))
    normalised = array / safe
    normalised = np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(normalised, dtype=np.float32)


def is_normalised(vector: NDArray[np.float32], tolerance: float = 1e-5) -> bool:
    """True when the row norm is 1 (or the row is exactly zero)."""
    if vector.size == 0:
        return False
    norm = float(np.linalg.norm(vector))
    return abs(norm - 1.0) <= tolerance or norm == 0.0


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding surface used by the rest of the application."""

    @property
    def model_name(self) -> str:
        """Identifier of the model that produces the vectors."""

    @property
    def dimension(self) -> int:
        """Length of each produced vector."""

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed a batch of documents into a ``(n, dimension)`` float32 matrix."""

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query string into a ``(dimension,)`` float32 vector."""


class SentenceTransformerEmbedder:
    """:class:`Embedder` backed by a local ``sentence-transformers`` model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        model: Any | None = None,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model = model
        self._batch_size = max(1, batch_size)
        self._device = device
        self._dimension_cache: int | None = None

    @property
    def model_name(self) -> str:
        """The configured model identifier."""
        return self._model_name

    @property
    def model(self) -> Any:
        """The loaded model, loading it on first access."""
        if self._model is None:
            logger.info("loading embedding model %s", self._model_name)
            self._model = load_sentence_transformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """The real embedding dimension reported by the loaded model."""
        if self._dimension_cache is None:
            self._dimension_cache = self._resolve_dimension()
        return self._dimension_cache

    def _resolve_dimension(self) -> int:
        model = self.model
        getter = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            value = getter()
            if isinstance(value, int) and value > 0:
                return value
            raise EmbeddingError(
                f"model {self._model_name!r} reported an invalid embedding dimension: {value!r}"
            )
        probe = self._encode(["dimension probe"])
        if probe.shape[1] == 0:
            raise EmbeddingError(
                f"model {self._model_name!r} did not report a usable embedding dimension"
            )
        return int(probe.shape[1])

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        kwargs: dict[str, Any] = {
            "batch_size": self._batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": False,
            # Normalisation is applied explicitly by this module.
            "normalize_embeddings": False,
        }
        if self._device is not None:
            kwargs["device"] = self._device
        raw = self.model.encode(list(texts), **kwargs)
        return normalize_embeddings(raw)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed ``texts`` into a normalised ``float32`` matrix."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._encode(texts)
        self._assert_matrix(vectors, len(texts))
        return vectors

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query string.

        The raw string is sent to the model directly -- no placeholder document
        object is ever fabricated just to obtain a query vector.
        """
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("query text must be a non-empty string")
        vector = self._encode([text])[0]
        if vector.shape[0] != self.dimension:
            raise EmbeddingError(
                f"model returned dimension {vector.shape[0]}, expected {self.dimension}"
            )
        return np.ascontiguousarray(vector, dtype=np.float32)

    def _assert_matrix(self, vectors: NDArray[np.float32], expected_rows: int) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise EmbeddingError(
                f"expected {expected_rows} embedding row(s), got shape {vectors.shape}"
            )
        if vectors.shape[1] != self.dimension:
            raise EmbeddingError(
                f"embedding dimension mismatch: got {vectors.shape[1]}, expected {self.dimension}"
            )
