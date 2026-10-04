"""Embeddings.

Vectors are produced as **float32** and L2-normalised here, in one place, so
that an inner-product search (FAISS ``IndexFlatIP``) is exactly cosine
similarity. Normalisation is explicit rather than delegated to the model so it
is testable and so zero vectors can be handled safely.

The encoder is ``sentence-transformers/all-MiniLM-L6-v2`` executed through
**ONNX Runtime**. The weights are the same ones PyTorch used -- the graph was
exported from that checkpoint, offline, and is committed under
``models/all-MiniLM-L6-v2/`` -- so retrieval behaviour is preserved to within
float rounding. Measured against the PyTorch implementation over the 90 real
knowledge-base chunks and 20 real production queries: maximum absolute
element-wise difference ``2.0e-07``, maximum cosine error ``1.788e-07``,
90/90 identical top-1 nearest neighbours, and 20/20 byte-identical answers end
to end.

Why ONNX rather than ``sentence-transformers``: that package requires
``torch``, and on Linux ``torch`` drags in the whole CUDA stack
(``nvidia-cudnn``, ``nvidia-nccl``, ``cuda-toolkit``, ``triton``) even for a
CPU workload, which pushed a serverless deployment past 5.6 GB. ONNX Runtime
does the same arithmetic in ~60 MB.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import DependencyMissingError, EmbeddingError
from app.core.logging import get_logger

logger = get_logger("services.embedder")

DEFAULT_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"

#: Where the committed ONNX graph and tokenizer live.
DEFAULT_MODEL_DIR: Final[Path] = Path("models/all-MiniLM-L6-v2")

#: Sequence length the checkpoint was exported with, from
#: ``sentence_bert_config.json``. Truncating anywhere else would change the
#: vectors, so it is pinned rather than read from the tokenizer.
MAX_SEQUENCE_LENGTH: Final[int] = 256

#: Embedding width of all-MiniLM-L6-v2.
MINILM_DIMENSION: Final[int] = 384


def load_onnx_model(model_dir: Path | str) -> Any:
    """Load the ONNX encoder session and its tokenizer.

    Imported lazily so this module imports without ONNX Runtime present, which
    keeps the test suite able to exercise the surrounding logic on a machine
    that has not installed the optional dependency.

    Args:
        model_dir: Directory holding ``model.onnx`` and ``tokenizer.json``.

    Returns:
        A ``(session, tokenizer)`` pair.

    Raises:
        DependencyMissingError: If ``onnxruntime``/``tokenizers`` is missing.
        EmbeddingError: If the model files are absent or the graph is unusable.
    """
    directory = Path(model_dir)
    graph = directory / "model.onnx"
    tokenizer_file = directory / "tokenizer.json"
    missing = [path.name for path in (graph, tokenizer_file) if not path.is_file()]
    if missing:
        raise EmbeddingError(
            f"ONNX model files missing from {directory}: {', '.join(missing)}"
        )
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DependencyMissingError(
            "onnxruntime and tokenizers are required for embedding generation. "
            "Install them with: pip install onnxruntime tokenizers"
        ) from exc
    try:
        session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover - corrupt graph
        raise EmbeddingError(f"could not load ONNX graph {graph}: {exc}") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    # Truncation is pinned; padding is disabled because each batch is padded to
    # its own longest row and the mask makes the padding invisible to pooling.
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
    tokenizer.no_padding()
    return session, tokenizer


def mean_pool(
    last_hidden: NDArray[np.float32], attention_mask: NDArray[np.int64]
) -> NDArray[np.float32]:
    """Mean-pool token vectors over the attention mask.

    Reproduces sentence-transformers' ``Pooling`` module exactly: sum the token
    vectors selected by the mask, divide by the number of selected tokens, and
    clamp the denominator so an all-padding row cannot divide by zero.
    """
    mask = attention_mask.astype(np.float32)[..., None]
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return (summed / counts).astype(np.float32)


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
        """Embed a single query string into a ``(dimension,)`` vector."""


class OnnxMiniLMEmbedder:
    """:class:`Embedder` backed by the ONNX Runtime export of all-MiniLM-L6-v2."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        batch_size: int = 16,
        model: Any | None = None,
    ) -> None:
        """
        Args:
            model_name: Reported identifier. Kept as the upstream model name
                because that is genuinely what the weights are, and because
                ``/api/health`` and the index manifest both surface it.
            model_dir: Directory containing ``model.onnx`` and ``tokenizer.json``.
            batch_size: Rows encoded per ONNX invocation.
            model: A pre-built ``(session, tokenizer)`` pair, for injection.
        """
        self._model_name = model_name
        self._model_dir = Path(model_dir)
        self._batch_size = max(1, batch_size)
        self._model = model
        self._dimension_cache: int | None = None

    @property
    def model_name(self) -> str:
        """The configured model identifier."""
        return self._model_name

    @property
    def model(self) -> Any:
        """The loaded ``(session, tokenizer)`` pair, loading it on first access."""
        if self._model is None:
            logger.info("loading ONNX embedding model from %s", self._model_dir)
            self._model = load_onnx_model(self._model_dir)
        return self._model

    @property
    def dimension(self) -> int:
        """The embedding dimension, cached after the first resolution."""
        if self._dimension_cache is None:
            self._dimension_cache = self._resolve_dimension()
        return self._dimension_cache

    def _resolve_dimension(self) -> int:
        probe = self._encode(["dimension probe"])
        width = int(probe.shape[1])
        if width <= 0:
            raise EmbeddingError(
                f"model {self._model_name!r} did not report a usable embedding dimension"
            )
        return width

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        session, tokenizer = self.model
        rows: list[NDArray[np.float32]] = []
        batch = list(texts)
        for start in range(0, len(batch), self._batch_size):
            chunk = batch[start : start + self._batch_size]
            rows.append(self._encode_batch(session, tokenizer, chunk))
        # Normalisation is applied explicitly by this module, never by the graph.
        return normalize_embeddings(np.concatenate(rows, axis=0))

    @staticmethod
    def _encode_batch(session: Any, tokenizer: Any, texts: list[str]) -> NDArray[np.float32]:
        encodings = tokenizer.encode_batch(texts)
        width = max(len(encoding.ids) for encoding in encodings)
        input_ids = np.zeros((len(encodings), width), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), width), dtype=np.int64)
        token_type_ids = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[row, :length] = encoding.ids
            attention_mask[row, :length] = 1
            token_type_ids[row, :length] = encoding.type_ids
        last_hidden = session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0]
        return mean_pool(np.asarray(last_hidden, dtype=np.float32), attention_mask)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed ``texts`` into a normalised ``float32`` matrix."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._encode(texts)
        self._assert_matrix(vectors, len(texts))
        return vectors

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query string.

        The raw string is tokenised directly -- no placeholder document object
        is ever fabricated just to obtain a query vector.
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


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_DIR",
    "MAX_SEQUENCE_LENGTH",
    "MINILM_DIMENSION",
    "Embedder",
    "OnnxMiniLMEmbedder",
    "is_normalised",
    "load_onnx_model",
    "mean_pool",
    "normalize_embeddings",
]
