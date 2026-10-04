"""FAISS vector store.

The index is ``IndexFlatIP`` over L2-normalised vectors, so a search score is a
cosine similarity.

The metadata mapping is strictly ``FAISS position -> DocumentChunk``. This class
refuses anything else at runtime, which makes it impossible to accidentally
index (and later return) an ``Embedding`` object.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

import faiss
import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import DimensionMismatchError, VectorStoreError
from app.core.logging import get_logger
from app.models.document import DocumentChunk
from app.models.index import IndexManifest

logger = get_logger("services.vector_store")

CHUNKS_FILE: Final[str] = "chunks.json"
INDEX_FILE: Final[str] = "index.faiss"
MANIFEST_FILE: Final[str] = "manifest.json"


class SearchHit(NamedTuple):
    """One raw FAISS result, before thresholding."""

    position: int
    similarity: float


class FaissVectorStore:
    """An ``IndexFlatIP`` index plus its ``DocumentChunk`` metadata."""

    def __init__(self, dimension: int, *, index: Any | None = None) -> None:
        if dimension <= 0:
            raise VectorStoreError(f"dimension must be positive, got {dimension}")
        self._dimension = int(dimension)
        self._index = faiss.IndexFlatIP(self._dimension) if index is None else index
        self._chunks: list[DocumentChunk] = []
        self._manifest: IndexManifest | None = None
        self._validate_index_type()

    def _validate_index_type(self) -> None:
        """Guard the invariants the rest of the class relies on.

        Raises:
            VectorStoreError: If the index is not an inner-product index or its
                dimension disagrees with the configured one.
        """
        if not isinstance(self._index, faiss.IndexFlatIP):
            raise VectorStoreError(
                f"expected a faiss.IndexFlatIP, got {type(self._index).__name__}"
            )
        if int(self._index.d) != self._dimension:
            raise VectorStoreError(
                f"index dimension {self._index.d} does not match the configured "
                f"dimension {self._dimension}"
            )

    # ------------------------------------------------------------------ props
    @property
    def dimension(self) -> int:
        """Embedding dimension the index was sized for."""
        return self._dimension

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        return int(self._index.ntotal)

    @property
    def is_empty(self) -> bool:
        """True when nothing has been indexed."""
        return self.size == 0

    @property
    def manifest(self) -> IndexManifest | None:
        """Provenance of the loaded or last-saved index, when available."""
        return self._manifest

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------ write
    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: NDArray[np.float32] | Any,
    ) -> int:
        """Add ``chunks`` and their ``embeddings`` to the index.

        Args:
            chunks: Metadata, positionally aligned with ``embeddings``.
            embeddings: ``(n, dimension)`` normalised vectors.

        Returns:
            The new number of indexed vectors.

        Raises:
            VectorStoreError: If the metadata is not ``DocumentChunk``, or the
                row counts disagree.
            DimensionMismatchError: If the matrix width differs from the index.
        """
        if not chunks:
            if embeddings is not None and np.asarray(embeddings).size:
                raise VectorStoreError("embeddings were supplied without any chunks")
            return self.size

        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise VectorStoreError(
                    "index metadata must be DocumentChunk objects; "
                    f"position {position} is {type(chunk).__name__}. "
                    "FAISS positions must never map to embeddings."
                )

        matrix = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise VectorStoreError(f"embeddings must be 2-D, got shape {matrix.shape}")
        if matrix.shape[0] != len(chunks):
            raise VectorStoreError(
                f"got {matrix.shape[0]} embedding row(s) for {len(chunks)} chunk(s)"
            )
        if matrix.shape[1] != self._dimension:
            raise DimensionMismatchError(
                f"index was built for dimension {self._dimension}, "
                f"received vectors of dimension {matrix.shape[1]}"
            )

        self._index.add(matrix)
        self._chunks.extend(chunks)
        logger.info("indexed %d chunk(s); index now holds %d", len(chunks), self.size)
        return self.size

    # ------------------------------------------------------------------- read
    def chunk_at(self, position: int) -> DocumentChunk:
        """Return the chunk stored at a FAISS position.

        Raises:
            VectorStoreError: If the position is out of range.
        """
        if not 0 <= position < len(self._chunks):
            raise VectorStoreError(
                f"FAISS position {position} is out of range (index holds {len(self._chunks)})"
            )
        chunk = self._chunks[position]
        if not isinstance(chunk, DocumentChunk):  # pragma: no cover - defensive
            raise VectorStoreError(f"metadata at position {position} is not a DocumentChunk")
        return chunk

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """All indexed chunks, in FAISS position order."""
        return tuple(self._chunks)

    def search(
        self,
        query_vector: NDArray[np.float32] | Any,
        top_k: int,
    ) -> list[SearchHit]:
        """Search the index for the ``top_k`` nearest neighbours.

        Args:
            query_vector: A single normalised vector of ``dimension`` length.
            top_k: Maximum number of hits to return.

        Returns:
            Hits sorted by descending similarity.

        Raises:
            VectorStoreError: On a non-positive ``top_k``.
            DimensionMismatchError: If the query length differs from the index.
        """
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be positive, got {top_k}")
        if self.is_empty:
            return []

        query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32)).reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise DimensionMismatchError(
                f"query dimension {query.shape[1]} does not match index dimension {self._dimension}"
            )

        k = min(top_k, self.size)
        scores, positions = self._index.search(query, k)
        hits: list[SearchHit] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            hits.append(SearchHit(position=int(position), similarity=float(score)))
        hits.sort(key=lambda hit: hit.similarity, reverse=True)
        return hits

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path | str, *, manifest: IndexManifest | None = None) -> None:
        """Persist the index, its metadata and its manifest to ``directory``.

        Every file is written to a temporary sibling first. Only once **all**
        writes have succeeded are the temporary files moved onto the real paths
        with :func:`os.replace`, which is atomic per file. A failure part-way
        through therefore leaves a previously saved, valid index completely
        untouched instead of a half-written mixture.

        Args:
            directory: Destination; created when missing.
            manifest: Provenance record describing this index. Omitting it
                produces an index that is reloadable but will be treated as
                stale, because nothing proves what it was built from.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        if manifest is not None:
            _validate_manifest(manifest, self.size, self._dimension)

        index_tmp = target / f"{INDEX_FILE}.tmp"
        chunks_tmp = target / f"{CHUNKS_FILE}.tmp"
        manifest_tmp = target / f"{MANIFEST_FILE}.tmp"
        payload = [chunk.model_dump(mode="json") for chunk in self._chunks]

        try:
            faiss.write_index(self._index, str(index_tmp))
            _write_json(chunks_tmp, payload)
            if manifest is not None:
                _write_json(manifest_tmp, manifest.model_dump(mode="json"))

            # Nothing above touched the live files, so a failure here still
            # leaves the previous index in place.
            os.replace(index_tmp, target / INDEX_FILE)
            os.replace(chunks_tmp, target / CHUNKS_FILE)
            if manifest is not None:
                os.replace(manifest_tmp, target / MANIFEST_FILE)
        finally:
            for temporary in (index_tmp, chunks_tmp, manifest_tmp):
                temporary.unlink(missing_ok=True)

        self._manifest = manifest
        logger.info("saved %d chunk(s) to %s", len(payload), target)

    @classmethod
    def load(cls, directory: Path | str) -> FaissVectorStore:
        """Load a previously saved index.

        The index and its metadata are verified to be positionally aligned
        before anything is returned, so a partially written directory is
        rejected instead of silently returning the wrong chunk for a hit.

        Raises:
            VectorStoreError: If the directory is incomplete or inconsistent.
        """
        source = Path(directory)
        index_path = source / INDEX_FILE
        chunks_path = source / CHUNKS_FILE
        if not index_path.is_file() or not chunks_path.is_file():
            raise VectorStoreError(f"no saved index found in {source}")

        try:
            index = faiss.read_index(str(index_path))
        except (RuntimeError, OSError) as exc:
            # FAISS reports an unreadable file as a bare RuntimeError; a corrupt
            # index must surface as VectorStoreError so callers can rebuild it.
            raise VectorStoreError(f"unreadable FAISS index in {source}: {exc}") from exc
        if index.ntotal == 0:
            raise VectorStoreError(f"saved index in {source} is empty")
        try:
            raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorStoreError(f"corrupt chunk metadata in {source}: {exc}") from exc
        if not isinstance(raw_chunks, list) or len(raw_chunks) != index.ntotal:
            raise VectorStoreError(
                f"chunk metadata in {source} does not match the index size ({index.ntotal})"
            )

        store = cls(dimension=index.d, index=index)
        store._chunks = [DocumentChunk.model_validate(item) for item in raw_chunks]
        store._manifest = _read_manifest(
            source, index.ntotal, index.d, tuple(sorted({c.source_file for c in store._chunks}))
        )
        return store


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` (UTF-8, human-readable)."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_manifest(
    directory: Path, chunks: int, dimension: int, source_files: tuple[str, ...]
) -> IndexManifest | None:
    """Load a manifest, returning ``None`` when it is absent or unusable.

    A missing or stale manifest is never fatal: the index itself is still
    verified, and the caller simply decides not to trust the vectors.
    """
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    try:
        manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("ignoring unusable index manifest %s: %s", manifest_path, exc)
        return None
    if manifest.chunks != chunks or manifest.dimension != dimension:
        logger.warning(
            "ignoring index manifest %s: describes %d chunk(s) at dimension %d, "
            "index holds %d at %d",
            manifest_path,
            manifest.chunks,
            manifest.dimension,
            chunks,
            dimension,
        )
        return None
    if manifest.source_files != source_files:
        logger.warning(
            "ignoring index manifest %s: describes source files %s, index holds %s",
            manifest_path,
            list(manifest.source_files),
            list(source_files),
        )
        return None
    return manifest


def _validate_manifest(manifest: IndexManifest, chunks: int, dimension: int) -> None:
    """Refuse to persist a manifest that does not describe this index.

    Raises:
        VectorStoreError: If the manifest disagrees with the stored vectors.
    """
    if manifest.chunks != chunks or manifest.dimension != dimension:
        raise VectorStoreError(
            f"manifest describes {manifest.chunks} chunk(s) at dimension "
            f"{manifest.dimension}, but the index holds {chunks} at {dimension}"
        )
