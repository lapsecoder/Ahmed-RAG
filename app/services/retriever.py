"""Retrieval: embed the query, search FAISS, apply the similarity threshold."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import RetrieverError
from app.core.logging import get_logger
from app.models.retrieval import RetrievalResult
from app.services.embedder import Embedder
from app.services.vector_store import FaissVectorStore

logger = get_logger("services.retriever")


class Retriever:
    """Thresholded similarity search over a :class:`FaissVectorStore`."""

    def __init__(
        self,
        store: FaissVectorStore,
        embedder: Embedder,
        *,
        similarity_threshold: float = 0.35,
        top_k: int = 5,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise RetrieverError(
                f"similarity_threshold must be in [-1, 1], got {similarity_threshold}"
            )
        if top_k <= 0:
            raise RetrieverError(f"top_k must be positive, got {top_k}")
        self._store = store
        self._embedder = embedder
        self._threshold = float(similarity_threshold)
        self._top_k = int(top_k)

    @property
    def store(self) -> FaissVectorStore:
        """The backing vector store."""
        return self._store

    @property
    def similarity_threshold(self) -> float:
        """Minimum cosine similarity a chunk needs to be returned."""
        return self._threshold

    @property
    def top_k(self) -> int:
        """Maximum number of chunks considered per query."""
        return self._top_k

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Embed a raw query string. No placeholder document is ever created."""
        if not isinstance(query, str) or not query.strip():
            raise RetrieverError("query must be a non-empty string")
        return self._embedder.embed_query(query)

    def retrieve(
        self,
        query: str,
        *,
        restrict_to: Collection[str] | None = None,
        relax_threshold: bool = False,
    ) -> list[RetrievalResult]:
        """Return the chunks that clear the similarity threshold.

        Args:
            query: The user's question.
            restrict_to: When given, only chunks from these ``source_file`` values
                are eligible. A restricted search searches the whole store but
                keeps only in-scope documents, so a hit it returns is never
                padding from an unrelated part of the corpus.
            relax_threshold: Admit the best chunk from the permitted documents
                even when it is below ``similarity_threshold``. Sound only
                alongside ``restrict_to``: the threshold exists to keep unrelated
                chunks out of a global search, which is moot once the search has
                been narrowed to the documents that own the question. The caller
                is then responsible for the evidence still being on topic.

        Returns:
            Results ordered by descending similarity, each pairing the original
            :class:`DocumentChunk` with its cosine similarity. An empty list is
            returned when nothing clears the threshold -- callers must treat that
            as "no information" and must not fall back to guessing.
        """
        if not self._store.is_empty:
            vector = self.embed_query(query)
            # Search the whole store so ranking is unaffected by the filter, then
            # take enough hits to be able to satisfy the filter from the top.
            depth = self._top_k if restrict_to is None else self._store.size
            hits = self._store.search(vector, depth)
        else:
            logger.info("index is empty; skipping query embedding")
            hits = []

        results: list[RetrievalResult] = []
        permitted = frozenset(restrict_to) if restrict_to is not None else None
        for hit in hits:
            chunk = self._store.chunk_at(hit.position)
            if permitted is not None and chunk.source_file not in permitted:
                continue
            if hit.similarity < self._threshold and not (relax_threshold and permitted):
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    similarity=hit.similarity,
                    position=hit.position,
                    rank=len(results),
                )
            )
        if not results:
            logger.debug("no chunk cleared the similarity threshold %.3f", self._threshold)
        return results

    def retrieve_many(self, queries: Sequence[str]) -> list[list[RetrievalResult]]:
        """Convenience wrapper for batch evaluation."""
        return [self.retrieve(query) for query in queries]
