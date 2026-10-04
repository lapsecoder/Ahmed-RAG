"""A scored chunk returned by the retriever."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentChunk


class RetrievalResult(BaseModel):
    """A retrieved chunk together with its dense and lexical scores.

    Because every vector in the index is L2-normalised and the index is
    ``IndexFlatIP``, ``similarity`` is a cosine similarity in ``[-1, 1]``. It is
    reported as 0.0 for a chunk that only the lexical retriever found, because
    no embedding similarity was computed for it.

    ``fused_score`` is the rank-fusion score that actually decided the order.
    It is on an unbounded, non-cosine scale, so callers must not treat it as a
    similarity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: DocumentChunk
    similarity: float = Field(ge=-1.0, le=1.0)
    position: int = Field(ge=0, description="FAISS position the chunk was read from.")
    rank: int = Field(ge=0, description="0-based rank in the filtered result list.")
    lexical_score: float = Field(
        default=0.0, ge=0.0, description="BM25 score; 0.0 when the chunk was not a lexical hit."
    )
    fused_score: float = Field(
        default=0.0, description="Reciprocal-rank-fusion score used for ordering."
    )

    @property
    def source_file(self) -> str:
        """Convenience accessor for the originating file."""
        return self.chunk.source_file
