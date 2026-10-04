"""Domain models shared by the whole pipeline."""

from __future__ import annotations

from app.models.document import DocumentChunk
from app.models.embedding import Embedding
from app.models.enums import ChatOutcome, QueryClassification
from app.models.index import IndexBuildResult
from app.models.retrieval import RetrievalResult

__all__ = [
    "ChatOutcome",
    "DocumentChunk",
    "Embedding",
    "IndexBuildResult",
    "QueryClassification",
    "RetrievalResult",
]
