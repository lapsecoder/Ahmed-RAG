"""Typed exception hierarchy.

Every failure mode that the pipeline can recover from has a dedicated type so
callers can react precisely instead of matching on message strings.
"""

from __future__ import annotations


class AhmedRAGError(Exception):
    """Base class for every error raised by this application."""


class DependencyMissingError(AhmedRAGError):
    """A required third-party runtime dependency is not installed."""


class ConfigurationError(AhmedRAGError):
    """The application configuration is internally inconsistent."""


class ChunkingError(AhmedRAGError):
    """A document could not be split into chunks."""


class EmbeddingError(AhmedRAGError):
    """The embedding model failed to produce usable vectors."""


class DimensionMismatchError(EmbeddingError):
    """A vector was produced with a dimension the index cannot accept."""


class VectorStoreError(AhmedRAGError):
    """The vector store was used in an invalid way."""


class IndexEmptyError(VectorStoreError):
    """A search was attempted against an index that holds no vectors."""


class RetrieverError(AhmedRAGError):
    """Retrieval failed."""


class LLMError(AhmedRAGError):
    """Base class for LLM transport/protocol failures."""


class LLMTimeoutError(LLMError):
    """The LLM did not answer within the configured timeout."""


class LLMConnectionError(LLMError):
    """The LLM endpoint could not be reached."""


class LLMAPIError(LLMError):
    """The LLM endpoint answered with an error or an unusable payload."""


class KnowledgeBaseError(AhmedRAGError):
    """The knowledge base directory is missing or unreadable."""
