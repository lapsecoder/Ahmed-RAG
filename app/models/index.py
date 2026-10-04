"""Index build results and the persisted index manifest."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Bumped whenever the meaning of a persisted manifest changes, so an index
#: written by an incompatible build is never reused.
#:
#: 2 - vectors are built from ``heading_path`` + body text (see
#:     :func:`app.services.knowledge_base.embedding_text`). Version 1 indexes
#: embedded body text alone, so its vectors are not comparable and must be
#: rebuilt even though the corpus itself is unchanged.
MANIFEST_SCHEMA_VERSION: int = 2


class IndexBuildResult(BaseModel):
    """Summary of what an index build produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: int = Field(ge=0, description="Markdown files read.")
    chunks: int = Field(ge=0, description="Chunks created and indexed.")
    dimension: int = Field(ge=1, description="Embedding dimension the index was sized for.")
    source_files: tuple[str, ...] = Field(
        default=(), description="Knowledge-base-relative paths that were indexed."
    )
    fingerprint: str = Field(default="", description="Content fingerprint of the indexed corpus.")


class IndexManifest(BaseModel):
    """Provenance record written next to a persisted index.

    The manifest is what makes a saved index *reusable*: the corpus content
    fingerprint, the embedding model and the dimension must all still agree
    before a persisted index is loaded instead of rebuilt. A manifest with
    ``chunks == 0`` describes an empty index, which is never persisted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(
        default=MANIFEST_SCHEMA_VERSION, ge=1, description="Manifest layout version."
    )
    fingerprint: str = Field(min_length=1, description="SHA-256 over the indexed file contents.")
    chunks: int = Field(ge=0, description="Vectors stored in the index.")
    dimension: int = Field(ge=1, description="Embedding dimension of the stored vectors.")
    embedding_model: str = Field(min_length=1, description="Model that produced the vectors.")
    source_files: tuple[str, ...] = Field(
        default=(), description="Knowledge-base-relative paths that were indexed."
    )
    chunk_max_chars: int = Field(ge=1, description="Character budget used when chunking.")
    chunk_overlap_chars: int = Field(ge=0, description="Overlap used when chunking.")

    def matches(
        self,
        *,
        fingerprint: str,
        dimension: int,
        embedding_model: str,
        chunk_max_chars: int,
        chunk_overlap_chars: int,
    ) -> bool:
        """Whether this manifest still describes the current corpus and settings.

        Everything that can change the *meaning* of the stored vectors is
        compared: corpus content, vector dimension, the embedding model, and the
        chunking parameters. Modification times are deliberately not used -- they
        say nothing about content and are unreliable across copies.
        """
        return (
            self.schema_version == MANIFEST_SCHEMA_VERSION
            and self.fingerprint == fingerprint
            and self.dimension == dimension
            and self.embedding_model == embedding_model
            and self.chunk_max_chars == chunk_max_chars
            and self.chunk_overlap_chars == chunk_overlap_chars
        )
