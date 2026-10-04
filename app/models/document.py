"""The document unit that the vector store indexes and the LLM receives."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """A coherent, retrievable slice of a Markdown knowledge document.

    Instances of this class -- and never raw embeddings -- are what a FAISS
    position maps to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1, description="Stable, content-derived identifier.")
    source_file: str = Field(min_length=1, description="Knowledge-base file this came from.")
    section: str = Field(min_length=1, description="Heading path the chunk belongs to.")
    text: str = Field(min_length=1, description="Chunk body, headings included.")

    heading_path: tuple[str, ...] = Field(default=(), description="Ordered heading hierarchy.")
    heading_level: int = Field(default=0, ge=0, description="Depth of the deepest heading (0-3).")
    start_line: int = Field(default=0, ge=0, description="1-based first source line.")
    end_line: int = Field(default=0, ge=0, description="1-based last source line.")
    char_start: int = Field(default=0, ge=0, description="0-based offset in the source text.")
    char_end: int = Field(default=0, ge=0, description="Exclusive 0-based source offset.")
    ordinal: int = Field(
        default=0, ge=0, description="Position of the chunk inside its source file."
    )

    @property
    def citation(self) -> str:
        """Human-readable citation used in logs and API responses."""
        return f"{self.source_file} :: {self.section}"
