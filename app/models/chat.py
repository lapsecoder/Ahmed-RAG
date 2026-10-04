"""Chat-level result models shared by the service layer and the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ChatOutcome, QueryClassification


class SourceRef(BaseModel):
    """A retrieved chunk, as reported back to the caller."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    source_file: str
    section: str
    similarity: float = Field(ge=-1.0, le=1.0)


class ChatResult(BaseModel):
    """The complete outcome of one chat turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: str = Field(min_length=1)
    classification: QueryClassification
    outcome: ChatOutcome
    sources: tuple[SourceRef, ...] = ()
    retrieval_scores: tuple[float, ...] = ()
    llm_used: bool = Field(
        default=False,
        description="False whenever a deterministic path answered instead of the LLM.",
    )
    injection_rule_ids: tuple[str, ...] = Field(
        default=(),
        description="Security rules that fired, empty unless an attempt was blocked.",
    )
