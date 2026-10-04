"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.chat import SourceRef
from app.models.enums import ChatOutcome, QueryClassification

#: Absolute ceiling enforced by the schema. The (usually smaller) per-deployment
#: limit from :class:`app.config.Settings` is checked by the route.
HARD_MESSAGE_LIMIT: int = 8000


class ChatRequest(BaseModel):
    """Inbound chat payload."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: Annotated[
        str,
        Field(min_length=1, max_length=HARD_MESSAGE_LIMIT, description="The user's question."),
    ]

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty or whitespace only")
        return value


class ChatResponse(BaseModel):
    """Outbound chat payload."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1)
    classification: QueryClassification
    outcome: ChatOutcome
    sources: tuple[SourceRef, ...] = ()
    retrieval_scores: tuple[float, ...] = ()
    llm_used: bool = Field(
        default=False,
        description="Always false: answers are extracted from the knowledge base, not generated.",
    )
    injection_rule_ids: tuple[str, ...] = ()


class HealthResponse(BaseModel):
    """Service health."""

    model_config = ConfigDict(extra="forbid")

    status: str
    app: str
    answerer: str = Field(
        description="What produces the answer text, e.g. 'deterministic-extractive'.",
    )
    llm_model: str | None = Field(
        default=None,
        description=(
            "Always null. No language model participates in answering; retained so "
            "existing clients reading this key keep parsing the response."
        ),
    )
    embedding_model: str
    index_size: int
    index_dimension: int
    similarity_threshold: float
    knowledge_base_documents: int


class ErrorResponse(BaseModel):
    """Uniform error body."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    code: str
