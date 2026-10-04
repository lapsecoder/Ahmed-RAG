"""Application settings.

Every knob is environment driven with an ``AHMED_RAG_`` prefix so the service
stays local-first and free: no hosted model, no database, no paid API keys.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.services.knowledge_base import DEFAULT_EXCLUDE_GLOBS

DEFAULT_TOPIC_KEYWORDS: tuple[str, ...] = (
    # Portfolio-domain vocabulary. Deliberately generic: nothing is asserted
    # about Ahmed beyond the conversational surface the assistant must cover.
    "ahmed",
    "portfolio",
    "about you",
    "about ahmed",
    "about the assistant",
    # Identity questions phrased as a request for a name. "What is your name?"
    # carries no other portfolio vocabulary, so without these it was refused as
    # off-topic before retrieval ever ran -- while "who are you" and "what are
    # you" right above it were admitted and then answered.
    "your name",
    "his name",
    "full name",
    "what's your name",
    "whats your name",
    "who are you",
    "what are you",
    "what can you do",
    "your skills",
    "your experience",
    "your background",
    "your projects",
    "your project",
    "your work",
    "your knowledge",
    "your knowledge base",
    "your docs",
    "your documentation",
    "your resume",
    "your cv",
    "your education",
    "your certifications",
    "your achievements",
    "your stack",
    "tech stack",
    "tech used",
    "technologies used",
    "tools used",
    "skills",
    "experience",
    "projects",
    "project",
    "resume",
    "cv",
    "curriculum vitae",
    "education",
    "degree",
    "university",
    "certifications",
    "achievements",
    "contact",
    "email",
    "github",
    "linkedin",
    "website",
    "blog",
    "availability",
    # Interest and hobby wording. interests.md is a real document, but its own
    # vocabulary ("anime", "manga", "gaming", "comic books") is not what a user
    # asks with, and "What does Ahmed do for fun?" reached retrieval as an
    # off-topic message and was never given the chance to be answered from it.
    "interests",
    "interest",
    "hobbies",
    "hobby",
    "pastimes",
    "pastime",
    "for fun",
    "do for fun",
    "enjoy",
    "free time",
    "spare time",
    "help me",
    # "recommend" on its own is far too generic: it routed "Recommend a restaurant
    # near me" in scope. Recommendation vocabulary is kept only where it is
    # anchored to this portfolio's own recommendation work. Portfolio questions
    # such as "Recommend a project of Ahmed to read about" still route in scope
    # via "project"/"ahmed".
    "recommend a movie",
    "recommend movies",
    "movie recommendation",
    "recommendation engine",
    "recommendation system",
    "similar movies",
    "advice",
    "summarize",
    "summary",
    "summarise",
    "difference between",
    # Canonical project and platform names. A bare product name ("What is
    # ResumeForge?") carries no portfolio vocabulary at all, so without these it
    # would be routed off-topic before retrieval ever ran. Deliberately a closed
    # list of real names from the knowledge base -- not a generic widening, so
    # arbitrary off-topic questions stay blocked.
    "resumeforge",
    "resume forge",
    "resume project",
    "resume builder",
    "ats analysis",
    "moviemind",
    "movie mind",
    "movie recommendation project",
    "tmdb",
    "infosys",
    "springboard",
)


class Settings(BaseSettings):
    """Typed configuration for the whole pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="AHMED_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Complex fields are parsed by the validators below, which accept both a
        # comma-separated list and a JSON array. Leaving pydantic-settings' own
        # JSON decoding on would make a plain ``a,b`` in .env a hard error.
        enable_decoding=False,
    )

    app_name: str = "Ahmed-RAG"
    log_level: str = "INFO"

    # Knowledge base / index storage (plain files, no database).
    kb_dir: Path = Field(default=Path("knowledge_base"))
    index_dir: Path = Field(default=Path("storage/index"))
    kb_exclude_globs: tuple[str, ...] = Field(
        default=DEFAULT_EXCLUDE_GLOBS,
        description=(
            "Patterns excluded from indexing, relative to the knowledge base. "
            "A bare name matches any path component; 'dir/**' matches that "
            "directory at any depth. Comma-separated in .env; set to an empty "
            "value to disable."
        ),
    )

    # Embeddings.
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    #: Directory holding the committed ONNX graph and tokenizer. The weights are
    #: the same ``all-MiniLM-L6-v2`` checkpoint PyTorch used, exported to ONNX
    #: so the runtime needs ~60 MB of inference library instead of PyTorch's
    #: multi-gigabyte CUDA stack.
    onnx_model_dir: Path = Field(default=Path("models/all-MiniLM-L6-v2"))

    # Local Ollama.
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5-coder:7b")
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)

    # Chunking.
    chunk_max_chars: int = Field(default=1200, gt=0)
    chunk_overlap_chars: int = Field(default=200, ge=0)

    # Retrieval.
    # 12, not 5: on an 89-chunk corpus the authoritative chunk for a pointed
    # question ("what tech stack", "which certifications") sits at rank 6-12
    # because generic identity chunks dominate MiniLM cosine, and a whole
    # *section* can be absent from the head of the list when its text never
    # repeats the question's noun (the skill-name bullets inside skills.md).
    # 5 made the assistant refuse questions whose answer was in the corpus.
    # The similarity threshold is deliberately untouched -- only recall widened.
    retrieval_top_k: int = Field(default=12, gt=0)
    similarity_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)

    # API.
    max_message_chars: int = Field(default=2000, gt=0)
    topic_keywords: tuple[str, ...] = Field(default=DEFAULT_TOPIC_KEYWORDS)

    @field_validator("kb_dir", "index_dir", "onnx_model_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @field_validator("ollama_base_url", mode="after")
    @classmethod
    def _strip_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("kb_exclude_globs", "topic_keywords", mode="before")
    @classmethod
    def _normalise_keywords(cls, value: object) -> object:
        """Accept a comma-separated list or a JSON array from every source.

        ``pydantic-settings`` would otherwise insist on JSON for tuple fields,
        which makes ``.env`` unreadable for anyone who has never seen a JSON
        array in a shell file.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ()
            if stripped.startswith("["):
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"expected a JSON array or comma-separated list: {exc}"
                    ) from exc
                if not isinstance(decoded, list):
                    raise ValueError(f"expected a JSON array, got {type(decoded).__name__}")
                return tuple(str(item).strip().lower() for item in decoded if str(item).strip())
            return tuple(part.strip().lower() for part in stripped.split(",") if part.strip())
        if isinstance(value, (list, tuple, set)):
            return tuple(str(item).strip().lower() for item in value if str(item).strip())
        return value

    @model_validator(mode="after")
    def _validate_overlap(self) -> Settings:
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError(
                "chunk_overlap_chars must be smaller than chunk_max_chars, "
                f"got {self.chunk_overlap_chars} >= {self.chunk_max_chars}"
            )
        return self

    @property
    def index_file(self) -> Path:
        """Path of the serialised FAISS index."""
        return self.index_dir / "index.faiss"

    @property
    def chunks_file(self) -> Path:
        """Path of the FAISS-position -> DocumentChunk metadata store."""
        return self.index_dir / "chunks.json"

    @property
    def manifest_file(self) -> Path:
        """Path of the index manifest used for staleness detection."""
        return self.index_dir / "manifest.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
