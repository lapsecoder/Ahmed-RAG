"""Configuration surface: defaults, environment parsing and derived paths.

Environment variables are set explicitly on every test so the result never
depends on a developer's own ``.env`` file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.config import DEFAULT_TOPIC_KEYWORDS, Settings, get_settings
from app.services.knowledge_base import DEFAULT_EXCLUDE_GLOBS
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any AHMED_RAG_* variable and ignore a developer's own .env."""
    for variable in list(os.environ):
        if variable.startswith("AHMED_RAG_"):
            monkeypatch.delenv(variable)
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    get_settings.cache_clear()


def test_the_defaults_are_local_first() -> None:
    settings = Settings()
    assert settings.kb_dir == Path("knowledge_base")
    assert settings.index_dir == Path("storage/index")
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.topic_keywords == DEFAULT_TOPIC_KEYWORDS


def test_exclude_globs_default_to_the_documented_patterns() -> None:
    assert Settings().kb_exclude_globs == DEFAULT_EXCLUDE_GLOBS
    assert ".*" in DEFAULT_EXCLUDE_GLOBS, "dot files are never indexed by default"


@pytest.mark.parametrize(
    "raw",
    [".*,drafts/**", ".*, drafts/** ,  exports/**", ".*,,drafts/**"],
)
def test_exclude_globs_accept_a_comma_separated_env_value(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("AHMED_RAG_KB_EXCLUDE_GLOBS", raw)
    assert Settings().kb_exclude_globs == tuple(
        part.strip() for part in raw.split(",") if part.strip()
    )


def test_exclude_globs_still_accept_a_json_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AHMED_RAG_KB_EXCLUDE_GLOBS", '[".*", "drafts/**"]')
    assert Settings().kb_exclude_globs == (".*", "drafts/**")


def test_an_empty_exclude_globs_value_disables_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AHMED_RAG_KB_EXCLUDE_GLOBS", "")
    assert Settings().kb_exclude_globs == ()


def test_a_malformed_exclude_globs_value_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AHMED_RAG_KB_EXCLUDE_GLOBS", "[not json")
    with pytest.raises(ValueError, match="comma-separated list"):
        Settings()


def test_topic_keywords_accept_both_env_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AHMED_RAG_TOPIC_KEYWORDS", "RAG, Python")
    assert Settings().topic_keywords == ("rag", "python")
    monkeypatch.setenv("AHMED_RAG_TOPIC_KEYWORDS", '["rag", "python"]')
    assert Settings().topic_keywords == ("rag", "python")


def test_exclude_globs_are_case_insensitive() -> None:
    assert Settings(kb_exclude_globs="DRAFTS/**").kb_exclude_globs == ("drafts/**",)


def test_paths_are_expanded() -> None:
    assert Settings(kb_dir="~/kb").kb_dir == Path("~/kb").expanduser()


def test_the_ollama_base_url_loses_its_trailing_slash() -> None:
    assert Settings(ollama_base_url="http://localhost:11434/").ollama_base_url == (
        "http://localhost:11434"
    )


def test_overlap_must_be_smaller_than_the_chunk_budget() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_chars must be smaller"):
        Settings(chunk_max_chars=400, chunk_overlap_chars=400)


def test_index_paths_are_derived_from_the_index_directory() -> None:
    settings = Settings(index_dir=Path("storage/index"))
    assert settings.index_file == Path("storage/index/index.faiss")
    assert settings.chunks_file == Path("storage/index/chunks.json")
    assert settings.manifest_file == Path("storage/index/manifest.json")


def test_the_dead_fallback_dimension_setting_is_gone() -> None:
    """It was never read; the index dimension comes from the embedder."""
    assert "fallback_embedding_dimension" not in Settings.model_fields
    assert not hasattr(Settings(), "fallback_embedding_dimension")
