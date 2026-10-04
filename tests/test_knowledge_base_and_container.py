"""Index construction from Markdown files, persistence and container wiring.

Uses temporary directories with synthetic Markdown, never Ahmed's real
knowledge base, and a deterministic embedder instead of the real model.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
from app.config import Settings
from app.container import build_container
from app.core.exceptions import DimensionMismatchError, VectorStoreError
from app.models.document import DocumentChunk
from app.models.index import IndexBuildResult, IndexManifest
from app.security.injection import InjectionDetector
from app.services.knowledge_base import (
    build_index,
    build_index_from_directory,
    corpus_fingerprint,
    embedding_text,
    list_markdown_files,
    load_chunks,
    save_index,
)
from app.services.vector_store import (
    CHUNKS_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    FaissVectorStore,
)

from .conftest import ScriptedEmbedder, make_chunk

PROFILE = """# Profile

Ahmed builds local-first RAG systems.

## Projects

### Ahmed RAG

A FastAPI service that answers questions with FAISS retrieval and Ollama.
"""

NOTES = """# Notes

Ahmed enjoys long walks and distributed systems reading.
"""


def write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


@pytest.fixture
def knowledge_base(tmp_path: Path) -> Path:
    """A tiny synthetic knowledge base."""
    kb_dir = tmp_path / "kb"
    write(kb_dir / "profile.md", PROFILE)
    write(kb_dir / "notes.md", NOTES)
    write(kb_dir / "ignored.txt", "not markdown")
    return kb_dir


def settings_for(kb_dir: Path, root: Path, **overrides: object) -> Settings:
    """Settings pointed at a synthetic knowledge base and index."""
    options: dict[str, object] = {
        "kb_dir": kb_dir,
        "index_dir": root / "index",
        "chunk_max_chars": 400,
        "chunk_overlap_chars": 60,
    }
    options.update(overrides)
    return Settings(**options)  # type: ignore[arg-type]


def manifest_for(
    *,
    fingerprint: str,
    chunks: int,
    dimension: int = 8,
    embedding_model: str = "mock/minilm",
    source_files: tuple[str, ...] = ("profile.md",),
) -> IndexManifest:
    """A hand-written manifest for tests that need an inconsistent one."""
    return IndexManifest(
        fingerprint=fingerprint,
        chunks=chunks,
        dimension=dimension,
        embedding_model=embedding_model,
        source_files=source_files,
        chunk_max_chars=400,
        chunk_overlap_chars=60,
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_only_markdown_files_are_listed(knowledge_base: Path) -> None:
    names = [path.name for path in list_markdown_files(knowledge_base)]
    assert names == ["notes.md", "profile.md"]


def test_a_missing_knowledge_base_lists_nothing(tmp_path: Path) -> None:
    assert list_markdown_files(tmp_path / "absent") == []


def test_chunks_carry_their_source_file_and_section(knowledge_base: Path) -> None:
    chunks = load_chunks(knowledge_base, max_chars=400, overlap_chars=60)
    assert {chunk.source_file for chunk in chunks} == {"profile.md", "notes.md"}
    assert any(chunk.section == "Profile > Projects > Ahmed RAG" for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def test_building_an_index_from_the_knowledge_base(knowledge_base: Path) -> None:
    embedder = ScriptedEmbedder(dimension=8)
    store = FaissVectorStore(embedder.dimension)
    chunks = load_chunks(knowledge_base, max_chars=400, overlap_chars=60)

    result = build_index(store, embedder, chunks)

    assert isinstance(result, IndexBuildResult)
    assert result.documents == 2
    assert result.chunks == len(chunks)
    assert result.dimension == 8
    assert store.size == len(chunks)
    assert embedder.document_calls == 1
    # Vectors are built from heading_path + body text, never body text alone:
    # a chunk whose heading carries the topical noun ("Contact > Email") must be
    # findable by that noun. See embedding_text().
    assert embedder.documents[0] == [embedding_text(chunk) for chunk in chunks]
    assert embedder.documents[0] != [chunk.text for chunk in chunks]


def test_index_metadata_is_always_document_chunks(knowledge_base: Path) -> None:
    embedder = ScriptedEmbedder(dimension=8)
    store = FaissVectorStore(embedder.dimension)
    build_index_from_directory(knowledge_base, store, embedder, max_chars=400, overlap_chars=60)

    for position in range(store.size):
        chunk = store.chunk_at(position)
        assert isinstance(chunk, DocumentChunk)
        assert chunk.chunk_id.startswith(Path(chunk.source_file).stem)


def test_an_empty_knowledge_base_produces_an_empty_index(tmp_path: Path) -> None:
    embedder = ScriptedEmbedder(dimension=8)
    store = FaissVectorStore(embedder.dimension)
    result = build_index_from_directory(tmp_path / "empty", store, embedder)
    assert result.chunks == 0
    assert result.documents == 0
    assert store.is_empty
    assert embedder.document_calls == 0


def test_a_dimension_mismatch_surfaces_as_an_error() -> None:
    embedder = ScriptedEmbedder(dimension=8)
    store = FaissVectorStore(4)
    with pytest.raises(DimensionMismatchError, match="index was built for dimension 4"):
        build_index(store, embedder, [make_chunk("text")])


def test_built_indexes_persist_and_reload(tmp_path: Path, knowledge_base: Path) -> None:
    embedder = ScriptedEmbedder(dimension=8)
    store = FaissVectorStore(embedder.dimension)
    build_index_from_directory(knowledge_base, store, embedder, max_chars=400, overlap_chars=60)
    store.save(tmp_path / "index")

    reloaded = FaissVectorStore.load(tmp_path / "index")
    assert reloaded.size == store.size
    assert reloaded.chunks == store.chunks


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def build_once(kb_dir: Path, index_dir: Path, **overrides: object) -> tuple[object, IndexManifest]:
    """Build and persist a synthetic index, returning the container and manifest."""
    settings = settings_for(kb_dir, index_dir.parent, **overrides)
    container = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert container.index_manifest is not None
    return container, container.index_manifest


def test_a_built_index_is_written_to_disk_with_a_manifest(
    tmp_path: Path, knowledge_base: Path
) -> None:
    container, manifest = build_once(knowledge_base, tmp_path / "index")
    index_dir = tmp_path / "index"

    assert (index_dir / INDEX_FILE).is_file()
    assert (index_dir / CHUNKS_FILE).is_file()
    assert (index_dir / MANIFEST_FILE).is_file()
    assert not list(index_dir.glob("*.tmp")), "no temporary files may survive a save"

    on_disk = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert on_disk["fingerprint"] == manifest.fingerprint
    assert on_disk["chunks"] == container.store.size
    assert on_disk["dimension"] == 8
    assert on_disk["source_files"] == ["notes.md", "profile.md"]

    reloaded = FaissVectorStore.load(index_dir)
    assert reloaded.manifest == manifest
    assert reloaded.chunks == container.store.chunks


def test_an_empty_index_is_not_persisted(tmp_path: Path) -> None:
    settings = settings_for(tmp_path / "empty-kb", tmp_path)
    container = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert container.store.is_empty
    assert not (settings.index_dir / INDEX_FILE).exists()
    assert container.index_manifest is not None
    assert container.index_manifest.chunks == 0


def test_saving_a_manifest_that_disagrees_with_the_index_is_refused(tmp_path: Path) -> None:
    store = FaissVectorStore(4)
    store.add([make_chunk("text")], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(VectorStoreError, match="manifest describes 7 chunk"):
        store.save(
            tmp_path / "index", manifest=manifest_for(fingerprint="a" * 64, chunks=7, dimension=4)
        )
    assert not (tmp_path / "index" / MANIFEST_FILE).exists()


def test_a_manifest_that_disagrees_with_the_files_is_ignored_on_load(
    tmp_path: Path, knowledge_base: Path
) -> None:
    container, manifest = build_once(knowledge_base, tmp_path / "index")
    index_dir = tmp_path / "index"
    (index_dir / MANIFEST_FILE).write_text(
        manifest.model_copy(update={"chunks": manifest.chunks + 1}).model_dump_json(),
        encoding="utf-8",
    )

    reloaded = FaissVectorStore.load(index_dir)
    assert reloaded.chunks == container.store.chunks
    assert reloaded.manifest is None, "an inconsistent manifest must never grant reuse"


def test_a_corrupt_manifest_is_ignored_on_load(tmp_path: Path, knowledge_base: Path) -> None:
    container, _ = build_once(knowledge_base, tmp_path / "index")
    (tmp_path / "index" / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    reloaded = FaissVectorStore.load(tmp_path / "index")
    assert reloaded.chunks == container.store.chunks
    assert reloaded.manifest is None


def test_metadata_stays_positionally_aligned_after_a_reload(
    tmp_path: Path, knowledge_base: Path
) -> None:
    container, _ = build_once(knowledge_base, tmp_path / "index")
    reloaded = FaissVectorStore.load(tmp_path / "index")

    assert reloaded.size == container.store.size
    for position in range(reloaded.size):
        assert reloaded.chunk_at(position) == container.store.chunk_at(position)

    # Search must return the metadata that belongs to the matching vector.
    last = container.store.size - 1
    last_chunk = container.store.chunk_at(last)
    query = ScriptedEmbedder(dimension=8).embed_query(embedding_text(last_chunk))
    hits = reloaded.search(query, top_k=1)
    assert hits[0].position == last
    assert reloaded.chunk_at(hits[0].position) == container.store.chunk_at(last)


def test_a_failed_save_leaves_the_previous_index_intact(
    tmp_path: Path, knowledge_base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container, _ = build_once(knowledge_base, tmp_path / "index")
    index_dir = tmp_path / "index"
    original = FaissVectorStore.load(index_dir)

    def explode(path: Path, payload: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("app.services.vector_store._write_json", explode)
    with pytest.raises(OSError, match="disk full"):
        save_index(
            FaissVectorStore.load(index_dir),
            index_dir,
            kb_dir=knowledge_base,
            embedder=ScriptedEmbedder(dimension=8),
            max_chars=400,
            overlap_chars=60,
        )

    survivors = FaissVectorStore.load(index_dir)
    assert survivors.chunks == original.chunks == container.store.chunks
    assert not list(index_dir.glob("*.tmp")), "a failed save must not leave temporary files"


def test_a_failed_save_never_partially_replaces_the_live_files(
    tmp_path: Path, knowledge_base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every write must land in a temporary file before any file is replaced."""
    build_once(knowledge_base, tmp_path / "index")
    index_dir = tmp_path / "index"
    before = {
        name: (index_dir / name).read_bytes() for name in (INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE)
    }

    events: list[str] = []
    monkeypatch.setattr(
        "app.services.vector_store.faiss.write_index", lambda *_: events.append("write-index")
    )

    def record_json(path: Path, payload: object) -> None:
        events.append(f"write-{path.name}")

    monkeypatch.setattr("app.services.vector_store._write_json", record_json)
    monkeypatch.setattr(
        "os.replace", lambda src, dst: events.append(f"replace-{Path(str(dst)).name}")
    )

    manifest = FaissVectorStore.load(index_dir).manifest
    assert manifest is not None
    FaissVectorStore.load(index_dir).save(index_dir, manifest=manifest)

    assert events == [
        "write-index",
        "write-chunks.json.tmp",
        "write-manifest.json.tmp",
        "replace-index.faiss",
        "replace-chunks.json",
        "replace-manifest.json",
    ], "nothing may be replaced until every temporary file has been written"

    for name in before:
        assert (index_dir / name).is_file()


def test_the_manifest_fingerprint_is_the_corpus_fingerprint(
    tmp_path: Path, knowledge_base: Path
) -> None:
    _, manifest = build_once(knowledge_base, tmp_path / "index")
    assert manifest.fingerprint == corpus_fingerprint(knowledge_base)
    assert len(manifest.fingerprint) == 64


# --------------------------------------------------------------------------- #
# Reuse and staleness
# --------------------------------------------------------------------------- #


def test_an_unchanged_knowledge_base_reuses_the_saved_index(
    tmp_path: Path, knowledge_base: Path
) -> None:
    first, _ = build_once(knowledge_base, tmp_path / "index")
    assert first.index_reused is False

    settings = settings_for(knowledge_base, tmp_path)
    second = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert second.index_reused is True
    assert second.embedder.document_calls == 0, "an unchanged corpus must not be re-embedded"
    assert second.store.size == first.store.size
    assert second.store.chunks == first.store.chunks
    assert second.index_manifest == first.index_manifest


def test_editing_a_knowledge_file_rebuilds_the_index(tmp_path: Path, knowledge_base: Path) -> None:
    first, _ = build_once(knowledge_base, tmp_path / "index")
    write(knowledge_base / "notes.md", NOTES.replace("long walks", "long rides"))

    settings = settings_for(knowledge_base, tmp_path)
    second = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert second.index_reused is False
    assert second.embedder.document_calls == 1
    assert second.store.size == first.store.size
    assert any("long rides" in chunk.text for chunk in second.store.chunks)
    assert second.index_manifest is not None
    assert second.index_manifest.fingerprint != first.index_manifest.fingerprint


def test_adding_or_removing_a_file_rebuilds_the_index(tmp_path: Path, knowledge_base: Path) -> None:
    first, _ = build_once(knowledge_base, tmp_path / "index")
    write(knowledge_base / "contact.md", "# Contact\n\nReachable by email.\n")

    settings = settings_for(knowledge_base, tmp_path)
    second = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert second.index_reused is False
    assert second.knowledge_base_documents == 3
    assert second.index_manifest.fingerprint != first.index_manifest.fingerprint  # type: ignore[union-attr]

    (knowledge_base / "contact.md").unlink()
    settings = settings_for(knowledge_base, tmp_path)
    third = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert third.index_manifest == first.index_manifest


def test_a_touched_but_unchanged_file_still_reuses_the_index(
    tmp_path: Path, knowledge_base: Path
) -> None:
    build_once(knowledge_base, tmp_path / "index")
    target = knowledge_base / "notes.md"
    stat = target.stat()
    os.utime(target, (stat.st_atime + 5000, stat.st_mtime + 5000))

    settings = settings_for(knowledge_base, tmp_path)
    reused = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert reused.index_reused is True
    assert reused.embedder.document_calls == 0


def test_changing_the_chunk_settings_rebuilds_the_index(
    tmp_path: Path, knowledge_base: Path
) -> None:
    build_once(knowledge_base, tmp_path / "index", chunk_max_chars=400)
    settings = settings_for(knowledge_base, tmp_path, chunk_max_chars=500)
    rebuilt = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert rebuilt.index_reused is False
    assert rebuilt.embedder.document_calls == 1


def test_changing_the_embedding_model_rebuilds_the_index(
    tmp_path: Path, knowledge_base: Path
) -> None:
    """Vectors from another model are meaningless in this index."""
    build_once(knowledge_base, tmp_path / "index")
    settings = settings_for(knowledge_base, tmp_path, embedding_model="other-model")
    rebuilt = build_container(
        settings,
        embedder=ScriptedEmbedder(dimension=8, model_name="other-model"),
    )
    assert rebuilt.index_reused is False
    assert rebuilt.embedder.document_calls == 1


def test_changing_the_exclude_globs_rebuilds_the_index(
    tmp_path: Path, knowledge_base: Path
) -> None:
    build_once(knowledge_base, tmp_path / "index")
    settings = settings_for(knowledge_base, tmp_path, kb_exclude_globs="profile.md")
    rebuilt = build_container(settings, embedder=ScriptedEmbedder(dimension=8))

    assert rebuilt.index_reused is False
    assert rebuilt.embedder.document_calls == 1
    assert {chunk.source_file for chunk in rebuilt.store.chunks} == {"notes.md"}


def test_a_file_inside_an_excluded_directory_does_not_invalidate_the_index(
    tmp_path: Path, knowledge_base: Path
) -> None:
    """Excluded files are invisible to the fingerprint, not just to the index."""
    first, _ = build_once(knowledge_base, tmp_path / "index")
    write(knowledge_base / "drafts" / "todo.md", "# Draft\n\nNot finished.\n")

    settings = settings_for(knowledge_base, tmp_path)
    second = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert second.index_reused is True
    assert second.embedder.document_calls == 0
    assert second.index_manifest == first.index_manifest


def test_a_saved_index_without_a_manifest_is_rebuilt(tmp_path: Path, knowledge_base: Path) -> None:
    """An index nobody can vouch for must never be trusted."""
    store = FaissVectorStore(8)
    build_index_from_directory(
        knowledge_base, store, ScriptedEmbedder(dimension=8), max_chars=400, overlap_chars=60
    )
    store.save(tmp_path / "index")
    assert not (tmp_path / "index" / MANIFEST_FILE).exists()

    settings = settings_for(knowledge_base, tmp_path)
    container = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert container.index_reused is False
    assert container.embedder.document_calls == 1
    assert (tmp_path / "index" / MANIFEST_FILE).is_file()


def test_a_corrupt_index_file_is_rebuilt(tmp_path: Path, knowledge_base: Path) -> None:
    build_once(knowledge_base, tmp_path / "index")
    (tmp_path / "index" / INDEX_FILE).write_bytes(b"not a faiss index")

    settings = settings_for(knowledge_base, tmp_path)
    container = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert container.index_reused is False
    assert container.embedder.document_calls == 1
    assert container.store.size > 0


def test_a_manifest_describing_other_files_is_rejected(
    tmp_path: Path, knowledge_base: Path
) -> None:
    """A manifest that disagrees with the metadata it sits next to is ignored."""
    settings = settings_for(knowledge_base, tmp_path)
    legacy = FaissVectorStore(8)
    build_index_from_directory(
        knowledge_base, legacy, ScriptedEmbedder(dimension=8), max_chars=400, overlap_chars=60
    )
    legacy.save(
        tmp_path / "index",
        manifest=manifest_for(fingerprint="b" * 64, chunks=legacy.size),
    )
    assert FaissVectorStore.load(tmp_path / "index").manifest is None

    rebuilt = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert rebuilt.index_reused is False
    assert rebuilt.embedder.document_calls == 1


def test_indexing_is_skipped_when_a_container_is_built_without_directory_building(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path / "kb", tmp_path)
    container = build_container(
        settings,
        embedder=ScriptedEmbedder(dimension=8),
        build_from_directory=False,
    )
    assert container.store.is_empty
    assert container.index_result.chunks == 0


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #


def test_the_container_reuses_one_embedder_and_index(tmp_path: Path, knowledge_base: Path) -> None:
    settings = settings_for(knowledge_base, tmp_path, similarity_threshold=0.2)
    embedder = ScriptedEmbedder(dimension=8)
    container = build_container(settings, embedder=embedder)

    assert container.store.size > 0
    assert container.retriever.similarity_threshold == 0.2
    assert container.index_result.chunks == container.store.size
    assert container.knowledge_base_documents == 2
    assert container.index_result.source_files == ("notes.md", "profile.md")
    # One embedder instance, loaded once.
    assert container.embedder is embedder
    assert embedder.document_calls == 1
    assert embedder.query_calls == 0


def test_the_container_reports_the_manifest_it_loaded_or_built(
    tmp_path: Path, knowledge_base: Path
) -> None:
    _, manifest = build_once(knowledge_base, tmp_path / "index")
    settings = settings_for(knowledge_base, tmp_path)
    container = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert container.index_manifest == manifest
    assert container.index_manifest is not None
    assert container.index_manifest.embedding_model == ScriptedEmbedder(dimension=8).model_name
    assert container.index_manifest.chunks == container.store.size


def test_the_container_exposes_every_collaborator(tmp_path: Path) -> None:
    from app.security.classification import QueryClassifier
    from app.security.output_validator import OutputValidator
    from app.services.answering.composer import AnswerComposer
    from app.services.hybrid_retriever import HybridRetriever

    settings = settings_for(tmp_path / "kb", tmp_path)
    container = build_container(settings, embedder=ScriptedEmbedder(dimension=8))
    assert isinstance(container.detector, InjectionDetector)
    assert isinstance(container.classifier, QueryClassifier)
    assert isinstance(container.output_validator, OutputValidator)
    assert isinstance(container.composer, AnswerComposer)
    assert isinstance(container.retriever, HybridRetriever)
    assert container.index_dir == settings.index_dir
    assert container.knowledge_base_documents == 0


def test_the_container_builds_no_language_model(tmp_path: Path) -> None:
    """Nothing in the container can reach a network model.

    Asserting absence rather than a mock's call count: with no provider
    constructed there is nothing that *could* generate, which is a stronger
    statement than "the provider was not called".
    """
    settings = settings_for(tmp_path / "kb", tmp_path)
    container = build_container(settings, embedder=ScriptedEmbedder(dimension=8))

    field_names = {field.name for field in dataclasses.fields(container)}
    assert not {"llm", "prompt_builder"} & field_names

    collaborator_names = {
        type(getattr(container, field.name)).__name__ for field in dataclasses.fields(container)
    }
    assert not any(name.endswith("LLM") or "Ollama" in name for name in collaborator_names), (
        collaborator_names
    )
    assert not hasattr(container.chat_service, "llm")


def test_the_index_dimension_follows_the_embedder(tmp_path: Path) -> None:
    for dimension in (4, 8, 16):
        settings = settings_for(tmp_path / f"kb-{dimension}", tmp_path / f"idx-{dimension}")
        container = build_container(settings, embedder=ScriptedEmbedder(dimension=dimension))
        assert container.store.dimension == dimension
        assert np.issubdtype(np.float32, np.floating)
