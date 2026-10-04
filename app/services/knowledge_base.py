"""Knowledge-base discovery, fingerprinting and index construction.

Discovery is content-driven and deterministic:

* Markdown files are found recursively and their identity is their path
  **relative to the knowledge base**, so two files that share a name in
  different folders stay distinguishable.
* Exclude globs remove drafts, exports and dot-prefixed paths. The defaults are
  conservative and are overridable through settings.
* The corpus fingerprint is a SHA-256 over the relative path and the *content*
  of every file that would be indexed. Modification times are never consulted:
  they say nothing about content and are unreliable across copies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Final

from app.core.exceptions import KnowledgeBaseError
from app.core.logging import get_logger
from app.models.document import DocumentChunk
from app.models.index import IndexBuildResult, IndexManifest
from app.services.chunker import chunk_file
from app.services.embedder import Embedder
from app.services.vector_store import FaissVectorStore

logger = get_logger("services.knowledge_base")

MARKDOWN_SUFFIXES: Final[tuple[str, ...]] = (".md", ".markdown")

#: Conservative defaults. A slash-less pattern matches any single path
#: component; a ``dir/**`` pattern matches that directory at any depth.
DEFAULT_EXCLUDE_GLOBS: Final[tuple[str, ...]] = (
    ".*",
    "drafts/**",
    "_drafts/**",
    "exports/**",
    "*.tmp.md",
    "*~",
)


def relative_source_path(path: Path, kb_dir: Path) -> str:
    """Return ``path`` relative to ``kb_dir`` using ``/`` separators.

    Falls back to the bare file name when the path lies outside ``kb_dir``,
    which keeps chunking usable for ad-hoc callers.

    Raises:
        KnowledgeBaseError: If ``kb_dir`` is not a directory.
    """
    if not kb_dir.is_dir():
        raise KnowledgeBaseError(f"knowledge base directory does not exist: {kb_dir}")
    try:
        return path.relative_to(kb_dir).as_posix()
    except ValueError:  # pragma: no cover - only for out-of-tree callers
        return path.name


def is_excluded(relative_path: str, globs: Iterable[str]) -> bool:
    """Whether a knowledge-base-relative path matches any exclude glob.

    Args:
        relative_path: Posix-style path relative to the knowledge base.
        globs: Patterns such as ``.*``, ``drafts/**`` or ``*.tmp.md``.

    Returns:
        ``True`` when the path must not be indexed.
    """
    path = PurePosixPath(relative_path)
    parts = path.parts
    for raw_pattern in globs:
        pattern = raw_pattern.strip().replace("\\", "/")
        if not pattern:
            continue
        if fnmatch(relative_path, pattern):
            return True
        if "/" not in pattern:
            # A bare name matches any component: a file or a directory.
            if any(fnmatch(part, pattern) for part in parts):
                return True
        elif pattern.endswith("/**"):
            prefix = pattern[:-3]
            directories = relative_path.split("/")[:-1]
            for index in range(len(directories)):
                if fnmatch(directories[index], prefix) or fnmatch(
                    "/".join(directories[: index + 1]), prefix
                ):
                    return True
    return False


def list_markdown_files(
    kb_dir: Path | str,
    exclude_globs: Sequence[str] | None = None,
) -> list[Path]:
    """Return the Markdown files of a knowledge base, sorted by relative path.

    Args:
        kb_dir: Knowledge-base root. A missing directory yields no files.
        exclude_globs: Patterns to skip. Defaults to
            :data:`DEFAULT_EXCLUDE_GLOBS`.

    Returns:
        Absolute paths, sorted by their knowledge-base-relative form so the
        result -- and therefore the index -- is identical on every platform.

    Raises:
        KnowledgeBaseError: If the directory exists but cannot be listed.
    """
    directory = Path(kb_dir)
    if not directory.is_dir():
        return []
    globs = tuple(DEFAULT_EXCLUDE_GLOBS if exclude_globs is None else exclude_globs)
    try:
        candidates = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in MARKDOWN_SUFFIXES
        ]
    except OSError as exc:
        raise KnowledgeBaseError(f"cannot list knowledge base {directory}: {exc}") from exc

    selected: list[tuple[str, Path]] = []
    for path in candidates:
        try:
            relative = path.relative_to(directory).as_posix()
        except ValueError:  # pragma: no cover - defensive
            relative = path.name
        if is_excluded(relative, globs):
            logger.debug("skipping excluded knowledge file %s", relative)
            continue
        selected.append((relative, path))

    selected.sort(key=lambda item: item[0])
    return [path for _, path in selected]


def source_paths(kb_dir: Path | str, exclude_globs: Sequence[str] | None = None) -> list[str]:
    """Return the relative paths of the files that would be indexed."""
    directory = Path(kb_dir)
    return [
        relative_source_path(path, directory)
        for path in list_markdown_files(directory, exclude_globs)
    ]


def corpus_fingerprint(kb_dir: Path | str, exclude_globs: Sequence[str] | None = None) -> str:
    """Hash the exact corpus that would be indexed.

    The digest covers each file's relative path *and* its bytes, so it changes
    when a file is added, removed, renamed or edited, and is stable when only
    the modification time or the filesystem layout changes.

    Returns:
        A hex SHA-256 digest. An empty corpus hashes the empty string, which is
        still a valid, comparable fingerprint.
    """
    digest = hashlib.sha256()
    for relative, path in _indexable_files(kb_dir, exclude_globs):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            raise KnowledgeBaseError(f"cannot read knowledge file {path}: {exc}") from exc
        digest.update(b"\n")
    return digest.hexdigest()


def _indexable_files(
    kb_dir: Path | str, exclude_globs: Sequence[str] | None
) -> list[tuple[str, Path]]:
    directory = Path(kb_dir)
    return [
        (relative_source_path(path, directory), path)
        for path in list_markdown_files(directory, exclude_globs)
    ]


def load_chunks(
    kb_dir: Path | str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    exclude_globs: Sequence[str] | None = None,
) -> list[DocumentChunk]:
    """Chunk every Markdown document in the knowledge base.

    Each chunk records the file's knowledge-base-relative path, so
    ``projects/ahmed-rag.md`` and ``archive/ahmed-rag.md`` remain distinct in
    citations and in API responses.
    """
    directory = Path(kb_dir)
    chunks: list[DocumentChunk] = []
    for path in list_markdown_files(directory, exclude_globs):
        relative = relative_source_path(path, directory)
        file_chunks = chunk_file(
            path, source_name=relative, max_chars=max_chars, overlap_chars=overlap_chars
        )
        logger.info("chunked %s into %d chunk(s)", relative, len(file_chunks))
        chunks.extend(file_chunks)
    return chunks


def build_manifest(
    *,
    fingerprint: str,
    store: FaissVectorStore,
    embedder: Embedder,
    max_chars: int,
    overlap_chars: int,
) -> IndexManifest:
    """Describe the index that was just built."""
    return IndexManifest(
        fingerprint=fingerprint,
        chunks=store.size,
        dimension=store.dimension,
        embedding_model=embedder.model_name,
        source_files=tuple(sorted({chunk.source_file for chunk in store.chunks})),
        chunk_max_chars=max_chars,
        chunk_overlap_chars=overlap_chars,
    )


def embedding_text(chunk: DocumentChunk) -> str:
    """Return the text actually embedded for ``chunk``.

    Retrieval failed for chunks whose *heading* carried the topical noun while
    the body did not. "How can I contact Ahmed?" never reached ``Contact >
    Email`` (rank 10) and "What certifications does Ahmed have?" never reached
    ``Certifications > Recorded certifications`` (outside the top 16), because
    the vectors were built from body text alone. Prefixing the heading path
    puts the document's own vocabulary into the vector.

    Only the embedded string changes. ``DocumentChunk.text`` -- and therefore
    chunk ids, offsets, the rendered prompt and every stored field -- is left
    untouched, so this adds no new surface for injection to hide in.
    """
    heading = " > ".join(chunk.heading_path).strip()
    if not heading:
        return chunk.text
    return f"{heading}\n{chunk.text}"


def build_index(
    store: FaissVectorStore,
    embedder: Embedder,
    chunks: list[DocumentChunk],
) -> IndexBuildResult:
    """Embed ``chunks`` and add them to ``store``.

    The embedder output is paired with the chunks here; the store itself only
    ever receives ``DocumentChunk`` metadata.
    """
    if not chunks:
        return IndexBuildResult(documents=0, chunks=0, dimension=store.dimension, source_files=())

    texts = [embedding_text(chunk) for chunk in chunks]
    vectors = embedder.embed_documents(texts)
    if vectors.shape[0] != len(chunks):
        raise KnowledgeBaseError(
            f"embedder returned {vectors.shape[0]} vector(s) for {len(chunks)} chunk(s)"
        )
    store.add(list(chunks), vectors)
    logger.info("built index with %d chunk(s) at dimension %d", len(chunks), store.dimension)
    return IndexBuildResult(
        documents=len({chunk.source_file for chunk in chunks}),
        chunks=len(chunks),
        dimension=store.dimension,
        source_files=tuple(sorted({chunk.source_file for chunk in chunks})),
    )


def build_index_from_directory(
    kb_dir: Path | str,
    store: FaissVectorStore,
    embedder: Embedder,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    exclude_globs: Sequence[str] | None = None,
) -> IndexBuildResult:
    """Load the knowledge base from disk and index it.

    The returned result carries the corpus ``fingerprint`` so the caller can
    persist a manifest that proves what the index was built from.
    """
    chunks = load_chunks(
        kb_dir,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        exclude_globs=exclude_globs,
    )
    if not chunks:
        logger.info("knowledge base %s contains no Markdown documents yet", kb_dir)
    result = build_index(store, embedder, chunks)
    return result.model_copy(update={"fingerprint": corpus_fingerprint(kb_dir, exclude_globs)})


def save_index(
    store: FaissVectorStore,
    index_dir: Path | str,
    *,
    kb_dir: Path | str,
    embedder: Embedder,
    max_chars: int,
    overlap_chars: int,
    exclude_globs: Sequence[str] | None = None,
    fingerprint: str | None = None,
) -> IndexManifest:
    """Persist ``store`` together with a manifest describing its provenance.

    An empty index is not written: ``load`` rejects an empty index, so writing
    one would only produce a file that is rebuilt again on the next start.

    Args:
        store: The index to persist.
        index_dir: Destination directory for the three index files.
        kb_dir: Knowledge base the index was built from.
        embedder: Supplies the model name and dimension for the manifest.
        max_chars: Character budget used when chunking.
        overlap_chars: Overlap used when chunking.
        exclude_globs: Patterns excluded from the corpus.
        fingerprint: Pre-computed corpus fingerprint. Passed in by callers that
            already computed it, so a large corpus is not read twice per start.
    """
    target = Path(index_dir)
    digest = fingerprint if fingerprint is not None else corpus_fingerprint(kb_dir, exclude_globs)
    manifest = build_manifest(
        fingerprint=digest,
        store=store,
        embedder=embedder,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    if store.is_empty:
        logger.info("not persisting an empty index to %s", target)
        return manifest

    store.save(target, manifest=manifest)
    return manifest
