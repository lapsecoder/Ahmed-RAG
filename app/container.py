"""Application container: the single place where the object graph is built.

The container is created once during application startup and reused for every
request, so the embedding model and the FAISS index are loaded exactly once.

Index lifecycle at startup:

1. Fingerprint the corpus (paths + contents, never mtimes).
2. Load a persisted index when its manifest proves it was built from exactly
   this corpus, this embedding model and these chunking settings.
3. Otherwise rebuild, and persist the new index with a fresh manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.models.index import IndexBuildResult, IndexManifest
from app.security.classification import QueryClassifier
from app.security.injection import InjectionDetector
from app.security.output_validator import OutputValidator
from app.services.answering.composer import AnswerComposer
from app.services.answering.corpus import build_profile
from app.services.bm25 import Bm25Index
from app.services.chat import ChatService
from app.services.embedder import Embedder, SentenceTransformerEmbedder
from app.services.hybrid_retriever import HybridRetriever
from app.services.knowledge_base import (
    build_index_from_directory,
    corpus_fingerprint,
    list_markdown_files,
    save_index,
)
from app.services.retriever import Retriever
from app.services.vector_store import FaissVectorStore

logger = get_logger("container")


@dataclass(frozen=True, slots=True)
class Container:
    """Every long-lived dependency the request handlers need."""

    settings: Settings
    detector: InjectionDetector
    classifier: QueryClassifier
    embedder: Embedder
    store: FaissVectorStore
    retriever: Retriever
    composer: AnswerComposer
    output_validator: OutputValidator
    chat_service: ChatService
    index_result: IndexBuildResult
    index_manifest: IndexManifest
    index_reused: bool

    @property
    def knowledge_base_documents(self) -> int:
        """Number of Markdown files currently in the knowledge base."""
        return len(list_markdown_files(self.settings.kb_dir, self.settings.kb_exclude_globs))

    @property
    def indexed_source_files(self) -> tuple[str, ...]:
        """Knowledge-base-relative paths whose chunks are in the index."""
        return self.index_manifest.source_files

    @property
    def index_dir(self) -> Path:
        """Where the persisted index lives."""
        return self.settings.index_dir


def build_container(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    build_from_directory: bool = True,
) -> Container:
    """Construct the full object graph.

    No language model is constructed or contacted. Retrieval is dense FAISS plus
    BM25 fused by reciprocal rank fusion, and answers are assembled by the
    deterministic composer.

    Args:
        settings: Configuration; the process singleton by default.
        embedder: Injection point for tests. When omitted, the real
            ``sentence-transformers`` model is loaded.
        build_from_directory: When ``False``, an existing index is not rebuilt
            from the knowledge base.

    Returns:
        A ready-to-use :class:`Container`.

    Raises:
        VectorStoreError: If a saved index cannot be loaded and no embedder
            dimension is available to rebuild it.
    """
    settings = settings or get_settings()
    detector = InjectionDetector()
    classifier = QueryClassifier(detector, settings.topic_keywords)
    output_validator = OutputValidator()

    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    store, index_result, manifest, reused = _prepare_store(settings, embedder, build_from_directory)

    bm25 = Bm25Index(store.chunks)
    retriever = HybridRetriever(
        store,
        embedder,
        bm25=bm25,
        similarity_threshold=settings.similarity_threshold,
        top_k=settings.retrieval_top_k,
    )
    composer = AnswerComposer(
        build_profile(settings.kb_dir, settings.kb_exclude_globs),
        idf=bm25.idf,
        # The vocabulary is what lets the composer tell "asked in words the
        # corpus does not use" apart from "asked about something never recorded".
        vocabulary=bm25.vocabulary,
        # ...and the scoped form answers the same question about the documents
        # routing selected, which is the only evidence a given answer can come
        # from. See AnswerComposer.has_lexical_support.
        scoped_vocabulary=bm25.vocabulary_for,
    )
    chat_service = ChatService(
        retriever=retriever,
        composer=composer,
        classifier=classifier,
        output_validator=output_validator,
        detector=detector,
    )
    return Container(
        settings=settings,
        detector=detector,
        classifier=classifier,
        embedder=embedder,
        store=store,
        retriever=retriever,
        composer=composer,
        output_validator=output_validator,
        chat_service=chat_service,
        index_result=index_result,
        index_manifest=manifest,
        index_reused=reused,
    )


def _prepare_store(
    settings: Settings,
    embedder: Embedder,
    build_from_directory: bool,
) -> tuple[FaissVectorStore, IndexBuildResult, IndexManifest, bool]:
    """Load a valid persisted index, or rebuild the corpus from Markdown.

    Returns:
        The store, a build summary, the manifest describing the index, and
        whether a persisted index was reused instead of rebuilt.
    """
    fingerprint = corpus_fingerprint(settings.kb_dir, settings.kb_exclude_globs)

    if settings.index_file.is_file() and settings.chunks_file.is_file():
        try:
            store = FaissVectorStore.load(settings.index_dir)
        except VectorStoreError as exc:
            logger.warning("could not load saved index (%s); rebuilding", exc)
        else:
            manifest = store.manifest
            if manifest is None:
                logger.warning("saved index has no usable manifest; rebuilding")
            elif not manifest.matches(
                fingerprint=fingerprint,
                dimension=embedder.dimension,
                embedding_model=embedder.model_name,
                chunk_max_chars=settings.chunk_max_chars,
                chunk_overlap_chars=settings.chunk_overlap_chars,
            ):
                logger.info("knowledge base or settings changed; rebuilding the index")
            else:
                logger.info("reusing saved index with %d chunk(s)", store.size)
                return (
                    store,
                    IndexBuildResult(
                        documents=len(manifest.source_files),
                        chunks=store.size,
                        dimension=store.dimension,
                        source_files=manifest.source_files,
                        fingerprint=manifest.fingerprint,
                    ),
                    manifest,
                    True,
                )

    if embedder.dimension <= 0:
        raise VectorStoreError("cannot determine the embedding dimension for the index")

    store = FaissVectorStore(embedder.dimension)
    if build_from_directory:
        result = build_index_from_directory(
            settings.kb_dir,
            store,
            embedder,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
            exclude_globs=settings.kb_exclude_globs,
        )
    else:
        result = IndexBuildResult(
            documents=0, chunks=0, dimension=store.dimension, fingerprint=fingerprint
        )
    manifest = save_index(
        store,
        settings.index_dir,
        kb_dir=settings.kb_dir,
        embedder=embedder,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
        exclude_globs=settings.kb_exclude_globs,
        fingerprint=fingerprint,
    )
    logger.info("indexed %d chunk(s) from the knowledge base", result.chunks)
    return store, result.model_copy(update={"fingerprint": fingerprint}), manifest, False
