"""Shared test fixtures and offline doubles.

No fixture here touches the network, downloads a model, starts a server, or
reads a populated knowledge base. The only external dependency the unit suite
uses is the real ``faiss`` library, because FAISS behaviour is precisely what the
store tests are meant to pin down.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from app.config import Settings
from app.container import Container
from app.models.chat import ChatResult
from app.models.document import DocumentChunk
from app.models.index import IndexBuildResult, IndexManifest
from app.security.classification import QueryClassifier
from app.security.injection import InjectionDetector
from app.security.output_validator import OutputValidator
from app.security.prompt_builder import PromptBuilder
from app.services.answering.composer import AnswerComposer
from app.services.answering.corpus import CorpusProfile
from app.services.bm25 import Bm25Index
from app.services.chat import ChatService
from app.services.llm import LLMProvider
from app.services.retriever import Retriever
from app.services.text import normalise_key
from app.services.vector_store import FaissVectorStore

# --------------------------------------------------------------------------- #
# Embedding doubles
# --------------------------------------------------------------------------- #


class ScriptedEmbedder:
    """Deterministic embedder that maps a known vocabulary to fixed vectors.

    The mapping is a *mock* of the sentence-transformer model: it never loads
    weights, so tests stay offline and instant.
    """

    def __init__(
        self,
        dimension: int = 8,
        vectors: dict[str, np.ndarray] | None = None,
        *,
        model_name: str = "mock/minilm",
    ) -> None:
        self._dimension = dimension
        self.vectors: dict[str, np.ndarray] = dict(vectors or {})
        self._model_name = model_name
        self.queries: list[str] = []
        self.documents: list[list[str]] = []
        self.query_calls = 0
        self.document_calls = 0

    @property
    def model_name(self) -> str:
        """Name reported by the health endpoint."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Configured vector width."""
        return self._dimension

    def _vector_for(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return np.asarray(self.vectors[text], dtype=np.float32)
        # Seeded from a stable digest, not ``hash()``. Python randomises string
        # hashing per process, so ``hash(text)`` gave this "deterministic" embedder
        # a different vector on every run: the same test passed eight times in a
        # row and then failed, because the generated query vector happened to have
        # a negative component along the chunk's axis. A double whose whole
        # purpose is reproducibility has to survive a change of PYTHONHASHSEED.
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "big") % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self._dimension).astype(np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).astype(np.float32)

    def embed_documents(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        """Embed a batch."""
        self.document_calls += 1
        self.documents.append(list(texts))
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack([self._vector_for(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query, recording the call for assertions."""
        self.query_calls += 1
        self.queries.append(text)
        return self._vector_for(text)


def unit_vector(values: list[float]) -> np.ndarray:
    """Build an already-normalised float32 unit vector."""
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return vector.astype(np.float32)


class FakeEncoding:
    """One tokenised row, matching what ``tokenizers`` hands back."""

    def __init__(self, ids: list[int], type_ids: list[int] | None = None) -> None:
        self.ids = ids
        self.type_ids = type_ids if type_ids is not None else [0] * len(ids)


class FakeTokenizer:
    """Stand-in for ``tokenizers.Tokenizer``.

    Deterministic token ids derived from the text, so a test can assert on what
    was passed through without a real vocabulary.
    """

    def __init__(self) -> None:
        self.truncation: int | None = None
        self.padding_disabled = False
        self.encode_calls: list[list[str]] = []

    def enable_truncation(self, max_length: int) -> None:
        self.truncation = max_length

    def no_padding(self) -> None:
        self.padding_disabled = True

    def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
        self.encode_calls.append(list(texts))
        out = []
        for text in texts:
            ids = [(sum(ord(char) for char in text) + index) % 30_000 + 1 for index in range(3)]
            out.append(FakeEncoding(ids))
        return out


class FakeOnnxSession:
    """Stand-in for ``onnxruntime.InferenceSession``.

    Returns token vectors that are identical across the sequence, so mean
    pooling over the attention mask returns exactly the vector under test and a
    failure points at pooling rather than at the pooling input.
    """

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.run_calls: list[dict[str, np.ndarray]] = []
        self.override: np.ndarray | None = None

    def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.run_calls.append({key: np.array(value) for key, value in feed.items()})
        batch = int(feed["input_ids"].shape[0])
        sequence = int(feed["input_ids"].shape[1])
        if self.override is not None:
            return [self.override]
        hidden = np.zeros((batch, sequence, self.dimension), dtype=np.float32)
        for row in range(batch):
            # Deterministic per-row value derived from the first token id, so the
            # mean-pooled output is a real, distinct vector per input row.
            base = float(int(feed["input_ids"][row, 0]) % 97) + 1.0
            hidden[row, :, :] = base
        return [hidden]


@pytest.fixture
def fake_onnx_model() -> tuple[FakeOnnxSession, FakeTokenizer]:
    """A ready-made ``(session, tokenizer)`` pair for injection."""
    return FakeOnnxSession(), FakeTokenizer()


# --------------------------------------------------------------------------- #
# LLM doubles
# --------------------------------------------------------------------------- #


class RecordingLLM(LLMProvider):
    """LLM double that records every prompt it receives."""

    def __init__(self, response: str = "Ahmed is a software engineer.") -> None:
        self._response = response
        self.prompts: list[str] = []
        self.calls = 0

    @property
    def model_name(self) -> str:
        """Name reported by the health endpoint."""
        return "mock/llm"

    def generate(self, prompt: str) -> str:
        """Record the prompt and return the canned answer."""
        self.calls += 1
        self.prompts.append(prompt)
        return self._response


class ExplodingLLM(RecordingLLM):
    """LLM double that fails the test if it is ever called."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - must not run
        """Always fail: this double exists to prove a path never generates."""
        raise AssertionError("the LLM must not be called on this path")


# --------------------------------------------------------------------------- #
# Chunks and stores
# --------------------------------------------------------------------------- #

SAMPLE_MARKDOWN = """# Ahmed

A short introduction paragraph.

## Projects

### RAG Backend

A local-first retrieval augmented generation backend.

Details that make the paragraph long enough to be split into more than one
unit so that the overlap behaviour can actually be observed by the tests.
"""


def make_chunk(
    text: str = "sample chunk text",
    *,
    source_file: str = "about.md",
    section: str = "About",
    ordinal: int = 0,
) -> DocumentChunk:
    """Build a :class:`DocumentChunk` for tests."""
    return DocumentChunk(
        chunk_id=f"test-{ordinal:04d}",
        source_file=source_file,
        section=section,
        text=text,
        heading_path=(section,),
        heading_level=2,
        ordinal=ordinal,
    )


@pytest.fixture
def settings(tmp_path: Any) -> Settings:
    """Isolated settings backed by a temporary directory."""
    return Settings(
        kb_dir=tmp_path / "kb",
        index_dir=tmp_path / "index",
        embedding_model="mock/minilm",
        similarity_threshold=0.35,
        retrieval_top_k=5,
        chunk_max_chars=400,
        chunk_overlap_chars=80,
    )


@pytest.fixture
def detector() -> InjectionDetector:
    """Production injection detector."""
    return InjectionDetector()


@pytest.fixture
def classifier(detector: InjectionDetector) -> QueryClassifier:
    """Production classifier with the default topic vocabulary."""
    return QueryClassifier(detector)


@pytest.fixture
def prompt_builder(detector: InjectionDetector) -> PromptBuilder:
    """Production prompt builder."""
    return PromptBuilder(detector=detector)


@pytest.fixture
def output_validator() -> OutputValidator:
    """Production output validator."""
    return OutputValidator()


@pytest.fixture
def embedder() -> ScriptedEmbedder:
    """Deterministic 8-dimensional embedder."""
    return ScriptedEmbedder(dimension=8)


def build_retriever(
    embedder: ScriptedEmbedder,
    store: FaissVectorStore,
    *,
    threshold: float = 0.35,
    top_k: int = 5,
) -> Retriever:
    """Wire a retriever around the given store and embedder."""
    return Retriever(store, embedder, similarity_threshold=threshold, top_k=top_k)


# --------------------------------------------------------------------------- #
# Composer doubles
# --------------------------------------------------------------------------- #

#: Frontmatter category for each document the shared test corpus provides. The
#: composer routes on these, exactly as it does for the real knowledge base.
TEST_CATEGORIES: dict[str, str] = {
    "about.md": "profile",
    "skills.md": "skills",
    "projects/resumeforge.md": "project",
}


def make_profile(
    categories: dict[str, str] | None = None,
    projects: dict[str, str] | None = None,
) -> CorpusProfile:
    """Build a corpus profile in memory, without touching the filesystem.

    Args:
        categories: ``source_file`` -> frontmatter category.
        projects: Display name -> ``source_file`` for project documents.
    """
    mapping = dict(TEST_CATEGORIES if categories is None else categories)
    names = {normalise_key(name): name for name in (projects or {})}
    return CorpusProfile(
        categories=mapping,
        project_names=names,
        project_files=dict(projects or {}),
    )


def make_composer(
    profile: CorpusProfile | None = None,
    store: FaissVectorStore | None = None,
) -> AnswerComposer:
    """Wire a composer, borrowing the store's chunks for lexical weighting.

    Passing no ``store`` yields a composer with a neutral term weight, which is
    exactly the "no lexical prior available" case the class documents.
    """
    if store is not None and store.chunks:
        index = Bm25Index(store.chunks)
        return AnswerComposer(
            profile or make_profile(),
            idf=index.idf,
            vocabulary=index.vocabulary,
        )
    return AnswerComposer(profile or make_profile())


def build_chat_service(
    *,
    retriever: Retriever,
    detector: InjectionDetector,
    profile: CorpusProfile | None = None,
    composer: AnswerComposer | None = None,
) -> ChatService:
    """Wire a chat service around the given retriever and composer."""
    resolved = composer if composer is not None else make_composer(profile)
    return ChatService(
        retriever=retriever,
        composer=resolved,
        classifier=QueryClassifier(detector),
        output_validator=OutputValidator(),
        detector=detector,
    )


@pytest.fixture
def chat_service(
    embedder: ScriptedEmbedder,
    detector: InjectionDetector,
) -> Iterator[ChatService]:
    """Chat service backed by a real FAISS index and the real composer."""
    text = "Ahmed builds local RAG systems with Python and FastAPI."
    store = FaissVectorStore(embedder.dimension)
    vectors = embedder.embed_documents([text])
    store.add([make_chunk(text)], vectors)
    retriever = build_retriever(embedder, store, threshold=-1.0)
    yield build_chat_service(
        retriever=retriever,
        detector=detector,
        composer=make_composer(make_profile({"about.md": "profile"}), store),
    )


def make_container(
    settings: Settings,
    store: FaissVectorStore,
    embedder: ScriptedEmbedder,
    detector: InjectionDetector,
) -> Container:
    """Assemble a container without touching the real embedding model."""
    classifier = QueryClassifier(detector, settings.topic_keywords)
    retriever = build_retriever(embedder, store, threshold=settings.similarity_threshold)
    composer = make_composer(make_profile(), store)
    source_files = tuple(sorted({chunk.source_file for chunk in store.chunks}))
    return Container(
        settings=settings,
        detector=detector,
        classifier=classifier,
        embedder=embedder,
        store=store,
        retriever=retriever,
        composer=composer,
        output_validator=OutputValidator(),
        chat_service=build_chat_service(
            retriever=retriever,
            detector=detector,
            composer=composer,
        ),
        index_result=IndexBuildResult(
            documents=len(source_files),
            chunks=store.size,
            dimension=store.dimension,
            source_files=source_files,
        ),
        index_manifest=IndexManifest(
            fingerprint="injected" * 8,
            chunks=max(store.size, 1),
            dimension=store.dimension,
            embedding_model=embedder.model_name,
            source_files=source_files,
            chunk_max_chars=settings.chunk_max_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
        ),
        index_reused=False,
    )


@pytest.fixture
def container(
    settings: Settings,
    embedder: ScriptedEmbedder,
    detector: InjectionDetector,
) -> Container:
    """Container wired with an empty index and the real composer."""
    store = FaissVectorStore(embedder.dimension)
    return make_container(settings, store, embedder, detector)


def make_result(response: str) -> ChatResult:
    """Build a minimal :class:`ChatResult` for assertions."""
    from app.models.enums import ChatOutcome, QueryClassification

    return ChatResult(
        response=response,
        classification=QueryClassification.IN_SCOPE,
        outcome=ChatOutcome.ANSWERED,
    )
