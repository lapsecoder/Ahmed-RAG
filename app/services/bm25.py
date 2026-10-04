"""A small, dependency-free BM25 lexical index over the knowledge base.

Dense embeddings are good at paraphrase and bad at exact tokens. A question like
"What certifications does Ahmed have?" shares almost no vocabulary with the
chunk that answers it ("### Python Programming for AWS ... - Completed: ..."), so
MiniLM ranked that chunk 36th. BM25 is the opposite: it is exact, cheap and fully
deterministic, which is what the hybrid retriever needs to recover those misses.

The index is built from the *same* chunks the vector store holds, in the same
position order, so a lexical hit maps back to a FAISS position with no extra
bookkeeping. It is rebuilt in memory on every startup: for the portfolio corpus
this is a few hundred microseconds and it means no new persisted artefact and no
new manifest field to invalidate.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from app.core.exceptions import RetrieverError
from app.models.document import DocumentChunk
from app.services.text import content_terms, tokenise

DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75


@dataclass(frozen=True, slots=True)
class LexicalHit:
    """One BM25 match, in vector-store position order."""

    position: int
    score: float
    matched_terms: int = 0
    max_idf: float = 0.0


def lexical_document_text(chunk: DocumentChunk) -> str:
    """Text BM25 indexes for ``chunk``.

    The heading path is included for the same reason
    :func:`app.services.knowledge_base.embedding_text` includes it: a chunk is
    often named by its heading rather than its body, and "Certifications >
    Recorded certifications" must be findable by the word "certifications".
    """
    heading = " > ".join(chunk.heading_path).strip()
    source = chunk.source_file.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("-", " ")
    prefix = f"{source} {heading}".strip()
    return f"{prefix}\n{chunk.text}" if prefix else chunk.text


class Bm25Index:
    """Okapi BM25 over an in-memory posting list."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if k1 < 0:
            raise RetrieverError(f"bm25 k1 must be non-negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise RetrieverError(f"bm25 b must be in [0, 1], got {b}")
        self._k1 = float(k1)
        self._b = float(b)
        self._chunks = tuple(chunks)

        self._term_frequencies: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for chunk in self._chunks:
            terms = tokenise(lexical_document_text(chunk))
            counts = Counter(terms)
            self._term_frequencies.append(counts)
            total_length += len(terms)
            document_frequency.update(counts.keys())

        self._lengths = [sum(counts.values()) for counts in self._term_frequencies]
        count = len(self._chunks)
        self._average_length = (total_length / count) if count else 0.0
        # Lucene-style IDF, always positive, so a term appearing in every
        # document contributes ~0 instead of a negative score.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._chunks)

    @property
    def vocabulary(self) -> frozenset[str]:
        """Every token present anywhere in the indexed corpus.

        The answer layer uses this to tell "asked about something that is simply
        not recorded" apart from "asked using different words". A query term
        absent from the vocabulary can never be answered, which is what makes
        "What is Ahmed's favourite film?" return no information instead of a
        confident quote about the MovieMind project.
        """
        return frozenset(self._idf)

    def vocabulary_for(self, source_files: Collection[str]) -> frozenset[str]:
        """Every token present in the chunks of the named source files.

        The whole-corpus vocabulary answers "could any document talk about
        this?", which is the wrong question when routing has already decided
        which documents own the answer. "What's Ahmed's background?" routes to
        the identity, education and goals documents, and none of them contains
        the word "background" -- but two unrelated documents do, in the sense of
        Celery background *workers*. Judged against the whole corpus the term
        looks supported, so the retriever is left to search with a word that
        leads nowhere, and the question is declined although the knowledge base
        answers it in three places.

        Scoping the vocabulary to the routed documents separates those cases
        without any per-question knowledge: a word the routed documents never
        use is a word the retrievers cannot key on there, and routing becomes
        the only remaining relevance signal.
        """
        wanted = frozenset(source_files)
        terms: set[str] = set()
        for position, chunk in enumerate(self._chunks):
            if chunk.source_file in wanted:
                terms.update(self._term_frequencies[position])
        return frozenset(terms)

    @property
    def average_length(self) -> float:
        """Mean document length in tokens."""
        return self._average_length

    def idf(self, term: str) -> float:
        """Inverse document frequency of ``term``; 0.0 when unknown."""
        return self._idf.get(term, 0.0)

    def score(self, query_terms: Iterable[str], position: int) -> float:
        """BM25 score of one document. 0.0 when nothing matches."""
        counts = self._term_frequencies[position]
        length = self._lengths[position]
        total = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            normalisation = self._k1 * (
                1.0
                - self._b
                + self._b * (length / self._average_length if self._average_length else 1.0)
            )
            total += idf * (frequency * (self._k1 + 1.0)) / (frequency + normalisation)
        return total

    def search(self, query: str, top_k: int) -> list[LexicalHit]:
        """Return up to ``top_k`` documents ranked by descending BM25 score.

        Only documents with a strictly positive score are returned, so a query
        that shares no vocabulary with the corpus yields no lexical hits at all.
        Ties are broken by ascending position, keeping ranking reproducible.
        """
        if top_k <= 0:
            raise RetrieverError(f"top_k must be positive, got {top_k}")
        terms = content_terms(query)
        if not terms or not self._chunks:
            return []

        unique_terms = sorted(set(terms))
        hits: list[LexicalHit] = []
        for position in range(len(self._chunks)):
            counts = self._term_frequencies[position]
            matched = [term for term in unique_terms if counts.get(term)]
            score = self.score(unique_terms, position)
            if score > 0.0:
                hits.append(
                    LexicalHit(
                        position=position,
                        score=score,
                        matched_terms=len(matched),
                        max_idf=max((self._idf.get(term, 0.0) for term in matched), default=0.0),
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, hit.position))
        return hits[:top_k]
