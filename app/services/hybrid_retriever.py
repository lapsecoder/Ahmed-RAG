"""Hybrid retrieval: dense FAISS plus BM25, fused with reciprocal rank fusion.

Why two retrievers
------------------
Neither signal is sufficient on this corpus. MiniLM handles paraphrase but
mis-ranks exact-token questions ("what certifications" vs a chunk whose body only
lists a course name). BM25 handles exact tokens but knows nothing about
paraphrase. Fusing them fixes both failure modes without a learned reranker,
which keeps the whole pipeline deterministic and offline.

Fusion
------
Reciprocal Rank Fusion (RRF) is used deliberately in preference to normalising
and adding the two scores. Cosine similarity and BM25 live on incompatible
scales, so any weighted sum needs a per-query normalisation and a tuned weight;
RRF needs neither. It consumes only *ranks*, so it cannot be skewed by one
retriever producing a much larger raw score than the other:

    fused(d) = sum over rankers of 1 / (k + rank_r(d))

``k`` defaults to 60, the value from the original Cormack et al. paper, which
flattens the influence of the very top of each list.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from app.core.logging import get_logger
from app.models.retrieval import RetrievalResult
from app.services.bm25 import Bm25Index
from app.services.embedder import Embedder
from app.services.retriever import Retriever
from app.services.vector_store import FaissVectorStore

logger = get_logger("services.hybrid_retriever")

#: RRF damping constant. Larger values flatten the head of each ranking.
DEFAULT_RRF_K: int = 60

#: A lexical-only hit needs at least this many distinct query terms in common
#: with the chunk, unless one of them is rare enough to be decisive on its own.
#:
#: Two terms is the default because the subject's own name ("ahmed") appears in
#: most chunks, so a single-term match is usually meaningless. But requiring two
#: unconditionally loses real recall: a question like "What did Ahmed study?"
#: carries only one discriminating term ("study") once the subject name is
#: discounted, and the authoritative chunk matched only that.
DEFAULT_MIN_LEXICAL_TERMS: int = 2

#: The lexical candidate pool is this multiple of ``top_k``. Rank fusion only
#: needs each retriever's ordering, so searching a wider lexical pool costs
#: nothing and recovers chunks the dense head of the list missed entirely -- a
#: section whose text never repeats the question's noun ("Programming" bullets
#: inside skills.md) is invisible to both a dense top-8 and a lexical top-8.
LEXICAL_POOL_FACTOR: int = 3

#: A single matched term is enough when it is this rare. Lucene IDF of 2.0 on
#: this corpus means the term appears in roughly a quarter of the chunks or
#: fewer, so "study", "cgpa" and "internship" qualify while "ahmed" (1.4) and
#: "skill" (1.4) do not.
RARE_TERM_IDF: float = 2.0

#: Sibling chunks added for each document that already contributed a hit.
#:
#: A per-chunk similarity threshold is the wrong tool for a list-shaped answer.
#: "What are Ahmed's skills?" is answered by nine sibling chunks inside
#: skills.md, each of which lists three to seven technologies and none of which
#: mentions the word "skills" in its own body. Measured on the real index, only
#: four of skills.md's thirteen chunks cleared 0.35, so the answer quoted four
#: Retrieval and Security bullets and silently omitted Python, FastAPI and
#: PostgreSQL.
#:
#: Chunk similarity therefore still gates *entry* into a document, but once a
#: document is admitted its siblings are supplied too: they share the document's
#: framing, they are screened by the same injection detector, and the composer
#: still only quotes statements that match the question.
DEFAULT_DOCUMENT_EXPANSION: int = 12

#: Hard ceiling on how many chunks one query may return, expansion included.
#:
#: Document expansion is what makes list-shaped answers complete, but it is
#: applied per admitted document, so a query whose top_k hits land in several
#: documents multiplies: ``top_k`` of 12 with an expansion of 12 returned 62 of
#: the corpus's 89 chunks for "What tech stack did Ahmed use?". The composer
#: then filters that down to a handful of statements, so the extra chunks were
#: never worth their cost in screening time.
#:
#: The cap is applied *after* the fused hits, which are the actual retrieval
#: signal, so the tail that gets dropped is always expanded siblings rather than
#: anything the retrievers ranked. 32 keeps the whole of a 13-chunk document
#: reachable -- skills.md is the largest, and its bullets are the answer to the
#: list-shaped questions expansion exists to serve.
DEFAULT_MAX_CONTEXT_CHUNKS: int = 32


class HybridRetriever(Retriever):
    """Dense + lexical retrieval fused by reciprocal rank fusion.

    Admission rules, in order:

    1. a chunk is admitted when its dense cosine clears ``similarity_threshold``,
       exactly as before; or
    2. it is admitted on lexical evidence alone when BM25 ranks it and at least
       ``min_lexical_terms`` distinct query terms are shared.

    Ordering is always by fused score, with the FAISS position as a
    deterministic tie-break.
    """

    def __init__(
        self,
        store: FaissVectorStore,
        embedder: Embedder,
        *,
        bm25: Bm25Index | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        min_lexical_terms: int = DEFAULT_MIN_LEXICAL_TERMS,
        similarity_threshold: float = 0.35,
        top_k: int = 8,
        document_expansion: int = DEFAULT_DOCUMENT_EXPANSION,
        max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    ) -> None:
        super().__init__(
            store,
            embedder,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        if document_expansion < 0:
            raise ValueError(f"document_expansion must be >= 0, got {document_expansion}")
        if min_lexical_terms < 0:
            raise ValueError(f"min_lexical_terms must be >= 0, got {min_lexical_terms}")
        if max_context_chunks <= 0:
            raise ValueError(f"max_context_chunks must be positive, got {max_context_chunks}")
        self._bm25 = bm25
        self._rrf_k = int(rrf_k)
        self._min_lexical_terms = int(min_lexical_terms)
        self._document_expansion = int(document_expansion)
        self._max_context_chunks = int(max_context_chunks)

    @classmethod
    def from_store(
        cls,
        store: FaissVectorStore,
        embedder: Embedder,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        min_lexical_terms: int = DEFAULT_MIN_LEXICAL_TERMS,
        similarity_threshold: float = 0.35,
        top_k: int = 8,
        document_expansion: int = DEFAULT_DOCUMENT_EXPANSION,
        max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    ) -> HybridRetriever:
        """Build a hybrid retriever whose lexical index mirrors ``store``.

        The lexical index is built from the chunks the store already holds, in
        store order, so lexical positions and FAISS positions are identical and
        no new persisted artefact or manifest field is needed.
        """
        return cls(
            store,
            embedder,
            bm25=Bm25Index(store.chunks),
            rrf_k=rrf_k,
            min_lexical_terms=min_lexical_terms,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            document_expansion=document_expansion,
            max_context_chunks=max_context_chunks,
        )

    @property
    def bm25(self) -> Bm25Index | None:
        """The lexical index, when one is attached."""
        return self._bm25

    @property
    def rrf_k(self) -> int:
        """The reciprocal-rank-fusion damping constant."""
        return self._rrf_k

    @property
    def min_lexical_terms(self) -> int:
        """Distinct query terms a lexical-only hit must share."""
        return self._min_lexical_terms

    @property
    def document_expansion(self) -> int:
        """Sibling chunks supplied for each admitted document."""
        return self._document_expansion

    @property
    def max_context_chunks(self) -> int:
        """Hard ceiling on the number of chunks one query may return."""
        return self._max_context_chunks

    def retrieve(
        self,
        query: str,
        *,
        restrict_to: Collection[str] | None = None,
        relax_threshold: bool = False,
    ) -> list[RetrievalResult]:
        """Fuse dense and lexical rankings into one deterministic result list.

        Args:
            query: The user's question.
            restrict_to: When given, only chunks from these ``source_file`` values
                may be admitted or expanded.
            relax_threshold: Admit permitted documents even below
                ``similarity_threshold``. Sound only alongside ``restrict_to``;
                see :meth:`~app.services.retriever.Retriever.retrieve`.
        """
        permitted = frozenset(restrict_to) if restrict_to is not None else None
        relax = relax_threshold and permitted is not None
        dense_hits: dict[int, float] = {}
        if not self._store.is_empty:
            vector = self.embed_query(query)
            depth = self._store.size if permitted is not None else self._top_k
            for hit in self._store.search(vector, depth):
                dense_hits[hit.position] = hit.similarity
        else:
            logger.info("index is empty; skipping query embedding")

        lexical_hits: dict[int, float] = {}
        if self._bm25 is not None:
            pool = max(self._top_k * LEXICAL_POOL_FACTOR, self._top_k)
            for lexical_hit in self._bm25.search(query, pool):
                decisive = (
                    lexical_hit.matched_terms >= self._min_lexical_terms
                    or lexical_hit.max_idf >= RARE_TERM_IDF
                )
                if not decisive:
                    continue
                if permitted is not None:
                    chunk = self._store.chunk_at(lexical_hit.position)
                    if chunk.source_file not in permitted:
                        continue
                lexical_hits[lexical_hit.position] = lexical_hit.score

        if not dense_hits and not lexical_hits:
            logger.debug("no dense or lexical evidence for the query")
            return []

        # Dense admission keeps the pre-existing threshold contract intact.
        admitted = {
            position
            for position, similarity in dense_hits.items()
            if similarity >= self._threshold or relax
        }
        # Lexical-only admission: the chunk was ranked by BM25 and shares enough
        # distinct query terms to count as a real match.
        admitted.update(lexical_hits)
        if not admitted:
            logger.debug("no chunk cleared the similarity threshold %.3f", self._threshold)
            return []

        dense_order = sorted(
            (position for position in admitted if position in dense_hits),
            key=lambda position: (-dense_hits[position], position),
        )
        lexical_order = sorted(
            (position for position in admitted if position in lexical_hits),
            key=lambda position: (-lexical_hits[position], position),
        )

        dense_rank = {position: rank for rank, position in enumerate(dense_order, start=1)}
        lexical_rank = {position: rank for rank, position in enumerate(lexical_order, start=1)}

        fused: list[tuple[float, int]] = []
        for position in admitted:
            score = 0.0
            if position in dense_rank:
                score += 1.0 / (self._rrf_k + dense_rank[position])
            if position in lexical_rank:
                score += 1.0 / (self._rrf_k + lexical_rank[position])
            fused.append((score, position))
        fused.sort(key=lambda pair: (-pair[0], pair[1]))

        # The scope filter is applied *before* ``top_k`` truncates, not after. Filtering
        # afterwards throws away the tail of the ranking for being out of scope and
        # then returns fewer results than asked for, so a scoped search over the
        # identity documents would lose most of them to an unrelated document that
        # happened to fuse higher.
        if permitted is not None:
            fused = [
                pair for pair in fused if self._store.chunk_at(pair[1]).source_file in permitted
            ]

        results: list[RetrievalResult] = []
        for score, position in fused[: self._top_k]:
            chunk = self._store.chunk_at(position)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    # 0.0 for lexical-only hits: no embedding similarity exists.
                    similarity=dense_hits.get(position, 0.0),
                    position=position,
                    rank=len(results),
                    lexical_score=lexical_hits.get(position, 0.0),
                    fused_score=score,
                )
            )
        return self._expand_documents(results, dense_hits)

    def _expand_documents(
        self,
        results: Sequence[RetrievalResult],
        dense_hits: Mapping[int, float],
    ) -> list[RetrievalResult]:
        """Append the remaining chunks of every document that already hit.

        Admission is unchanged: a chunk still has to clear the similarity
        threshold, or be admitted on lexical evidence, before its document is
        considered relevant. Expansion only adds the *rest* of a document that
        already proved relevant, because a list-shaped answer is spread over
        sibling chunks that each carry only part of the list and none of which
        repeat the question's own noun.

        See :data:`DEFAULT_DOCUMENT_EXPANSION` for the measurement that
        motivated this, and :data:`DEFAULT_MAX_CONTEXT_CHUNKS` for the ceiling
        that keeps the result set compact.
        """
        if self._document_expansion == 0 or not results:
            return list(results)

        selected = {result.position for result in results}
        positions_by_file: dict[str, list[int]] = {}
        for position, chunk in enumerate(self._store.chunks):
            positions_by_file.setdefault(chunk.source_file, []).append(position)

        # Expansion never displaces a chunk the retrievers actually ranked, so
        # the budget is whatever is left of the ceiling after the real hits.
        budget = self._max_context_chunks - len(results)
        expanded: list[RetrievalResult] = []
        # Documents are expanded best-first, by the score of their own best fused
        # hit, with the file name as a deterministic tie-break. Alphabetical
        # order looks harmless but starves the document the query is actually
        # about: with about.md and availability.md ahead of skills.md, a budget
        # of 20 was spent on the first two and "What are Ahmed's skills?" lost
        # the Python/FastAPI/PostgreSQL bullets entirely.
        best_score: dict[str, float] = {}
        for result in results:
            source = result.chunk.source_file
            best_score[source] = max(best_score.get(source, 0.0), result.fused_score)
        documents = sorted(best_score, key=lambda source: (-best_score[source], source))
        for source_file in documents:
            if budget <= 0:
                break
            siblings = [
                position for position in positions_by_file[source_file] if position not in selected
            ]
            for position in siblings[: min(self._document_expansion, budget)]:
                selected.add(position)
                budget -= 1
                expanded.append(
                    RetrievalResult(
                        chunk=self._store.chunks[position],
                        # Siblings were admitted by their document, not by their own
                        # score, so a score is reported only when the dense head of
                        # the ranking happened to reach them.
                        similarity=dense_hits.get(position, 0.0),
                        position=position,
                        rank=len(results) + len(expanded),
                        lexical_score=0.0,
                        fused_score=0.0,
                    )
                )
        if expanded:
            logger.info(
                "expanded %d hit(s) with %d sibling chunk(s) from the same document(s)",
                len(results),
                len(expanded),
            )
        return [*results, *expanded]
