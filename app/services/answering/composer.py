"""The deterministic answer composer: grounded extraction with no model.

Every sentence this module emits is a sentence that already exists in a
retrieved, approved knowledge-base chunk. There is no generation step, so the
usual failure modes of a grounded LLM -- inventing a plausible credential,
paraphrasing a number, quoting an instruction found in the context -- are
structurally impossible rather than merely discouraged.

Pipeline
--------
1. **Route.** :func:`~app.services.answering.domains.detect_domain` decides which
   part of the corpus can vouch for the question, and
   :func:`~app.services.answering.domains.detect_project` picks out a named
   project. If neither resolves, the composer returns ``None`` rather than
   quoting whatever happened to rank highest.
2. **Restrict.** Only blocks from the routed documents are eligible. This is what
   stops a question about a personal preference being answered with text about a
   project that happens to share a noun.
3. **Split and filter.** Each block becomes atomic statements; directives,
   meta-rules and ``flagged`` blocks are dropped.
4. **Select.** Statements matching the question's distinctive terms are ranked by
   inverse-document-frequency weight. When a routed domain has no matching
   statement -- the normal case for "what are his skills?", where the answer is a
   list of items that never repeat the word "skills" -- the lead of the domain's
   best-ranked document is quoted instead.
5. **Assemble.** Surviving statements are de-duplicated and rendered.

The security property worth stating plainly: retrieved text is only ever treated
as data. It cannot introduce a rule, cannot veto an answer and cannot add a fact,
because the composer copies characters and nothing else. Blocks arrive already
through :func:`~app.security.context.build_context`, so they are detector-flagged
and delimiter-neutralised before they are seen here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from app.core.logging import get_logger
from app.security.context import ContextBlock
from app.services.answering.corpus import CorpusProfile
from app.services.answering.domains import (
    CROSS_CUTTING_CATEGORY,
    Domain,
    detect_domain,
    detect_project,
    detect_unrecognised_project,
    is_overview_query,
    is_part_of_project_name,
    query_words,
)
from app.services.answering.intents import (
    WHOLE_DOCUMENT_INTENTS,
    AnswerShape,
    Intent,
    Routing,
    is_overview_intent,
    overview_framing_terms,
    route,
)
from app.services.text import (
    EvidenceUnit,
    content_terms,
    is_answerable_evidence,
    is_negative_evidence,
    normalise_key,
    split_evidence_units,
    terms_match,
)

logger = get_logger("services.answering.composer")

#: Statements shorter than this are fragments rather than answers.
MIN_STATEMENT_CHARS: int = 12
#: Longest single statement quoted. Guards against a wall of text.
MAX_STATEMENT_CHARS: int = 320
#: Maximum statements in a focused answer.
MAX_STATEMENTS: int = 5
#: Maximum statements when the question asks for a list.
MAX_LIST_STATEMENTS: int = 9
#: Maximum milestones quoted for an overview of a *record* -- an academic
#: history, a career so far.
#:
#: This is deliberately its own budget rather than
#: :data:`MAX_LIST_STATEMENTS`. A list question and a timeline are different
#: shapes, and borrowing the list budget cost the timeline its own tail: the
#: education record is eleven recorded lines -- four ``label: value`` rows for the
#: school, five for the diploma, and the two recorded lines in "Not on record" --
#: so the ninth slot landed on "Field of study" and the CGPA of 9.11 was never
#: reached. A journey is only worth quoting if it reaches the end of the record.
#:
#: Twelve leaves one slot of headroom over the record as it stands today, which
#: is enough for a stage to be added without this being the thing that hides it.
JOURNEY_MILESTONE_BUDGET: int = 12
#: A statement must score at least this fraction of the best statement's score to
#: join the answer, so one strong hit cannot drag in weak relatives.
RELATIVE_SCORE_FLOOR: float = 0.55
#: Jaccard similarity above which two statements count as duplicates.
DUPLICATE_JACCARD: float = 0.5
#: Minimum inverse-document-frequency weight for a term to count as *distinctive*.
#: Below this a term is so common that matching it says nothing: "Ahmed" appears
#: throughout the corpus and "skills" in every heading of skills.md. A match on
#: such a term is real but uninformative, so it must not by itself justify an
#: answer -- otherwise every question would match every document.
MIN_USEFUL_IDF: float = 1.6

#: Domains whose answers are naturally list-shaped.
_LISTING_DOMAINS: Final[frozenset[Domain]] = frozenset(
    {Domain.SKILLS, Domain.CERTIFICATIONS, Domain.GOALS, Domain.INTERESTS}
)
#: Sentences in a prose answer. Three is what "Ahmed is ..., he is ..., he
#: describes himself as ..." needs; more is a document, not an answer.
MAX_PROSE_CLAUSES: Final[int] = 3

#: How far into a document the lead path will look for a quotable block before
#: giving up on it. A document may open with a ``label: value`` row or a heading
#: line that carries nothing quotable, and a single-block look would then find
#: no statement in a document the answer is entitled to use.
LEAD_BLOCK_SEARCH: Final[int] = 5

#: The name used to restore a dropped antecedent at the head of a paragraph. It
#: is the subject's own name, taken from the corpus rather than invented, and it
#: is only ever *prepended* -- never substituted for anything the corpus says.
_SUBJECT_NAME: Final[str] = "Ahmed"

#: Statements whose subject is the document rather than the person.
#:
#: Documents open by describing themselves: "They describe how Ahmed spends time
#: away from study and project work", "Skills are grouped by how strongly they
#: are evidenced". As a bullet such a line is merely first in a list; in prose it
#: becomes the answer, and "Who is Ahmed?" opens with a description of interests.md.
#: The pattern is anchored on a document noun or a bare plural pronoun followed by
#: a verb of description, which is what distinguishes these from "They prefer
#: clear writing", which is about Ahmed.
#: Every branch is closed with ``\b`` and not just the last one. Without it,
#: "the record" matched inside "**The record**ed CGPA is 9.11", so the one
#: statement that actually answered "What is Ahmed's CGPA?" was classified as
#: document scaffolding and dropped from the prose. The trailing boundary is the
#: difference between a sentence about the file and a sentence about the subject.
_DOCUMENT_SCAFFOLDING_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:"
    r"(?:this|these|the|that)\s+"
    r"(?:document|documents|section|sections|page|pages|file|files|chunk|chunks|"
    r"knowledge[- ]base|record|records|entry|entries|list|lists|table|tables)\b"
    r"|they\s+(?:describe|document|list|record|cover|group|are\s+grouped|separate)\b"
    r"|are\s+grouped\b"
    r")",
    re.IGNORECASE,
)

#: Openings that need the subject restored once the sentence that introduced it
#: has been dropped.
#:
#: Subject pronouns only. A possessive already names its owner -- "His full name
#: is Mohammed Ayaan Ahmed" becomes "Ahmed his full name is Mohammed Ayaan
#: Ahmed" if it is prefixed too, which is not English.
_ANAPHORIC_OPENING_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:he|she|they|it)\b", re.IGNORECASE
)

#: Of the pronouns above, only these refer to the subject as a person. "It" refers
#: to the thing being described, and "It is not recorded" is a statement about the
#: record rather than about Ahmed.
_SUBJECT_PRONOUNS: Final[frozenset[str]] = frozenset({"he", "she", "they"})
#: Question words that signal a list-shaped answer.
_LIST_CUES: Final[frozenset[str]] = frozenset({"list", "all", "which", "many", "what"})

#: The subject's own name. Excluded when judging whether a retrieved block is
#: about the question, because it appears in nearly every chunk of the knowledge
#: base and so cannot distinguish one subject matter from another. ``he``/``his``
#: are already dropped as weak query terms.
_SUBJECT_ALIASES: Final[frozenset[str]] = frozenset({"ahmed", "ayaan", "mohammed"})

#: Sections that tell the assistant how to handle a document rather than what is
#: in it. They are instructions to the reader, so quoting them answers a question
#: nobody asked: "What are Ahmed's interests?" was answered with
#: "He has not listed any competitive gaming, esports or publishing activity.",
#: a bullet from interests.md's "How to treat these" section.
#:
#: "Not on record" and "Stated limitations" are deliberately absent: those are
#: statements about the record itself, which is exactly what a user asking what is
#: not on file needs to hear.
_META_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|>\s*)(?:how to treat|knowledge boundaries|guardrails|rules|levels)\b",
    re.IGNORECASE,
)

#: A bullet of the form "**Completed:** 6 September 2026". These are attributes
#: of a thing, not entries in a list, so they must not lead a list-shaped answer:
#: "What certifications does Ahmed have?" led with "Duration: 10.5 total hours"
#: instead of naming the course.
_FIELD_ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*[-*+]?\s*\*{0,2}[A-Z][\w /&.-]{0,38}\*{0,2}:\s+\S"
)

#: Field labels that describe somebody other than Ahmed.
#:
#: A ``label: value`` row is the recorded attribute of a thing. "Issuing platform:
#: Infosys Springboard" and "Completed: 6 September 2026" are attributes of *his*
#: certification and belong in the answer. "Instructor: Neal Davis, AWS Certified
#: Solutions Architect and Developer" and "Co-instructor: Eric E. Huerta" are
#: attributes of third parties, and quoting them under "What certifications does
#: Ahmed have?" tells the reader about his teachers' credentials.
#:
#: A closed list is deliberate. Deciding "is this row about the subject?" in
#: general needs a model, and this pipeline has none; enumerating the labels that
#: occurred is auditable and cannot start dropping rows nobody has seen.
_THIRD_PARTY_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*[-*+]?\s*\*{0,2}("
    r"instructors?|co-?instructors?|teachers?|trainers?|mentors?|supervisors?|"
    r"coordinators?|reviewers?|authors?|verified\s+by|"
    r"reference(?:\s+number)?|ref\b"
    r")\*{0,2}:",
    re.IGNORECASE,
)

#: Field labels whose value is an opaque identifier rather than a readable fact.
#:
#: These *are* attributes of Ahmed's own certification, so the third-party rule
#: above does not apply and they stay quotable -- but "Certificate ID:
#: UC-fd24728f-6fe7-4349-b78c-00cafb23cac5" is unreadable filler in an answer to
#: "What certifications does Ahmed have?". It is dropped only when the question
#: did not ask for it, which is the bargain the composer makes everywhere else:
#: the corpus decides what is true, the question decides what is relevant.
#:
#: Nothing is removed from the knowledge base. "What is Ahmed's certificate ID?"
#: still answers with the value.
_OPAQUE_ID_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*[-*+]?\s*\*{0,2}("
    r"cert(?:ificate)?\s*id|registration\s*id|enrol?lment\s*id|"
    r"verification\s*(?:url|link|id)|uuid|serial(?:\s*(?:no|number))?"
    r")\*{0,2}:",
    re.IGNORECASE,
)

#: A statement that opens with a pronoun followed by a copula has no subject of its
#: own: "It is the way to reach him for project inquiries" continues the sentence
#: before it, and "This is the only certification on record" continues the
#: sentence before that. Quoted alone, as this composer must, either one refers
#: to a subject the reader cannot see.
#:
#: Pronoun plus copula is what separates these from "It uses only free,
#: open-source and locally hosted tooling", which has a verb of its own and reads
#: fine out of context. The rule is deliberately narrow: a demonstrative with a
#: real predicate ("This project was built...") is left alone, and a recorded
#: absence is exempt in the caller because "That is not recorded" is an answer
#: rather than a continuation.
_ANAPHORIC_COPULAR_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:it|this|that|these|those)\s+(?:is|are|was|were)\b", re.IGNORECASE
)


def _is_meta_section(section: str) -> bool:
    """True when a heading describes handling rules rather than subject matter."""
    return bool(_META_SECTION_RE.search(section))


def _is_field_entry(statement: str) -> bool:
    """True when a statement is a ``label: value`` pair rather than a list entry."""
    return bool(_FIELD_ENTRY_RE.match(statement))


def _is_third_party_field(statement: str) -> bool:
    """True when a field row records an attribute of somebody other than Ahmed."""
    return bool(_THIRD_PARTY_LABEL_RE.match(statement))


def _is_anaphoric(statement: str) -> bool:
    """True when a statement only makes sense after the one before it."""
    return bool(_ANAPHORIC_COPULAR_RE.match(statement))


def _asks_for_the_label(statement: str, concepts: Sequence[str]) -> bool:
    """True when the question names the thing a field row's label identifies.

    Compares the row's label -- the text before its colon -- with the question's
    own concepts, so "What is Ahmed's certificate ID?" keeps the row while
    "What certifications does Ahmed have?" does not.
    """
    if not concepts:
        return False
    label = statement.split(":", 1)[0]
    label_terms = frozenset(content_terms(label))
    if not label_terms:
        return False
    return any(terms_match(concept, label_terms) for concept in concepts)


def _is_unwanted_identifier(statement: str, concepts: Sequence[str]) -> bool:
    """True for an opaque-identifier row the question did not ask about."""
    if not _OPAQUE_ID_LABEL_RE.match(statement):
        return False
    return not _asks_for_the_label(statement, concepts)


#: Share of the shorter statement's distinctive terms that the longer one must
#: also contain for the pair to count as the same fact said twice.
#:
#: Both "Ahmed has exactly one recorded certification, and it is this course" and
#: "This is the only certification on record" were quoted, because ordinary
#: token overlap between them is too low for the existing Jaccard rule: the second
#: says three content words and two of them recur.
REDUNDANT_OVERLAP: float = 0.6
#: Both statements need this many distinctive terms before the rule applies.
#:
#: Without the floor, "Build meaningful projects" is entirely contained in
#: "Projects that are worth building and that move him forward, rather than filler
#: work" -- by the numbers a redundant restatement, but it is one of the recorded
#: goals and losing it would be a loss of evidence, not a gain in tidiness.
REDUNDANT_MIN_TERMS: int = 3


def _answers_a_different_question(unit: EvidenceUnit, concepts: Sequence[str]) -> bool:
    """True when a stored absence answers some *other* stored question.

    A recorded absence is only an answer to the question it was recorded against.
    The FAQ stores the two together -- the bold question in
    :attr:`EvidenceUnit.prompt`, the answer beneath it -- but only the answer is
    ever rendered, so without this check the answer loses its subject and reads
    as a claim about whatever the user happened to ask.

    Concretely, ``**Where is he based?** / That is not recorded in this
    knowledge base.`` was being quoted onto the end of "What is your name?",
    asserting that a name is not recorded in a knowledge base that records it
    three lines earlier. Anchoring on the prompt restores the subject it needs.

    Only *negative* evidence is treated this way. A positive statement is
    informative whatever question it was stored against, so it stays quotable,
    and a negative statement with no stored prompt -- prose such as "Ahmed has no
    formal employment history" -- stays quotable too, because there is nothing to
    contradict.
    """
    if unit.prompt is None or not is_negative_evidence(unit.text):
        return False
    prompt_surface = frozenset(content_terms(unit.prompt))
    if not prompt_surface:
        return False
    return not any(terms_match(concept, prompt_surface) for concept in concepts)


def _is_milestone(statement: str, *, recorded_line: bool) -> bool:
    """True when a statement records a *stage* of a longer record.

    Two shapes, and only two:

    * a ``label: value`` row -- "Institution: Ramaiah Polytechnic", "Expected
      graduation: 2027", "Percentage: 83%" -- which is how a document states an
      attribute of one stage; and
    * a recorded line carrying a number, which is where the recorded *measures*
      of a stage live: the CGPA of 9.11, a year, a percentage.

    ``recorded_line`` is what keeps ordinary prose out. "He is entering the 5th
    semester and is expected to graduate in 2027" mentions two numbers and is
    not a milestone -- it is a sentence *about* a milestone, and quoting it on
    its own is how a journey ends up listing the year and dropping the
    institution the year belongs to. A bullet or a row is a recorded fact in
    isolation; a paragraph is not. So a document that keeps its education in
    prose has no milestones, :meth:`AnswerComposer._compose_journey` declines,
    and the ordinary prose answer is returned exactly as before.

    The number rule is also the reason a recorded absence can appear on the
    timeline: "No school leaving certificates, 10th or 12th grade results, or
    prior qualifications are recorded" is the stage a knowledge base is obliged
    to account for once a journey has been asked about, and it is here only
    because the corpus states it -- nothing is added to make a journey look
    complete.

    Everything else in such a document is context rather than stage. "Relevant
    coursework: Python programming" describes what a stage covered rather than
    when it happened, and quoting it turns a timeline into a transcript.
    """
    if _is_field_entry(statement):
        return True
    return recorded_line and any(character.isdigit() for character in statement)


def _statement_class(candidate: _Candidate) -> int:
    """Rank a statement within a list-shaped answer; lower sorts first."""
    if _is_field_entry(candidate.statement):
        return 2
    return 0 if candidate.is_list_item else 1


def _is_on_topic(
    block: ContextBlock, concepts: Sequence[str], *, allow_absence: bool = True
) -> bool:
    """True when a block discusses one of the question's concepts.

    A block that records an explicit absence counts as on-topic even without a
    matching concept, so that "How long has he worked?" can be answered with
    "Ahmed has no formal employment history" rather than a vague "not available".

    ``allow_absence`` is switched off for list questions. An absence recorded in
    one corner of a document is not a substitute for a missing item somewhere
    else in it: with the escape left on, "What is Ahmed's favourite movie?"
    matched interests.md purely on "He has not listed any competitive gaming,
    esports or publishing activity" and answered with his hobbies.
    """
    surface = frozenset(content_terms(f"{block.section} {block.text}"))
    if any(terms_match(concept, surface) for concept in concepts):
        return True
    if not allow_absence:
        return False
    return any(is_negative_evidence(unit.text) for unit in split_evidence_units(block.text))


@dataclass(frozen=True, slots=True)
class ComposedAnswer:
    """A grounded answer plus the evidence it was assembled from."""

    text: str
    statements: tuple[str, ...]
    citations: tuple[str, ...]
    domain: Domain
    project: str | None = None
    matched_terms: tuple[str, ...] = ()

    @property
    def source_files(self) -> tuple[str, ...]:
        """Knowledge-base files the answer was drawn from."""
        return self.citations


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One scored statement awaiting selection."""

    statement: str
    #: Terms the statement itself contributes. Used for de-duplication, because
    #: two bullets under one heading differ only here -- folding the heading into
    #: this set made "Python" and "FastAPI" look 60% alike and silently dropped
    #: all but the first item of every list.
    terms: frozenset[str]
    #: Statement terms plus the section heading. Used for matching the question,
    #: so a block under "Recorded certifications" can answer a certification
    #: question even when no sentence in it uses the word.
    surface: frozenset[str]
    score: float
    #: Inverse-document-frequency weight of the *rarest* query concept this
    #: statement matches. This is the primary ranking signal.
    peak: float
    citation: str
    is_list_item: bool
    #: Position within its source document, used as the tie-break so equally
    #: relevant statements come out in the order a human would read them.
    document_order: tuple[str, int, int]
    #: Weight of each content term, used to spot two statements that say the same
    #: thing in different words. Terms nobody would use to identify a fact are
    #: ignored, so "only certification on record" overlaps "exactly one recorded
    #: certification" on the words that carry the meaning and not on the filler.
    weights: Mapping[str, float]

    @property
    def key(self) -> str:
        """De-duplication key."""
        return normalise_key(self.statement)


class AnswerComposer:
    """Assemble grounded answers from retrieved evidence, without a model."""

    def __init__(
        self,
        profile: CorpusProfile,
        *,
        idf: Callable[[str], float] | None = None,
        vocabulary: frozenset[str] | None = None,
        scoped_vocabulary: Callable[[tuple[str, ...]], frozenset[str]] | None = None,
        max_statements: int = MAX_STATEMENTS,
        max_list_statements: int = MAX_LIST_STATEMENTS,
    ) -> None:
        """
        Args:
            profile: Which documents back which domain, and the project names.
            idf: Optional term-weight function. Supplying the BM25 index's
                ``idf`` makes ranking favour rare, distinctive query terms over
                near-universal ones like the subject's own name.
            vocabulary: Every term the corpus contains. Supplying the BM25
                index's ``vocabulary`` lets the composer tell "asked using
                different words" apart from "asked about something that is simply
                not recorded" -- see :meth:`_attested`.
            scoped_vocabulary: Optional ``source_files -> terms`` lookup, the
                BM25 index's ``vocabulary_for``. It answers the same question as
                ``vocabulary`` but only about the documents routing selected,
                which is the question that actually governs whether a word can
                reach the evidence -- see :meth:`has_lexical_support`.
            max_statements: Statement budget for a focused answer.
            max_list_statements: Statement budget when a list is expected.
        """
        self._profile = profile
        self._idf: Callable[[str], float] = idf if idf is not None else _unit_idf
        # :data:`MIN_USEFUL_IDF` is a statement about inverse document frequency,
        # so it only means anything when there is an IDF to compare against.
        # Without a lexical prior every term weighs the same and any match is as
        # informative as any other.
        self._min_useful_idf = MIN_USEFUL_IDF if idf is not None else 0.0
        self._vocabulary = vocabulary
        self._scoped_vocabulary = scoped_vocabulary
        self._max_statements = max_statements
        self._max_list_statements = max_list_statements

    @property
    def profile(self) -> CorpusProfile:
        """The corpus map backing routing."""
        return self._profile

    def compose(self, query: str, blocks: Sequence[ContextBlock]) -> ComposedAnswer | None:
        """Answer ``query`` from ``blocks``, or return ``None``.

        Args:
            query: The user's question.
            blocks: Security-processed context blocks, best first.

        Returns:
            A :class:`ComposedAnswer`, or ``None`` when the question routes to no
            domain or has no supporting evidence. Callers must turn ``None`` into
            the standard no-information response rather than inventing an answer.
        """
        project = detect_project(query, self._profile.project_names)
        routing = self._route(query, project)

        if routing.intent is Intent.UNKNOWN:
            logger.debug(
                "no intent matched the question (trace=%s); declining to answer",
                ", ".join(routing.trace) or "no signal",
            )
            return None

        # A question that names a project the corpus does not contain has no
        # evidence behind it, and there is nothing legitimate to answer it with.
        # Answering from a *different* project is the worst option available: the
        # citations would check out and the answer would still be about something
        # the user did not ask for.
        #
        # The check runs over *every* question, not just ones that routed to the
        # projects domain. "Tell me about a project called FakeProject." also
        # routes to identity -- "called" is identity vocabulary -- and answering
        # that from about.md is the same substitution failure one domain over.
        if project is None:
            unknown = detect_unrecognised_project(query, self._profile.project_names)
            if unknown is not None:
                logger.info(
                    "declining to answer: %r names a project the corpus does not contain",
                    unknown,
                )
                return None

        allowed = self._allowed_files(routing, project)
        if not allowed:
            logger.debug("intent %s has no documents; declining to answer", routing.intent.value)
            return None

        eligible = self._eligible(blocks, allowed, routing)
        if not eligible:
            logger.debug("no retrieved block came from the routed documents")
            return None

        query_terms = content_terms(query)
        # The terms used to *match statements* may differ from the terms used to
        # check the question is answerable at all. A project's name fragments are
        # dropped from matching -- left in, "forge" matches only the word inside
        # "resume-forge-one-zeta.vercel.app" and the live URL outranked the
        # summary as the answer to "What is resume forge?" -- but they are kept
        # for the attestation check, where the name is exactly the evidence that
        # the project is in the corpus.
        match_terms = query_terms
        if project is not None:
            match_terms = [
                term for term in query_terms if not is_part_of_project_name(term, project)
            ]

        # Nothing in the corpus says anything about this question, so there is no
        # evidence to ground an answer in. Declining here is what turns "Who is
        # Ahmed's father?" into an honest no-information reply instead of a
        # confident quote about his diploma and his 5th semester.
        concepts = [term for term in query_terms if term not in _SUBJECT_ALIASES]
        if concepts and not self._attested(concepts, routing.vocabulary):
            logger.info(
                "declining to answer: %s occurs nowhere in the corpus and did not "
                "route the question either",
                ", ".join(concepts),
            )
            return None

        # The routed documents must actually be *about* what was asked. This check
        # has to live here rather than inside one of the composition paths,
        # because a path that declines hands over to the next one: gating only the
        # lead path let the targeted path answer out of the very blocks the gate
        # had just rejected.
        listing = routing.intent in WHOLE_DOCUMENT_INTENTS and bool(query_words(query) & _LIST_CUES)
        eligible = self._on_topic(
            eligible,
            query_terms,
            routing.terms,
            framing=overview_framing_terms(query, routing.intent),
            allow_absence=not listing,
            primary_files=frozenset(self._profile.primary_files_for(routing.primary)),
            recognised=routing.vocabulary,
            spanned_files=frozenset(
                source
                for domain in routing.spanned
                for source in self._profile.primary_files_for(domain)
            ),
        )
        if not eligible:
            logger.info(
                "declining to quote %s: the question asks about something that "
                "document does not discuss",
                routing.intent.value,
            )
            return None

        # A question whose answer *is* a document -- a list-shaped question, or an
        # overview request -- is answered by that document's own content rather
        # than by whichever sentence happens to contain the question's noun.
        # "What are his skills?" wants skills.md, not the sentence with "skills"
        # in it.
        #
        # A question that legitimately spans two domains is assembled from both,
        # for the same reason. "Any chance you could walk me through his
        # academic history?" routes to education with experience as a co-answer,
        # and "history" occurs only in experience.md -- so the targeted path
        # found a distinctive match there and answered a question about his
        # diploma with his employment record. Taking one contribution per routed
        # document is the whole point of the spanning branch.
        project_inventory = project is None and routing.intent is Intent.PROJECTS
        if (
            listing
            or project_inventory
            or is_overview_intent(routing.intent)
            or is_overview_query(query, routing.intent)
            or routing.spanned
        ):
            answer = self._compose_lead(
                query,
                query_terms,
                eligible,
                routing,
                project,
                spanning=listing,
                match_terms=match_terms,
            )
            if answer is not None:
                return answer

        return self._compose_targeted(
            query, query_terms, eligible, routing, project, match_terms=match_terms
        )

    # -- routing -----------------------------------------------------------

    def _route(self, query: str, project: str | None) -> Routing:
        """Route ``query``, forcing the projects domain when a project is named.

        Naming a project is decisive: "What technologies does ResumeForge use?"
        is about ResumeForge even though "technologies" is skills vocabulary, and
        the project's own document is the only one allowed to answer it. The
        expansion is dropped rather than kept, because a project's document
        already holds its own stack, and admitting skills.md would let the
        document's list of Ahmed's general skills answer a question about what
        that project uses.
        """
        routing = route(query)
        if project is None:
            return routing
        return Routing(
            intent=Intent.PROJECT_DETAIL,
            domains=(Domain.PROJECTS,),
            terms=routing.terms,
            trace=(*routing.trace, "project:" + project),
            vocabulary=routing.vocabulary,
            spanned=(),
        )

    def allowed_files_for(self, query: str) -> tuple[str, ...]:
        """Documents permitted to back an answer to ``query``.

        Exposed so the retrieval layer can fall back to a document-scoped search
        when a query carries no terms for either retriever to key on. Routing is
        the only relevance signal such a query has, so the caller needs to know
        which documents that signal points at.

        Returns:
            The routed documents, or ``()`` when the question names no project
            and routes to no domain.
        """
        project = detect_project(query, self._profile.project_names)
        routing = self._route(query, project)
        if routing.intent is Intent.UNKNOWN:
            return ()
        return self._allowed_files(routing, project)

    def document_scope(self, query: str) -> tuple[str, ...] | None:
        """Documents retrieval should confine itself to, or ``None``.

        For a question whose answer *is* a document -- a list-shaped question
        about a list-shaped domain, or an overview request -- retrieval should be
        confined to the routed documents. Two things follow, and both are gains.

        The composer only ever quotes documents returned by
        :meth:`allowed_files_for`, so nothing outside the scope was ever going to
        be quotable: confining retrieval can therefore only remove context that
        was already destined for the bin. And because sibling expansion is
        similarity-free, a single admitted chunk is enough to pull in the rest of
        its document. Without the scope, "What does Ahmed do for fun?" matched
        only interests.md's introductory chunk, expansion spent its budget on
        higher-ranked documents, and the answer stopped after two sentences
        instead of naming the anime, the gaming and the comics.

        Returns ``None`` when the question is a targeted lookup, so ordinary
        lookups keep unrestricted retrieval.
        """
        project = detect_project(query, self._profile.project_names)
        routing = self._route(query, project)
        if routing.intent is Intent.UNKNOWN:
            return None
        whole_document = routing.intent in WHOLE_DOCUMENT_INTENTS and bool(
            query_words(query) & _LIST_CUES
        )
        if not (whole_document or is_overview_intent(routing.intent) or is_overview_query(query, routing.intent)):
            return None
        allowed = self._allowed_files(routing, project)
        return allowed or None

    def _attested(self, concepts: Sequence[str], routed: frozenset[str] = frozenset()) -> bool:
        """True when the corpus can say something about this question.

        Two independent escapes from "not answered":

        * at least one concept occurs in the corpus, or
        * routing accounts for every concept.

        The first is the difference between "asked in words the corpus does not
        use" and "asked about something that was never recorded". A concept
        absent from the vocabulary can never be answered by any document, so when
        *nothing* is attested the honest reply is that there is no information --
        not a confident quote from the nearest document that shares a pronoun.
        That is what turns "Who is Ahmed's father?" into a no-information reply
        instead of his diploma and his 5th semester.

        The second escape exists because routing is the system's own name for the
        subject matter, and the documents are not obliged to repeat it: "fun" is
        how a question asks about interests, and interests.md does not contain
        the string "fun". :meth:`_concept_accounted_for` states that rule precisely.

        The caller passes :attr:`Routing.vocabulary` here rather than
        :attr:`Routing.terms`, because this gate asks only "could any document
        answer this?" -- and a word the router recognises is evidence about the
        question even when it belonged to an intent that lost. "What's his
        academic background?" routes to education on the strength of
        "academic"; "background" also matched, just less strongly. Requiring the
        corpus to contain "background" anyway is demanding a coincidence, and
        the question was refused on the real knowledge base only because two
        unrelated documents happen to use the word for Celery workers.

        :meth:`_on_topic` deliberately keeps the stricter set. Admitting a
        *document* on the grounds that a word is familiar would defeat the point
        of the check, which is whether that particular document discusses the
        subject asked about.

        The bar is deliberately *any* concept attested rather than all of them. A
        question carries words the corpus would never use -- "Which platform
        issued Ahmed's certifications?" contributes "issued", which matches no
        corpus token because "issuing" is not a bounded-prefix match of it.
        Demanding every concept be attested refused a question the knowledge base
        answers.

        Morphological variants count throughout: "technologies" is attested by
        "technology", and "certifications" by "certification", via
        :func:`terms_match`.

        Returns ``True`` when no vocabulary was supplied, so a composer built
        without a lexical prior keeps its previous, more permissive behaviour.
        """
        if self._vocabulary is None:
            return True
        if any(terms_match(concept, self._vocabulary) for concept in concepts):
            return True
        return (
            bool(routed)
            and self._has_positive_accounting(concepts, routed)
            and any(self._concept_accounted_for(concept, routed) for concept in concepts)
        )

    def _has_positive_accounting(self, concepts: Sequence[str], routed: frozenset[str]) -> bool:
        """True when at least one concept is what the question is *about*.

        The complement of :meth:`_concept_accounted_for`. Every concept must be
        explained by routing, but something has to be explained for the question to
        be *about* anything.

        "Who is Ahmed's father?" is the case that needs this. "father" is routed by
        nothing and attested by nothing, so the per-concept test alone cannot
        distinguish it from a question whose whole vocabulary the corpus lacks --
        except that "who" matched the identity concept, which was enough for the
        identity fallback to fire. The question was then answered with his full
        name. Requiring one concept to be genuinely accounted for -- routed, or
        present in the corpus vocabulary -- is what stops that.
        """
        if self._vocabulary is None:
            return True
        return any(
            terms_match(concept, routed) or terms_match(concept, self._vocabulary)
            for concept in concepts
        )

    def answered_documents_retrieved(self, query: str, source_files: Iterable[str]) -> bool:
        """True when retrieval returned something from the document that answers.

        The other half of :meth:`has_lexical_support`. That method asks whether
        the retrievers had a word to key on; this one asks whether their results
        landed where the answer is. Both can be false at once and the question is
        then unreachable.

        "Any chance you could walk me through his academic history?" routes to
        education and retrieves nine chunks -- from the FAQ and from
        experience.md -- and none from education.md. Retrieval had clearly not
        failed, so nothing looked wrong, and the reply was declined; but lexical
        search could never have reached education.md either, because it does not
        contain the word "academic" or "history". Those two conditions together
        are what this reports.

        True is returned when the question routes to no domain or names no
        document, since there is then nothing specific to have missed.
        """
        expected = self._contributing_files(query)
        if not expected:
            return True
        return bool(set(source_files) & set(expected))

    def has_lexical_support(self, query: str) -> bool:
        """True when the retriever has at least one word of ``query`` to key on.

        The lexical retriever can only match corpus vocabulary, and the dense
        retriever needs an embedding close enough to clear the threshold. A
        question whose every content word is absent from the corpus leaves neither
        with anything to work on -- "What's Ahmed's background?" contributes the
        single word "background", which no document contains, and it retrieved
        nothing at all despite the knowledge base answering it in three places.

        When this is ``False``, routing is the only relevance signal available,
        and the caller is entitled to search within the routed documents instead
        of giving up. When no vocabulary was supplied at all the question is
        treated as supported, which keeps a composer built without a lexical prior
        behaving exactly as it did before.

        The test is made against the *routed documents* when a scoped vocabulary
        is available, because that is the evidence this question could possibly
        be answered from. Whole-corpus membership answers a different question --
        "does these words occur anywhere" -- and the two diverge exactly where
        this method matters. "background" occurs twice in the corpus, both times
        as a Celery/ARQ *worker*, in documents a background question does not
        route to. Judged globally the word looked supported, so the caller never
        retried, and the question was declined even though identity.md,
        education.md and goals.md answer it between them. Nothing about "worker"
        is relevant to "background"; only the location of the evidence decides.

        Scoping is what makes this general rather than another special case: no
        question string is inspected, and a word the routed documents never use
        is by construction a word retrieval cannot reach there. It is scoped to
        the *contributing* documents -- see :meth:`_contributing_files` -- because
        a word that appears only in a document that will not answer the question
        is not a lead, it is a coincidence.
        """
        if self._vocabulary is None:
            return True
        concepts = [term for term in content_terms(query) if term not in _SUBJECT_ALIASES]
        if not concepts:
            return False
        vocabulary = self._vocabulary
        if self._scoped_vocabulary is not None:
            contributing = self._contributing_files(query)
            if contributing:
                scoped = self._scoped_vocabulary(contributing)
                # An empty scope means those documents contributed no tokens at
                # all, which is a statement about the index rather than about
                # the question. Fall back to the whole corpus so a partially
                # indexed corpus cannot turn every question into a retry.
                if scoped:
                    vocabulary = scoped
        return any(terms_match(concept, vocabulary) for concept in concepts)

    def _contributing_files(self, query: str) -> tuple[str, ...]:
        """The document this question is chiefly answered by.

        Only the *primary* routed domain's own documents, and never the
        cross-cutting FAQ. The FAQ restates every domain, so almost any question
        finds one of its words there, and counting it would defeat the test
        entirely.

        Primary-only, rather than every contributing domain, because a secondary
        domain's incidental word match is not a lead. "Any chance you could walk
        me through his academic history?" routes to education with *experience*
        as a co-answer; "history" occurs in ``experience.md`` and nowhere in
        ``education.md``. Lexical retrieval therefore looked confident, never
        reached the document that would actually answer the question, and the
        reply was declined. A word is a lead only where the answer lives.
        """
        project = detect_project(query, self._profile.project_names)
        routing = self._route(query, project)
        if routing.intent is Intent.UNKNOWN:
            return ()
        if project is not None:
            source = self._profile.project_files.get(project)
            return (source,) if source else ()
        return self._profile.primary_files_for(routing.primary)

    def _is_cross_cutting_entry_for(self, block: ContextBlock, domain: Domain) -> bool:
        """True when ``block`` is the cross-cutting FAQ's entry for ``domain``."""
        if self._profile.category(block.source_file) != CROSS_CUTTING_CATEGORY:
            return False
        return detect_domain(block.section) is domain

    def _concept_accounted_for(self, concept: str, routed: frozenset[str]) -> bool:
        """True when a question's own word is explained by the question.

        A concept is accounted for when routing named it. "fun" is how a question
        asks about interests, and requiring interests.md to contain the string
        "fun" is requiring a coincidence; the corpus is not obliged to use the
        word a user chose.

        That is the whole rule, and it is deliberately strict. An *earlier*,
        looser version also counted any word the corpus had never used, on the
        reasoning that no document could be expected to discuss it. That reasoning
        is wrong for anything in subject position: "What is Ahmed's favorite car?"
        is a real question about a real noun, and counting "car" as unaccounted-for
        because no document happens to mention cars is exactly how it came to be
        answered with his hobbies.

        Words that are grammatical rather than referential -- "someone" in "Can
        someone hire Ahmed?" -- are handled where that is what they are, in
        :data:`app.services.text.WEAK_TERMS`, so they never reach this check.
        """
        return terms_match(concept, routed)

    def _allowed_files(self, routing: Routing, project: str | None) -> tuple[str, ...]:
        """Documents permitted to back an answer.

        For a named project that is the project's own document and nothing else.
        Otherwise it is every routed domain's documents, primary first, which is
        what makes a composite question answerable from more than one document
        while a single-domain question is unaffected.
        """
        if project is not None:
            source = self._profile.project_files.get(project)
            return (source,) if source else ()
        return self._profile.files_for_domains(routing.domains)

    def _eligible(
        self,
        blocks: Sequence[ContextBlock],
        allowed: tuple[str, ...],
        routing: Routing,
    ) -> list[ContextBlock]:
        """Blocks that may back an answer, minus anything the detector flagged.

        Two filters apply:

        * the document must belong to the routed domain, and
        * a cross-cutting FAQ block must *also* have a heading that belongs to
          the routed domain.

        The second filter is what stops "What is Ahmed's favourite film?" from
        being answered out of the FAQ's Projects section, which happens to
        contain the word "movie". Without it the cross-cutting document leaks
        unrelated sections into every domain.

        A ``flagged`` block is never quoted. Extraction already makes obeying it
        impossible, but an attacker's payload must not reach the output at all.
        """
        permitted = set(allowed)
        routed_domains = set(routing.domains)
        eligible: list[ContextBlock] = []
        for block in blocks:
            if block.source_file not in permitted or block.hostile:
                continue
            if self._profile.category(block.source_file) == CROSS_CUTTING_CATEGORY:
                heading_domain = detect_domain(block.section)
                if heading_domain not in routed_domains | {Domain.UNKNOWN}:
                    logger.debug(
                        "skipping %s :: %s (belongs to %s, not %s)",
                        block.source_file,
                        block.section,
                        heading_domain.value,
                        routing.intent.value,
                    )
                    continue
            eligible.append(block)
        return eligible

    # -- selection ---------------------------------------------------------

    def _collect(
        self,
        blocks: Sequence[ContextBlock],
        query_terms: Sequence[str],
    ) -> list[_Candidate]:
        """Score every answerable statement in ``blocks``, in document order."""
        concepts = [term for term in query_terms if term not in _SUBJECT_ALIASES]
        candidates: list[_Candidate] = []
        ordered = sorted(blocks, key=lambda block: (block.source_file, block.ordinal))
        for block in ordered:
            # The section heading is part of the matching surface: a block under
            # "Recorded certifications" is evidence for a certification question
            # even when none of its sentences contain the word.
            heading = block.section
            if _is_meta_section(heading):
                # Guidance to the reader, not subject matter. See
                # :data:`_META_SECTION_RE`.
                continue
            for index, unit in enumerate(split_evidence_units(block.text)):
                if not is_answerable_evidence(unit.text):
                    continue
                if _answers_a_different_question(unit, concepts):
                    continue
                if _is_anaphoric(unit.text) and not is_negative_evidence(unit.text):
                    # A recorded absence is exempt. "That is not recorded in this
                    # knowledge base" looks like a continuation of the question
                    # above it, but it is the answer to that question, and dropping
                    # it would turn a question with a recorded answer into a
                    # refusal.
                    logger.debug("dropped continuation statement: %r", unit.text[:60])
                    continue
                if _is_third_party_field(unit.text):
                    logger.debug("dropped third-party metadata: %r", unit.text[:60])
                    continue
                if _is_unwanted_identifier(unit.text, concepts):
                    logger.debug("dropped unrequested identifier row: %r", unit.text[:60])
                    continue
                statement = unit.text.strip()
                if len(statement) < MIN_STATEMENT_CHARS and not _is_standalone(unit, statement):
                    continue
                terms = frozenset(content_terms(unit.match_text))
                if not terms:
                    continue
                surface = frozenset(content_terms(f"{heading} {unit.match_text}"))
                # Weight each matched query concept by the IDF of the token that
                # actually matched, not of the query's own spelling. A question
                # asking about "technologies" must be able to score against a
                # document that says "technology", and the query spelling may not
                # exist in the corpus at all -- in which case its IDF would be
                # zero and the real match would be silently discarded.
                score = 0.0
                peak = 0.0
                for token in surface:
                    if not any(terms_match(term, {token}) for term in query_terms):
                        continue
                    weight = self._idf(token)
                    score += weight
                    peak = max(peak, weight)
                candidates.append(
                    _Candidate(
                        statement=_clip(statement),
                        terms=terms,
                        surface=surface,
                        score=score,
                        peak=peak,
                        citation=f"{block.source_file} :: {block.section}",
                        is_list_item=unit.is_list_item,
                        document_order=(block.source_file, block.ordinal, index),
                        weights={
                            term: self._idf(term)
                            for term in terms
                            if self._idf(term) >= self._min_useful_idf
                        },
                    )
                )
        return candidates

    def _compose_targeted(
        self,
        query: str,
        query_terms: Sequence[str],
        eligible: Sequence[ContextBlock],
        routing: Routing,
        project: str | None,
        *,
        match_terms: Sequence[str] | None = None,
    ) -> ComposedAnswer | None:
        """Quote the statements that actually match the question."""
        collected = self._collect(eligible, query_terms if match_terms is None else match_terms)
        # Only a distinctive match counts. Matching the subject's own name or a
        # ubiquitous heading word is not evidence that this statement answers the
        # question.
        matched = [item for item in collected if item.peak >= self._min_useful_idf]
        if not matched:
            # Routed domain, but nothing in it says anything distinctive about the
            # question. The answer is the domain's own content, so quote its best
            # document -- subject to the check below.
            return self._compose_lead(
                query, query_terms, eligible, routing, project, match_terms=match_terms
            )

        best = max(item.peak for item in matched)
        cutoff = best * RELATIVE_SCORE_FLOOR
        selected = sorted(
            (item for item in matched if item.peak >= cutoff),
            # Rank by the rarest concept matched, then by total weight, then in
            # document order. Ranking on peak is what stops a project name --
            # which matches nearly every sentence in its own document, and is
            # therefore uninformative -- from deciding the answer.
            #
            # A recorded absence is ranked below any recorded fact. "The CGPA
            # scale is not stated." and "The recorded CGPA is 9.11." match
            # "cgpa" equally well, and document order put the absence first, so
            # "What is Ahmed's CGPA?" was answered with the knowledge boundary
            # and the number was left out of the reply entirely. An absence is
            # the right answer when there is nothing else -- "What is his
            # nickname?" has no competing fact -- and demoting rather than
            # excluding preserves that.
            key=lambda item: (
                is_negative_evidence(item.statement),
                -item.peak,
                -item.score,
                item.document_order,
            ),
        )
        selected = self._dedupe(selected)[: self._limit_for(query, routing)]
        if not selected:
            return None
        return self._render(selected, query_terms, routing, project, mode="matched")

    def _compose_lead(
        self,
        query: str,
        query_terms: Sequence[str],
        eligible: Sequence[ContextBlock],
        routing: Routing,
        project: str | None,
        *,
        spanning: bool = False,
        match_terms: Sequence[str] | None = None,
    ) -> ComposedAnswer | None:
        """Quote the opening statements of the routed document, in document order.

        Used for list-shaped questions ("what are his skills?"), where the answer
        is the document's own content rather than a matching sentence, and for
        overview requests ("summarise ResumeForge").

        Ordering is by the chunk's position *within its document*, not by
        retrieval rank. "Summarise ResumeForge" must quote the Summary section
        that a human would reach first, and retrieval rank is decided by embedding
        similarity, which has no opinion about what a document's summary is.

        Args:
            spanning: When set, keep drawing from later chunks of the same
                document until the statement budget is used. This is what lets
                "what are his skills?" return both the stated skills and the
                project-evidenced ones.
        """
        # A project-inventory question has no single project document to lead.
        # The FAQ contains the authoritative inventory of all recorded projects
        # in one statement, so prefer that entry instead of returning only the
        # alphabetically first project document.
        if project is None and routing.intent is Intent.PROJECTS and not routing.spanned:
            faq_blocks = [
                block
                for block in eligible
                if self._is_cross_cutting_entry_for(block, Domain.PROJECTS)
            ]
            if faq_blocks:
                harvested = self._harvest(
                    sorted(faq_blocks, key=lambda block: (block.source_file, block.ordinal)),
                    query_terms if match_terms is None else match_terms,
                    self._max_statements,
                )
                if harvested:
                    return self._render(
                        harvested[: self._max_statements],
                        query_terms,
                        routing,
                        project,
                        mode="lead",
                    )

        # An overview of a *record* -- an academic history, a work history -- is
        # answered as the timeline that record actually is, not as whichever
        # individual sentence happens to contain a word from the question.
        if self._wants_a_journey(query, routing):
            journey = self._compose_journey(
                query, query_terms, eligible, routing, project, match_terms=match_terms
            )
            if journey is not None:
                return journey

        # History questions have an explicit, question-shaped FAQ entry. Prefer
        # it over education.md's field rows so the user gets a coherent historical
        # summary rather than a database-style sequence of labels. The FAQ entry
        # is still quoted verbatim and remains subject to the same security filter.
        if is_overview_query(query, routing.intent) and "history" in query_words(query):
            faq_domain = routing.primary
            faq_blocks = [
                block
                for block in eligible
                if self._is_cross_cutting_entry_for(block, faq_domain)
            ]
            if faq_blocks:
                harvested = self._harvest(
                    sorted(faq_blocks, key=lambda block: (block.source_file, block.ordinal)),
                    query_terms if match_terms is None else match_terms,
                    self._max_statements,
                )
                if harvested:
                    return self._render(
                        harvested[: self._max_statements],
                        query_terms,
                        routing,
                        project,
                        mode="lead",
                    )

        # Prefer each routed domain's own document over the cross-cutting FAQ.
        primary: set[str] = set()
        # Only documents routing actually *spans* are drawn from when the answer
        # is assembled rather than looked up. A simple intent's expansion is
        # supporting context, not part of the answer: letting it contribute here
        # appended "Mohammed Ayaan Ahmed, commonly known as Ahmed" to the end of
        # "What is Ahmed focused on?", which goals.md had already answered in
        # full. Padding, and padding is what this phase exists to remove.
        contributors = (routing.domains if routing.spanned else ()) or (routing.primary,)
        for domain in contributors:
            primary.update(self._profile.primary_files_for(domain))
        if project is not None:
            source = self._profile.project_files.get(project)
            primary = {source} if source else primary
        # Fall back to every eligible block when none came from a primary
        # document, which happens when retrieval missed the routed file.
        chosen_blocks = [block for block in eligible if block.source_file in primary] or list(
            eligible
        )

        # Quoting a whole document is only honest if the document is about the
        # thing that was asked. :meth:`_on_topic` has already enforced that, so
        # this path can assume the routed documents are relevant.
        budget = self._limit_for(query, routing)
        terms = query_terms if match_terms is None else match_terms
        chosen: list[_Candidate]

        if spanning:
            # One document, all of it: "what are his skills?" wants skills.md's own
            # inventory, not one sentence from it.
            ordered = sorted(chosen_blocks, key=lambda block: (block.source_file, block.ordinal))
            collected = self._harvest(ordered, terms, budget * 6)
            # ``spanning`` is only ever set for a whole-document question, so this
            # sort always applies here. For a list-shaped question the items are
            # the answer and the prose around them is scaffolding: without the
            # sort, "what are his skills?" answers with "Skills are grouped by how
            # strongly they are evidenced" instead of "Python".
            #
            # Three classes, best first: bare list entries, then prose, then
            # ``label: value`` attributes. The third class is separated out
            # because skills.md lists bare technologies while certifications.md
            # lists attributes of one course; treating both as "bullets" made
            # "what certifications does he have?" answer with "Duration: 10.5
            # total hours" instead of naming the course.
            collected.sort(key=lambda item: (_statement_class(item), item.document_order))
            chosen = collected[:budget]
        else:
            # One contribution per routed document, in routed order. This is the
            # multi-domain path: "What's Ahmed's background?" takes from about.md,
            # then education.md, then goals.md, and stops. Each document
            # contributes only what it itself records -- nothing is merged, and
            # nothing is invented to bridge them.
            #
            # The *primary* domain contributes its opening, because that is what
            # "tell me about" asks for. A secondary domain contributes its
            # best-matching section instead, because it was admitted for a
            # specific reason and its opening is usually about something else:
            # about.md opens with his name, but "What is Ahmed focused on?" wants
            # its "Current focus areas" list, which is fifteen lines further down.
            chosen = []
            per_document = _per_document_budget(budget, len(contributors))
            for position, domain in enumerate(contributors):
                owned = [
                    block
                    for block in chosen_blocks
                    if block.source_file in set(self._profile.primary_files_for(domain))
                ]
                # The cross-cutting FAQ's entry for this domain is appended after
                # the domain's own blocks. education.md records its diploma as
                # ``label: value`` rows -- "**Institution:** Ramaiah
                # Polytechnic" -- which are not quotable prose, while the FAQ
                # restates the same fact as a sentence. Reading only the first
                # block therefore found nothing to say about an academic history
                # and cited the document without using it.
                owned.extend(
                    block
                    for block in eligible
                    if block.source_file not in primary
                    and self._is_cross_cutting_entry_for(block, domain)
                )
                if not owned:
                    continue
                owned.sort(
                    key=lambda block: (
                        (block.source_file, block.ordinal)
                        if position == 0
                        else _section_relevance(block, terms)
                    )
                )
                # Take the first block that yields anything, rather than the
                # first block outright: a document may open with a row that has
                # nothing quotable in it.
                lead: list[_Candidate] = []
                for block in owned[:LEAD_BLOCK_SEARCH]:
                    lead = self._harvest([block], terms, per_document * 6, chosen)
                    if lead:
                        break
                chosen.extend(lead[:per_document])
                if len(chosen) >= budget:
                    break
            if not chosen:
                chosen = self._harvest(
                    sorted(chosen_blocks, key=lambda b: (b.source_file, b.ordinal))[:1],
                    terms,
                    budget,
                )

        if not chosen:
            return None
        return self._render(chosen, query_terms, routing, project, mode="lead")

    @staticmethod
    def _wants_a_journey(query: str, routing: Routing) -> bool:
        """True when the question asks to be walked through a whole record.

        "Walk me through his academic history" and "Summarise his education" are
        the same request: not one fact from the education document, but the
        document's own record of it, in the order the document keeps it.
        """
        if not is_overview_query(query, routing.intent):
            return False
        # ``history`` names the record wherever it appears. Education is added on
        # its own because "his education journey" says the same thing without the
        # word, and because every other education question -- "Where does Ahmed
        # study?", "What is Ahmed's CGPA?" -- is a lookup that never gets here.
        return "history" in query_words(query) or routing.intent is Intent.EDUCATION

    def _compose_journey(
        self,
        query: str,
        query_terms: Sequence[str],
        eligible: Sequence[ContextBlock],
        routing: Routing,
        project: str | None,
        *,
        match_terms: Sequence[str] | None = None,
    ) -> ComposedAnswer | None:
        """Answer an overview request with the milestones of the routed record.

        "Any chance you could walk me through his academic history?" was answered
        with the FAQ's individual question-and-answer fragments -- "Ramaiah
        Polytechnic, for a Diploma in Computer Science. 2027 is the expected
        graduation year. Relevant coursework includes ..." -- which is a
        transcript of the FAQ, not a history. Three things were lost: the order
        the stages happen in, the recorded *scores* (the CGPA of 9.11), and the
        stages the document explicitly says are not recorded.

        The milestones come from the routed domain's own document, which is the
        one written as a record: ``label: value`` rows in document order, plus the
        recorded lines carrying a number, which is where years, percentages and
        grades live. See :func:`_is_milestone` for why a paragraph is not a
        milestone however many figures it quotes. Ordering is the document's own --
        a knowledge base that keeps a diploma above its "Not on record" section is
        describing the stages in the order they happened, and re-sorting them by a
        date extracted from the text would be a guess about the record rather than
        a reading of it.

        Everything quoted is a verbatim statement, so nothing here can invent a
        stage that is not recorded. When the routed document holds no milestones
        the journey is ``None`` and the caller falls through to the ordinary
        paths, which is what keeps a prose-only education document -- or a work
        history, which this knowledge base records as an absence rather than a
        timeline -- behaving exactly as it did before.
        """
        primary = frozenset(self._profile.primary_files_for(routing.primary))
        record = [block for block in eligible if block.source_file in primary]
        if not record:
            return None
        terms = query_terms if match_terms is None else match_terms
        milestones = [
            item
            for item in sorted(self._collect(record, terms), key=lambda i: i.document_order)
            if _is_milestone(item.statement, recorded_line=item.is_list_item)
        ]
        if not milestones:
            return None
        chosen: list[_Candidate] = []
        for item in milestones:
            if any(
                _jaccard(item.terms, kept.terms) >= DUPLICATE_JACCARD or _restates(item, kept)
                for kept in chosen
            ):
                continue
            chosen.append(item)
            if len(chosen) >= JOURNEY_MILESTONE_BUDGET:
                break
        if not chosen:
            return None
        return self._render(chosen, query_terms, routing, project, mode="journey")

    def _harvest(
        self,
        blocks: Sequence[ContextBlock],
        terms: Sequence[str],
        budget: int,
        already: Sequence[_Candidate] = (),
    ) -> list[_Candidate]:
        """De-duplicated candidates from ``blocks``, in document order.

        ``already`` seeds the duplicate check but is not returned. In the
        multi-domain path each document is harvested in turn against everything
        gathered so far, which is what stops the same sentence appearing twice
        when the profile and the FAQ both state it -- "He is a Diploma in Computer
        Science student" appears in both, and quoting it twice is the repetition
        the brief asks to remove.

        The budget is deliberately generous. Stopping at the final statement budget
        while collecting would spend it on whichever kind of statement came first
        in the document, which for a list section is the introductory prose rather
        than the items.
        """
        collected: list[_Candidate] = list(already)
        fresh: list[_Candidate] = []
        for block in blocks:
            for item in self._collect([block], terms):
                if any(
                    _jaccard(item.terms, kept.terms) >= DUPLICATE_JACCARD or _restates(item, kept)
                    for kept in collected
                ):
                    continue
                collected.append(item)
                fresh.append(item)
                if len(fresh) >= budget:
                    return fresh
        return fresh

    @staticmethod
    def _dedupe(candidates: Sequence[_Candidate]) -> list[_Candidate]:
        """Drop exact, near-duplicate and restated statements, keeping the first.

        Three passes, cheapest first: an exact key match, a Jaccard overlap on raw
        tokens, and finally an IDF-weighted overlap that can see through the
        different wording two sources use for one fact.
        """
        kept = _dedupe_by(candidates, lambda a, b: _jaccard(a.terms, b.terms) >= DUPLICATE_JACCARD)
        return _dedupe_by(kept, _restates)

    def _limit_for(self, query: str, routing: Routing) -> int:
        """Statement budget for this question shape.

        A composite question is held to the *focused* budget even though it spans
        several documents. Answering "What's Ahmed's background?" from three
        documents is legitimate; answering it with eighteen sentences is not.
        """
        if routing.intent in WHOLE_DOCUMENT_INTENTS and query_words(query) & _LIST_CUES:
            return self._max_list_statements
        return self._max_statements

    # -- what the corpus can and cannot vouch for ---------------------------

    def _on_topic(
        self,
        blocks: Sequence[ContextBlock],
        query_terms: Sequence[str],
        routed: frozenset[str] = frozenset(),
        *,
        framing: frozenset[str] = frozenset(),
        allow_absence: bool = True,
        primary_files: frozenset[str] = frozenset(),
        recognised: frozenset[str] = frozenset(),
        spanned_files: frozenset[str] = frozenset(),
    ) -> list[ContextBlock]:
        """Keep only the documents that discuss what the question actually asked.

        The subject's own name is excluded from the comparison. Nearly every
        chunk in the knowledge base mentions Ahmed, so a name match says nothing
        about a document's subject matter -- and it is precisely why
        ``interests.md`` was retrieved for "What is Ahmed's favourite movie?" in
        the first place. ``interests.md`` discusses anime, manga, gaming and
        comic books, so it must not be quoted as an answer about a film.

        The test is applied per *document*, not per block, and that granularity
        is the whole point. A document is a coherent unit of subject matter, so
        once a document is admitted its chunks are the evidence for it. Judging
        each block separately is what broke list-shaped answers: "What tech stack
        did Ahmed use?" matches the lead of skills.md ("Skills are grouped by how
        strongly they are evidenced ... use more of the stack"), but the bullets
        that actually answer it -- ``- Python``, ``- JavaScript basics`` -- repeat
        none of the question's words, so they were pruned and the reply degraded
        to prose about how skills are grouped.

        A question with no content terms beyond the name ("Tell me about Ahmed")
        keeps every block: there is nothing to be off-topic about.

        A document that records an explicit absence is always kept. "How long has
        he worked?" shares no vocabulary with experience.md, but
        "Ahmed has no formal employment history" is the answer to it, and
        declining would replace a recorded absence with a vague "not available".

        When *every* concept is a term that routed the question, the routed
        document is on topic by construction and the per-concept test is skipped.
        That is what lets "What does Ahmed do for fun?" reach interests.md: "fun"
        is the word that selected the interests domain, so the document's content
        *is* the answer. The exemption is deliberately narrow -- it needs every
        concept accounted for, which "What is Ahmed's favourite movie?" fails, because
        "movie" neither appears in interests.md nor routed anything.

        That question still has its subject matter pruned, and still declines.

        ``recognised`` widens which words count as having routed the question.
        The exemption is about *synonyms the corpus need not share*, and a word
        that matched an intent which lost the routing contest is exactly that:
        "What's his academic background?" decides on "academic", but "background"
        matched too, and requiring education.md to contain either word is asking
        it to use the questioner's phrasing. Only the primary document benefits,
        and it is the document routing selected, so the widening cannot admit
        anything the router did not already point at. A word that routed nothing
        at all is unaffected -- that is what keeps "favourite car" and "father"
        declining.

        ``spanned_files`` extends that exemption to the domains the question *named*,
        and to a composite intent's declared expansion. "Tell me about Ahmed's
        projects and the technologies he uses" named a second thing at
        near-equal weight, and requiring a project document to contain the
        literal word "projects" is demanding a coincidence; without this the
        question was answered from skills.md alone, having routed to two
        domains. "What's Ahmed's background?" is composite -- no single document
        holds the answer -- so its expansion counts too, and the reply stops
        being his name and nothing else.

        A simple intent's expansion is excluded. "What is Ahmed focused on?" is
        answered by goals.md outright, and letting identity.md in on the grounds
        that focus spans it appended a restatement of his name to an already
        complete sentence.

        ``framing`` names the query words that request a description instead of
        naming a subject -- see
        :func:`~app.services.answering.intents.overview_framing_terms` -- and they
        are dropped from the concepts the document must match. They are not a
        demand about content: "Any chance you could walk me through his academic
        history?" contributes "history", which routes nothing of its own (it is
        an :data:`~app.services.answering.intents.AMBIGUOUS_SURFACES` member, so
        it cannot even begin a route) and which education.md is under no obligation
        to contain. Only "academic" says what was asked, only "academic" routed,
        and demanding that education.md repeat the questioner's phrasing is
        refusing an overview question the knowledge base answers. Every remaining
        concept still has to be accounted for exactly as before, so "favourite
        car", "bank balance" and "father" -- none of which is a framing word --
        are untouched.
        """
        concepts = [
            term
            for term in query_terms
            if term not in _SUBJECT_ALIASES and term not in framing
        ]
        if not concepts:
            return list(blocks)
        if (
            primary_files
            and routed
            and all(
                self._concept_accounted_for(concept, routed | recognised) for concept in concepts
            )
            and self._has_positive_accounting(concepts, routed | recognised)
        ):
            domain = self._profile.domain_of(primary_files)
            # Every concept that routed the question is accounted for, so the
            # primary document is on topic by construction. "fun" is how a
            # question asks about interests, and requiring interests.md to
            # contain the string "fun" is requiring a coincidence.
            #
            # The exemption is deliberately scoped to the primary document.
            # Multi-domain routing admits other documents that earned their
            # place by naming a *different* concept -- about.md is admitted for
            # "What is Ahmed focused on?" because it holds the focus areas --
            # and those must still pass the per-concept test on their own.
            # Without that limit "What are Ahmed's goals?" was answered out of
            # about.md, on the grounds that "goals" routed the question.
            #
            # The cross-cutting FAQ is exempt too, because it restates facts in
            # question form and "not recorded" answers stored under the very
            # question being asked. It is per-block, not per-document: only the
            # matching entry survives, so the FAQ cannot flood a domain.
            logger.debug(
                "every concept (%s) routed the question; keeping the primary "
                "document unconditionally",
                ", ".join(concepts),
            )
            exempt = {
                block
                for block in blocks
                if block.source_file in primary_files or block.source_file in spanned_files
            }
            for block in blocks:
                if block in exempt:
                    continue
                # The cross-cutting FAQ's own entry for this domain. Its sections
                # are already gated by heading in :meth:`_eligible`, so admitting
                # the matching one adds the question-shaped restatement ("Three,
                # all designed and built by him: ...") without reopening the
                # leakage that scoping to the primary document was there to stop.
                # Without it, "What has he worked on?" had no evidence at all: the
                # project documents were not retrieved and the FAQ's Projects
                # section never says the word "worked".
                if self._is_cross_cutting_entry_for(block, domain):
                    exempt.add(block)
                    continue
                if _is_on_topic(block, concepts, allow_absence=allow_absence):
                    exempt.add(block)
            if exempt:
                return [block for block in blocks if block in exempt]
        on_topic_files: set[str] = set()
        for block in blocks:
            if _is_on_topic(block, concepts, allow_absence=allow_absence):
                on_topic_files.add(block.source_file)
        kept = [block for block in blocks if block.source_file in on_topic_files]
        if len(kept) != len(blocks):
            logger.debug(
                "dropped %d of %d routed block(s) from %d off-topic document(s) for concepts %s",
                len(blocks) - len(kept),
                len(blocks),
                len({block.source_file for block in blocks}) - len(on_topic_files),
                ", ".join(concepts),
            )
        return kept

    def _render(
        self,
        selected: Sequence[_Candidate],
        query_terms: Sequence[str],
        routing: Routing,
        project: str | None,
        *,
        mode: str,
    ) -> ComposedAnswer:
        """Turn selected statements into the final answer text.

        Layout is chosen by the question's intent, not by the shape of the
        retrieved text. See :meth:`_prose` for why the sentences themselves are
        never rewritten.
        """
        shape = AnswerShape.LIST if mode == "journey" else routing.shape
        if shape is AnswerShape.FACT:
            # A lookup wants the value, not a paragraph. The first statement is
            # the highest-ranked match, which is why the ranking cannot be relaxed
            # for this shape: "What is his CGPA?" must reach "9.11." and nothing
            # else.
            statements = tuple(item.statement for item in selected[:1])
            text = _as_sentence(statements[0]) if statements else ""
        elif shape is AnswerShape.PROSE:
            statements = tuple(item.statement for item in selected)
            text = _as_sentence(self._prose(selected))
        else:
            statements = tuple(item.statement for item in selected)
            text = "\n".join(f"- {statement}" for statement in statements)
        citations = _ordered_unique(item.citation for item in selected)
        matched = tuple(
            term
            for term in query_terms
            if any(terms_match(term, item.surface) for item in selected)
        )
        logger.info(
            "composed answer: intent=%s shape=%s domains=%s project=%s mode=%s "
            "statements=%d citations=%d",
            routing.intent.value,
            shape.value,
            ",".join(domain.value for domain in routing.domains),
            project or "-",
            mode,
            len(statements),
            len(citations),
        )
        return ComposedAnswer(
            text=text,
            statements=statements,
            citations=citations,
            domain=routing.primary,
            project=project,
            matched_terms=matched,
        )

    @staticmethod
    def _prose(selected: Sequence[_Candidate]) -> str:
        """Join selected statements into readable prose, deterministically.

        The rules, in order:

        * **Drop scaffolding.** A document's opening line often describes the
          document ("They describe how Ahmed spends time away from study and
          project work") rather than the subject, and its ``label: value`` rows
          are a table flattened onto one line. Prose has nowhere to bury either --
          there is no bullet to push them down the page -- so both are dropped.
          A record row is still the right answer to a question that named its
          label ("What is Ahmed's certificate ID?"), which is why the drop is
          here and not in :meth:`_collect`.
        * **Keep document order.** The statements were selected in the order a
          human would read them, and reordering by score is what produced answers
          that jumped between topics mid-paragraph.
        * **Bridge a dropped opener.** When the first surviving statement opens
          with a pronoun it no longer has an antecedent -- "He enjoys turning
          class concepts into..." with the sentence that introduced him gone --
          so the subject is restored by prefixing the name.
        * **Cap the paragraph.** :data:`MAX_PROSE_CLAUSES` sentences. A paragraph
          that runs on stops being readable.

        No sentence is ever paraphrased, merged or reworded. Every clause is a
        verbatim statement from a retrieved chunk, which is what keeps the
        end-to-end grounding check meaningful: it can compare the whole rendered
        paragraph against the corpus, not only bullet-prefixed lines.
        """
        kept = [
            item
            for item in selected
            if not _is_document_scaffolding(item.statement) and not _is_field_entry(item.statement)
        ]
        if not kept:
            kept = list(selected)
        clauses: list[str] = []
        for index, item in enumerate(kept[:MAX_PROSE_CLAUSES]):
            statement = item.statement.strip()
            if index == 0:
                statement = _name_the_subject(statement)
            clauses.append(_as_sentence(statement))
        return " ".join(clause for clause in clauses if clause)


def _unit_idf(_term: str) -> float:
    """Neutral term weight, used when no lexical prior is available."""
    return 1.0


def _is_document_scaffolding(statement: str) -> bool:
    """True when a statement describes the document rather than the subject."""
    return bool(_DOCUMENT_SCAFFOLDING_RE.match(statement))


def _name_the_subject(statement: str) -> str:
    """Replace a leading subject pronoun with the subject's own name.

    ``He is available part time`` becomes ``Ahmed is available part time``.

    The pronoun is *replaced*, not prefixed. Prefixing reads as "Ahmed he is
    available", which is how a mechanical fix betrays itself. Only the three
    person pronouns are replaced: ``it`` refers to the thing being described, and
    "It is not recorded" is a statement about the record rather than about Ahmed,
    so it keeps its own subject.
    """
    match = _ANAPHORIC_OPENING_RE.match(statement)
    if match is None:
        return statement
    if match.group(0).strip().lower() not in _SUBJECT_PRONOUNS:
        return statement
    return _SUBJECT_NAME + statement[match.end() :]


def _as_sentence(text: str) -> str:
    """Give ``text`` terminal punctuation, so clauses join as sentences.

    The corpus already punctuates most of its prose; this covers the bullets and
    table rows that do not, so a paragraph never runs two fragments together.
    Punctuation is the only thing ever added. No word is chosen, replaced or
    reordered here -- see :meth:`AnswerComposer._prose`.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return stripped + "."


def _section_relevance(block: ContextBlock, terms: Sequence[str]) -> tuple[int, str, int]:
    """Sort key placing the section most relevant to ``terms`` first.

    Counts how many of the question's terms the section's heading and text
    mention, then falls back to document order. Deterministic, and it needs no
    notion of what the section *means* -- only whether the question's own words
    appear in it.
    """
    surface = frozenset(content_terms(f"{block.section} {block.text}"))
    hits = sum(1 for term in terms if terms_match(term, surface))
    return (-hits, block.source_file, block.ordinal)


def _per_document_budget(total: int, documents: int) -> int:
    """How many statements one document may contribute to a multi-document answer.

    A composite question is answered by several documents at once, so the budget
    is shared rather than applied per document. Otherwise "What's Ahmed's
    background?" would pull five statements from each of three documents and
    answer a broad question with a wall of text -- the exact failure the bounded
    expansion exists to prevent.
    """
    return max(1, total // max(1, documents))


def _clip(statement: str) -> str:
    """Trim a statement to :data:`MAX_STATEMENT_CHARS` on a word boundary."""
    if len(statement) <= MAX_STATEMENT_CHARS:
        return statement
    clipped = statement[:MAX_STATEMENT_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{clipped}..."


def _is_standalone(unit: EvidenceUnit, statement: str) -> bool:
    """True when a short unit is still a complete answer.

    Two cases survive the minimum-length check:

    * a bullet, because "Python" and "pandas" *are* the recorded skills, and
    * a fragment carrying a number, because "9.11." is the recorded CGPA.
    """
    return unit.is_list_item or any(character.isdigit() for character in statement)


def _dedupe_by(
    candidates: Sequence[_Candidate], duplicate: Callable[[_Candidate, _Candidate], bool]
) -> list[_Candidate]:
    """Drop candidates that ``duplicate`` considers the same as one already kept."""
    kept: list[_Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.key in seen:
            continue
        if any(duplicate(candidate, other) for other in kept):
            continue
        seen.add(candidate.key)
        kept.append(candidate)
    return kept


def _restates(left: _Candidate, right: _Candidate) -> bool:
    """True when one statement says what the other already said.

    Compares only terms rare enough to identify a fact -- "Ahmed", "one" and
    "this" weigh nothing, so they cannot manufacture agreement -- and requires
    the shorter statement to be largely accounted for by the longer one.
    """
    if len(left.weights) < REDUNDANT_MIN_TERMS or len(right.weights) < REDUNDANT_MIN_TERMS:
        return False
    shorter, longer = (left, right) if len(left.weights) <= len(right.weights) else (right, left)
    if not shorter.weights:
        return False
    shared = sum(1 for term in shorter.weights if terms_match(term, frozenset(longer.weights)))
    return shared / len(shorter.weights) >= REDUNDANT_OVERLAP


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two term sets."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    """De-duplicate strings, preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


__all__ = ["AnswerComposer", "ComposedAnswer"]
