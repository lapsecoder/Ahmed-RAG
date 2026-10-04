"""Deterministic question intents: what is being asked, and which documents can answer it.

Why this module exists
----------------------
Routing used to be a count of distinct query words against a per-domain keyword
set. That is a closed list, so a question had to *contain* a listed word to reach
a document, and most of the natural ways to ask were missing: "What is Ahmed
focused on?", "What is Ahmed good at?", "Where does he study?" and "What has he
worked on?" all routed nowhere and were answered "I don't have that information"
by a knowledge base that answers every one of them.

Adding those words to the list would have fixed the symptom and left the
mechanism broken, so the list was replaced by a *scored* router with four
independent signals. Any one of them is enough; they are combined so a question
that matches none of the first three still has a fourth, honest answer.

1. **Concept groups** (:data:`CONCEPT_GROUPS`). A group is a *concept* -- "what
   somebody does for fun" -- carrying the surface words that English uses for it
   and the corpus domains whose documents hold the answer. Groups are weighted,
   so a term that names a domain outright ("diploma") outweighs a term that only
   hints at one ("study"). This is where paraphrases are absorbed, and it is a
   closed set of concepts rather than a growing pile of individual questions.

2. **Frames** (:data:`FRAMES`). Normalised multi-word patterns for the
   constructions English uses to ask these things: ``what can <subject> do``,
   ``what is <subject> good at``, ``work(s|ed) on``, ``for fun``. Slots are
   wildcards, so ``what can Ahmed do`` and ``what can he do`` are the same frame
   -- and a phrase nobody wrote a frame for can still be caught by signal 1.

3. **Conservative fallback**. A question about the subject that carries no other
   domain signal is an identity question. This is deliberately narrow: it fires
   only when every content word in the question is the subject's own name, so
   "What is Ahmed's salary expectation?" -- which carries real content words
   that match nothing -- does *not* get read as a request for an introduction.

4. **Nothing**. No signal matched, and the question has content words of its own.
   The router returns :attr:`Intent.UNKNOWN` and the composer declines. Guessing
   here is what produced confident answers about the wrong subject matter.

Why an unseen paraphrase routes correctly
-----------------------------------------
"Can you give me a quick introduction to Ahmed?" was never added as a rule. It
routes because:

* "introduction" matches the identity concept group (signal 1), and
* it names the subject, so even if that group were removed the fallback (signal
  3) would still route it to the identity document.

Any sentence containing a word from a known concept group routes to that group's
domains. That is the whole claim: routing is a function of concept membership,
not of a list of sentences.

Multi-domain
------------
Some questions legitimately span documents -- "What's Ahmed's background?" needs
the profile, the education record and the goals. :class:`Routing` carries a
primary domain and a *bounded* set of secondary domains
(:data:`MAX_EXPANSION_DOMAINS`). The bound is the point: a broad question gets a
wider scope, never an unbounded one, so "tell me everything" cannot turn into
dumping the knowledge base.

Everything here is pure and deterministic. No model, no randomness, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.services.text import _WORD_RE, content_terms

_WORD_SET_RE: Final[re.Pattern[str]] = _WORD_RE


class Domain(StrEnum):
    """A section of the knowledge base a question can be about."""

    IDENTITY = "identity"
    EDUCATION = "education"
    SKILLS = "skills"
    CERTIFICATIONS = "certifications"
    PROJECTS = "projects"
    EXPERIENCE = "experience"
    AVAILABILITY = "availability"
    GOALS = "goals"
    INTERESTS = "interests"
    CONTACT = "contact"
    UNKNOWN = "unknown"


#: Frontmatter ``category`` values that back each domain. Derived from the
#: corpus, not hardcoded per file, so a renamed document keeps working.
DOMAIN_CATEGORIES: Final[dict[Domain, tuple[str, ...]]] = {
    Domain.IDENTITY: ("profile",),
    Domain.EDUCATION: ("education",),
    Domain.SKILLS: ("skills",),
    Domain.CERTIFICATIONS: ("certifications",),
    Domain.PROJECTS: ("project",),
    Domain.EXPERIENCE: ("experience",),
    Domain.AVAILABILITY: ("availability",),
    Domain.GOALS: ("goals",),
    Domain.INTERESTS: ("interests",),
    Domain.CONTACT: ("contact",),
    Domain.UNKNOWN: (),
}

#: Cross-cutting document allowed as secondary evidence for any non-project
#: domain. It restates facts in question form, which answers pointed questions
#: ("What is his CGPA?") far better than the narrative documents do.
CROSS_CUTTING_CATEGORY: Final[str] = "faq"

#: Tie-break order for :func:`detect_domain`. A question scoring equally for two
#: domains resolves to the earlier one, so routing never depends on dictionary
#: iteration order.
DOMAIN_PRIORITY: Final[tuple[Domain, ...]] = (
    Domain.CERTIFICATIONS,
    Domain.CONTACT,
    Domain.EDUCATION,
    Domain.AVAILABILITY,
    Domain.EXPERIENCE,
    Domain.SKILLS,
    Domain.GOALS,
    Domain.INTERESTS,
    Domain.PROJECTS,
    Domain.IDENTITY,
)


class Intent(StrEnum):
    """What the user is asking for.

    Deliberately small. Every entry names a shape of answer the corpus can
    actually produce; nothing here is a topic the knowledge base lacks, so there
    is no "does he know Kotlin?" intent waiting to be unable to answer.
    """

    IDENTITY = "identity"
    BACKGROUND = "background"
    PROJECTS = "projects"
    PROJECT_DETAIL = "project_detail"
    SKILLS = "skills"
    EDUCATION = "education"
    #: Not in the brief's list, but the corpus has ``experience.md`` and two
    #: existing end-to-end cases ask about work history. "How long has Ahmed
    #: worked?" is a real question shape, so it needs a home.
    EXPERIENCE = "experience"
    INTERESTS = "interests"
    AVAILABILITY = "availability"
    CONTACT = "contact"
    CERTIFICATIONS = "certifications"
    FOCUS = "focus"
    #: Stated aims, as distinct from current attention. goals.md holds the former
    #: and about.md holds the latter, so they are separate intents even though
    #: both are read from "goals.md + about.md" when a question spans them.
    GOALS = "goals"
    UNKNOWN = "unknown"


#: Tie-break order when two intents score equally. More specific subjects first:
#: a question carrying both "certification" and "skill" is about the
#: certification, because that document names its issuer and its hours.
INTENT_PRIORITY: Final[tuple[Intent, ...]] = (
    Intent.CERTIFICATIONS,
    Intent.CONTACT,
    Intent.EDUCATION,
    Intent.AVAILABILITY,
    Intent.EXPERIENCE,
    Intent.SKILLS,
    Intent.FOCUS,
    Intent.GOALS,
    Intent.INTERESTS,
    Intent.PROJECTS,
    Intent.BACKGROUND,
    Intent.IDENTITY,
)


class AnswerShape(StrEnum):
    """How the answer should be laid out once its statements are chosen.

    Chosen by intent rather than by the shape of the retrieved text, because the
    shape of the answer is a property of the question: "What are his skills?"
    wants an inventory whether or not the evidence happens to be a bulleted list.
    """

    #: One or more complete sentences, read as prose.
    PROSE = "prose"
    #: A bulleted inventory, for questions whose answer *is* a list.
    LIST = "list"
    #: A single quoted fact.
    FACT = "fact"


#: Primary domain for each intent, and the shape its answer takes.
INTENT_SHAPE: Final[dict[Intent, tuple[Domain, AnswerShape]]] = {
    Intent.IDENTITY: (Domain.IDENTITY, AnswerShape.PROSE),
    Intent.BACKGROUND: (Domain.IDENTITY, AnswerShape.PROSE),
    Intent.FOCUS: (Domain.GOALS, AnswerShape.PROSE),
    Intent.PROJECTS: (Domain.PROJECTS, AnswerShape.LIST),
    Intent.PROJECT_DETAIL: (Domain.PROJECTS, AnswerShape.PROSE),
    Intent.SKILLS: (Domain.SKILLS, AnswerShape.LIST),
    Intent.EDUCATION: (Domain.EDUCATION, AnswerShape.PROSE),
    Intent.GOALS: (Domain.GOALS, AnswerShape.PROSE),
    Intent.EXPERIENCE: (Domain.EXPERIENCE, AnswerShape.PROSE),
    Intent.INTERESTS: (Domain.INTERESTS, AnswerShape.LIST),
    Intent.AVAILABILITY: (Domain.AVAILABILITY, AnswerShape.PROSE),
    Intent.CONTACT: (Domain.CONTACT, AnswerShape.PROSE),
    Intent.CERTIFICATIONS: (Domain.CERTIFICATIONS, AnswerShape.LIST),
    Intent.GOALS: (Domain.GOALS, AnswerShape.PROSE),
    Intent.EXPERIENCE: (Domain.EXPERIENCE, AnswerShape.PROSE),
    Intent.UNKNOWN: (Domain.UNKNOWN, AnswerShape.PROSE),
}

#: Domains a *composite* intent may also draw on, in the order they should be
#: quoted. "What's Ahmed's background?" is answered from who he is, where he
#: studies and what he is aiming at -- three documents, each already existing,
#: none of them combined into a new claim.
INTENT_EXPANSION: Final[dict[Intent, tuple[Domain, ...]]] = {
    Intent.BACKGROUND: (Domain.EDUCATION, Domain.GOALS),
    Intent.FOCUS: (Domain.IDENTITY, Domain.SKILLS),
    Intent.GOALS: (),
}

#: Intents whose answer is *assembled* from their expansion, rather than led by
#: it. Only these may draw an expansion document without that document having to
#: repeat the question's own words.
#:
#: The distinction is the difference between two kinds of expansion, and both
#: kinds were previously treated alike, wrongly in each direction.
#:
#: "What's Ahmed's background?" is a question no single document answers. The
#: expansion is the answer -- who he is, where he studies, what he is aiming at
#: -- so ``about.md`` leading with his name and stopping there is a thin answer
#: to a question the knowledge base answers in three places. Each expansion
#: document holds a real part of it, none of which has to contain the word
#: "background".
#:
#: "What is Ahmed focused on?" is not that at all. ``goals.md`` holds the
#: answer outright; identity and skills are supporting context that happens to be
#: worth having. Forcing them in produced "Ahmed's goals are to build a future
#: with AI... Mohammed Ayaan Ahmed, commonly known as Ahmed." -- a sentence of
#: his own name bolted onto the end of an answer that was already complete. That
#: is padding, and padding is the failure this phase exists to remove.
#:
#: So an expansion document is trusted on the strength of *why* it was admitted:
#: the question named it (a co-answer), or the intent is composite and cannot be
#: answered without it. Everything else must still show it is on topic.
COMPOSITE_INTENTS: Final[frozenset[Intent]] = frozenset({Intent.BACKGROUND})

#: A second intent is admitted as a co-answer when it scores at least this
#: fraction of the winner's.
#:
#: This is how a genuinely two-part question is recognised without a rule per
#: question: "Tell me about Ahmed's projects and the technologies he uses" raises
#: *building* ("projects") and *capability* ("technologies") at equal weight, so
#: both documents are admitted and neither can answer alone. A question that
#: mentions one concept far more strongly than another does not qualify, which
#: is what keeps "What has Ahmed built?" from dragging in the skills document.
MULTI_DOMAIN_RATIO: Final[float] = 0.9

#: Hard ceiling on secondary domains for one question. A broader question gets a
#: wider scope; it never gets an unbounded one.
MAX_EXPANSION_DOMAINS: Final[int] = 2

#: Intents whose answer *is* the routed document rather than a sentence inside it.
#:
#: "What are his skills?" wants skills.md's inventory, not whichever line happens
#: to contain the word "skills", so these intents quote the document itself.
#:
#: :attr:`Intent.PROJECTS` is deliberately absent even though its shape is
#: :attr:`AnswerShape.LIST`. There are several project documents, so "What has
#: Ahmed built?" has no single document to quote: the answer is the inventory
#: sentence that names all three, which a matching search finds and a
#: document-order walk does not. A list *shape* and a whole-document *strategy*
#: are different properties, and conflating them is what answered that question
#: with MovieMind's search-and-rank bullets.
WHOLE_DOCUMENT_INTENTS: Final[frozenset[Intent]] = frozenset(
    {Intent.SKILLS, Intent.CERTIFICATIONS, Intent.INTERESTS}
)

#: Weight of a term that names a domain outright.
STRONG: Final[float] = 2.0
#: Weight of a term that only hints at one.
WEAK: Final[float] = 1.0
#: Weight of a matched grammatical frame. Frames are the most specific signal
#: available -- "what is Ahmed good at" cannot be read as anything but a
#: capabilities question -- so they outweigh single words.
FRAME_WEIGHT: Final[float] = 2.5

#: Surfaces that name a subject matter only under a reading the word cannot
#: supply on its own, so they need a second, unambiguous signal before they may
#: establish a route.
#:
#: These are not words that are "sometimes wrong". They are words whose referent
#: is fixed by a modifier that may or may not be present: a *bank* balance, an
#: *account* balance, a *work-life* balance and a *skill* balance are four
#: different questions that share one noun; an *academic* history, a *work*
#: history, an *employment* history and a *git* history likewise. Nothing in
#: the noun picks which one was asked.
#:
#: So these count toward their group's score but cannot *begin* one. A group
#: whose only evidence is an ambiguous surface does not route at all. The word
#: still reaches :attr:`Routing.vocabulary` when some other group did route, so
#: "What is Ahmed's employment history?" is unaffected -- "employment"
#: corroborates "history" inside the very same group -- while "his academic
#: history" no longer drags in the experience document on the strength of a noun
#: that "academic" has already claimed for education.
#:
#: This is a property of the word, not of the question, so it needs no rule per
#: phrasing and cannot be defeated by paraphrasing the question around it.
#:
#: The set is deliberately short, and each entry had to be earned by a question
#: that mis-routed without it. "degree" is the instructive exclusion: it is
#: polysemous in the abstract ("to what degree"), but in this knowledge base it
#: is the only word that routes "What is his degree?" to education, so marking
#: it ambiguous silenced a question that was already correct.
AMBIGUOUS_SURFACES: Final[frozenset[str]] = frozenset(
    {
        # "a *bank* balance", "an *account* balance", "a *work-life* balance",
        # "a *skills* balance" -- and the one reading that belongs to this
        # system, "he balances his projects", needs the rest of the question.
        "balance",
        # "an *academic* history", "a *work* history", "a *git* history".
        "history",
    }
)


@dataclass(frozen=True, slots=True)
class ConceptGroup:
    """One concept, the words English uses for it, and the domains behind it.

    A group is the unit that absorbs paraphrase. "fun", "hobbies", "enjoy",
    "into" and "free time" are five surfaces of one idea, and interests.md is
    where the answer lives -- so all five route to interests regardless of which
    one a user happens to pick.

    :attr:`ambiguous` names the members that cannot identify the concept by
    themselves; see :data:`AMBIGUOUS_SURFACES`.
    """

    name: str
    intent: Intent
    domains: tuple[Domain, ...]
    surfaces: frozenset[str]
    weight: float = STRONG

    @property
    def ambiguous(self) -> frozenset[str]:
        """The subset of :attr:`surfaces` that needs corroboration to count."""
        return frozenset(word for word in self.surfaces if word in AMBIGUOUS_SURFACES)


#: The routing vocabulary, grouped by concept rather than by domain.
#:
#: The old table was keyed by domain, which made every new phrasing an edit to
#: the domain that happened to own it. Keying by concept instead means a
#: paraphrase joins an existing group -- and a group may legitimately point at
#: more than one domain, which is what makes multi-domain questions expressible.
CONCEPT_GROUPS: Final[tuple[ConceptGroup, ...]] = (
    ConceptGroup(
        name="identity-core",
        intent=Intent.IDENTITY,
        domains=(Domain.IDENTITY,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "who",
                "name",
                "named",
                "nickname",
                "alias",
                "aliases",
                "identity",
                "called",
                "profile",
                "introduce",
                "introduction",
                "introducing",
                "initial",
                "initials",
                "brand",
                "subject",
            }
        ),
    ),
    ConceptGroup(
        name="self-reference",
        intent=Intent.IDENTITY,
        domains=(Domain.IDENTITY,),
        weight=WEAK,
        surfaces=frozenset({"about", "describe", "overview", "summary", "summarise", "summarize"}),
    ),
    ConceptGroup(
        name="background",
        intent=Intent.BACKGROUND,
        domains=(Domain.IDENTITY,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "background",
                "backstory",
                "story",
                "trajectory",
                "journey",
                "snapshot",
                "picture",
                "roots",
                "origins",
                "pathway",
            }
        ),
    ),
    ConceptGroup(
        name="education",
        intent=Intent.EDUCATION,
        domains=(Domain.EDUCATION,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "education",
                "educational",
                "educate",
                "academic",
                "academically",
                "academia",
                "study",
                "studies",
                "studying",
                "studied",
                "diploma",
                "degree",
                "qualification",
                "qualifications",
                "university",
                "college",
                "polytechnic",
                "institution",
                "cgpa",
                "gpa",
                "grade",
                "grading",
                "semester",
                "graduation",
                "graduate",
                "course",
                "coursework",
                "major",
                "alma",
                "pursuing",
                "pursue",
            }
        ),
    ),
    ConceptGroup(
        name="capability",
        intent=Intent.SKILLS,
        domains=(Domain.SKILLS,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "skill",
                "skills",
                "tech",
                "technology",
                "technologies",
                "stack",
                "tool",
                "tools",
                "tooling",
                "programming",
                "language",
                "languages",
                "framework",
                "frameworks",
                "proficiency",
                "proficient",
                "expertise",
                "expert",
                "ability",
                "abilities",
                "capability",
                "capabilities",
                "talents",
                "good",
                "strong",
                "skilled",
                "strongest",
                "best",
                "knows",
                "know",
                "known",
                "familiar",
                "competent",
            }
        ),
    ),
    ConceptGroup(
        name="tooling",
        intent=Intent.SKILLS,
        domains=(Domain.SKILLS,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "python",
                "javascript",
                "typescript",
                "react",
                "fastapi",
                "docker",
                "faiss",
                "ollama",
                "sqlalchemy",
                "postgresql",
                "tailwind",
                "numpy",
                "pandas",
                "tensorflow",
                "pytorch",
                "embeddings",
                "astro",
                "bm25",
                "bm",
                "rag",
            }
        ),
    ),
    ConceptGroup(
        name="certification",
        intent=Intent.CERTIFICATIONS,
        domains=(Domain.CERTIFICATIONS,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "certification",
                "certifications",
                "certificate",
                "certificates",
                "certified",
                "credential",
                "credentials",
                "badge",
                "exam",
                "aws",
                "springboard",
                "infosys",
            }
        ),
    ),
    ConceptGroup(
        name="building",
        intent=Intent.PROJECTS,
        domains=(Domain.PROJECTS,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "project",
                "projects",
                "built",
                "build",
                "builds",
                "building",
                "made",
                "make",
                "creates",
                "created",
                "create",
                "shipped",
                "app",
                "apps",
                "application",
                "applications",
            }
        ),
    ),
    ConceptGroup(
        name="building-generic",
        intent=Intent.PROJECTS,
        domains=(Domain.PROJECTS,),
        weight=WEAK,
        surfaces=frozenset(
            {
                # These are all real project vocabulary, and all of them are
                # ordinary English words with several other homes. "Which
                # platform issued Ahmed's certifications?" is not a question
                # about something he built -- the platform is Springboard, a
                # course provider -- yet at full weight "platform" tied with
                # "certifications", became a co-answer, and the reply listed
                # ResumeForge's feature bullets. The citation checked out and
                # the subject was wrong, which is the failure shape this whole
                # phase is trying to remove.
                #
                # At weak weight they still route when nothing else speaks
                # ("What are Ahmed's systems?") and still contribute to a
                # genuinely two-part question, but they cannot outvote a
                # specific domain the way an unambiguous word like "built" can.
                "platform",
                "platforms",
                "engine",
                "repository",
                "repo",
                "deployed",
                "deployment",
                "architecture",
                "portfolio",
                "system",
                "systems",
                "tool",
                "tools",
                "toolkit",
                "product",
                "products",
            }
        ),
    ),
    ConceptGroup(
        name="employment",
        intent=Intent.EXPERIENCE,
        domains=(Domain.EXPERIENCE,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "experience",
                "experienced",
                "work",
                "works",
                "worked",
                "working",
                "employment",
                "employed",
                "job",
                "jobs",
                "role",
                "roles",
                "professional",
                "professionally",
                "company",
                "companies",
                "employer",
                "history",
                "years",
                "workplace",
                "stint",
                "stints",
            }
        ),
    ),
    ConceptGroup(
        name="availability",
        intent=Intent.AVAILABILITY,
        domains=(Domain.AVAILABILITY,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "available",
                "availability",
                "hire",
                "hiring",
                "hireable",
                "freelance",
                "freelancing",
                "internship",
                "internships",
                "intern",
                "collaborate",
                "collaboration",
                "collaborations",
                "services",
                "service",
                "offer",
                "offering",
                "offerings",
                "book",
                "bookable",
                "schedule",
                "rsvp",
                "rate",
                "pricing",
                "looking",
                "seeking",
                "seeks",
                "vacancy",
                "open",
            }
        ),
    ),
    # "What are Ahmed's goals?" and "What is Ahmed focused on?" are different
    # questions, and the corpus answers them from different documents. Splitting
    # them is what keeps a question about aims from also reading about.md, whose
    # "Current focus areas" section answers the second but not the first.
    ConceptGroup(
        name="goal",
        intent=Intent.GOALS,
        domains=(Domain.GOALS,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "goal",
                "goals",
                "aim",
                "aims",
                "aiming",
                "ambition",
                "ambitions",
                "ambitious",
                "aspire",
                "aspires",
                "aspiring",
                "plan",
                "plans",
                "planning",
                "career",
                "careers",
                "future",
                "milestone",
            }
        ),
    ),
    ConceptGroup(
        name="focus",
        intent=Intent.FOCUS,
        domains=(Domain.GOALS, Domain.IDENTITY),
        weight=STRONG,
        surfaces=frozenset(
            {
                "focus",
                "focused",
                "focuses",
                "focussing",
                "focusing",
                "concentration",
                "concentrates",
                "concentrating",
                "priority",
                "priorities",
                "direction",
                "interested",
                "keen",
                "passion",
                "passionate",
                "specialisation",
                "specialization",
            }
        ),
    ),
    ConceptGroup(
        name="hobby",
        intent=Intent.INTERESTS,
        domains=(Domain.INTERESTS,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "interest",
                "interests",
                "hobby",
                "hobbies",
                "pastime",
                "pastimes",
                "fun",
                "enjoy",
                "enjoys",
                "enjoyed",
                "enjoyment",
                "like",
                "likes",
                "liked",
                "favourite",
                "favorite",
                "into",
                "keen",
                "leisure",
                "anime",
                "manga",
                "gaming",
                "games",
                "comic",
                "comics",
                "personal",
                # Ambiguous, not dropped: see :data:`AMBIGUOUS_SURFACES`. Read
                # alongside "fun" or "pastime" this is the work-life-balance
                # reading and interests is right; read alone it is whatever
                # modifier happened to attach, which is how "What is Ahmed's
                # bank balance?" used to be answered with his hobbies.
                "balance",
                "hometown",
            }
        ),
    ),
    ConceptGroup(
        name="location",
        intent=Intent.IDENTITY,
        domains=(Domain.IDENTITY,),
        weight=STRONG,
        surfaces=frozenset({
            "live",
            "lives",
            "living",
            "based",
            "location",
            "located",
            "hometown",
        }),
    ),
    ConceptGroup(
        name="contact",
        intent=Intent.CONTACT,
        domains=(Domain.CONTACT,),
        weight=STRONG,
        surfaces=frozenset(
            {
                "contact",
                "contacts",
                "email",
                "emails",
                "mail",
                "phone",
                "mobile",
                "linkedin",
                "github",
                "instagram",
                "whatsapp",
                "telegram",
                "reach",
                "touch",
                "dm",
            }
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Frame:
    """A normalised multi-word construction that implies an intent.

    Slots are wildcards, which is what makes these generalise. ``what can <any>
    do`` covers "what can Ahmed do", "what can he do" and "what can you do"
    without a rule per phrasing, and any word the slot takes that happens to
    belong to another concept group will additionally raise *that* intent.
    """

    name: str
    intent: Intent
    pattern: re.Pattern[str]
    domains: tuple[Domain, ...] = ()


#: Wildcard slot: one or two words, or the "of" possessive link. Kept tight so a
#: frame cannot straddle a clause and claim a subject matter it never saw.
_SLOT: Final[str] = r"(?:[\w'-]+\s+){0,2}(?:of\s+)?"

FRAMES: Final[tuple[Frame, ...]] = (
    Frame(
        name="capability-do",
        intent=Intent.SKILLS,
        pattern=re.compile(rf"\b(?:what|which)\s+can\s+{_SLOT}(?:do|handle|help)\b"),
        domains=(Domain.SKILLS,),
    ),
    Frame(
        name="capability-good-at",
        intent=Intent.SKILLS,
        # "strong in" was a second frame of its own, and both of its other
        # alternatives were already listed here, so it only ever fired *alongside*
        # this one: "what is he strong at" scored skills twice and nothing else in
        # the table could. One construction, one frame.
        pattern=re.compile(
            r"\b(?:good|strong|skilled|proficient|expert|comfortable)\s+at\b"
            r"|\b(?:strong|skilled|expert)\s+in\b"
            r"|\bwhat\s+(?:is|are)\s+" + _SLOT + r"(?:good|strong|skilled|best)\s+at\b"
        ),
        domains=(Domain.SKILLS,),
    ),
    Frame(
        name="working-on",
        intent=Intent.PROJECTS,
        pattern=re.compile(
            r"\b(?:work|works|worked|working|build|builds|built|building|make|makes|made|making)"
            r"\s+(?:on|upon)\b"
        ),
        domains=(Domain.PROJECTS,),
    ),
    Frame(
        name="work-with",
        intent=Intent.SKILLS,
        pattern=re.compile(r"\b(?:work|works|worked|working)\s+with\b"),
        domains=(Domain.SKILLS,),
    ),
    Frame(
        name="help-with",
        intent=Intent.AVAILABILITY,
        pattern=re.compile(r"\b(?:help|helps|helping|help\s+out)\s+with\b"),
        domains=(Domain.AVAILABILITY,),
    ),
    Frame(
        name="outside-leisure",
        intent=Intent.INTERESTS,
        pattern=re.compile(
            r"\b(?:outside|beyond)\s+(?:of\s+)?(?:work|study|school|college|programming|coding|class)\b"
        ),
        domains=(Domain.INTERESTS,),
    ),
    Frame(
        name="for-fun",
        intent=Intent.INTERESTS,
        pattern=re.compile(r"\bfor\s+fun\b|\bin\s+(?:his|her|their)\s+spare\s+time\b"),
        domains=(Domain.INTERESTS,),
    ),
    Frame(
        name="working-toward",
        intent=Intent.FOCUS,
        pattern=re.compile(r"\b(?:working|work|aiming|headed|driving)\s+towards?\b"),
        domains=(Domain.GOALS,),
    ),
    Frame(
        name="focused-on",
        intent=Intent.FOCUS,
        pattern=re.compile(r"\bfocus(?:es|ed|ing)?\s+(?:on|upon)\b"),
        domains=(Domain.GOALS,),
    ),
    Frame(
        name="interested-in",
        intent=Intent.FOCUS,
        pattern=re.compile(r"\binterested\s+in\b|\bkeen\s+on\b"),
        domains=(Domain.GOALS,),
    ),
    Frame(
        name="studies-where",
        intent=Intent.EDUCATION,
        pattern=re.compile(r"\b(?:where|which)\b.*\bstud(?:y|ies|ying)\b", re.IGNORECASE),
        domains=(Domain.EDUCATION,),
    ),
    Frame(
        name="can-hire",
        intent=Intent.AVAILABILITY,
        pattern=re.compile(rf"\b(?:can|could|may)\b{_SLOT}(?:hire|employ|recruit|work\s+with)\b"),
        domains=(Domain.AVAILABILITY,),
    ),
    Frame(
        name="looking-for",
        intent=Intent.AVAILABILITY,
        pattern=re.compile(r"\b(?:looking|seeking|hoping)\s+for\b"),
        domains=(Domain.AVAILABILITY,),
    ),
    Frame(
        name="introductory",
        intent=Intent.IDENTITY,
        pattern=re.compile(
            r"\b(?:intro|introductory|introduction|introductions|overview|summary|summarise|"
            r"summarize|describe|description|profile|explain)\b"
        ),
        domains=(Domain.IDENTITY,),
    ),
)


#: Words that make the *subject* of the sentence the assistant or the assistant's
#: subject, rather than the world at large.
#:
#: This is the hook the pre-retrieval classifier uses. It is not a routing signal
#: on its own -- "Where does he study?" says nothing about education until a
#: concept group or a frame says so -- it only answers "is this question about
#: the subject at all?".
SUBJECT_REFERENCES: Final[frozenset[str]] = frozenset(
    {
        "ahmed",
        "ayaan",
        "mohammed",
        "you",
        "your",
        "yours",
        "yourself",
        "u",
        "ur",
        "he",
        "his",
        "him",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "they",
        "them",
        "their",
        "theirs",
    }
)

#: Question words and request verbs. A message carrying one of these together
#: with a subject reference is a question *about* that subject; without a subject
#: reference it is a question about the world, which this assistant does not
#: answer.
QUESTION_FORMS: Final[frozenset[str]] = frozenset(
    {
        "what",
        "whats",
        "who",
        "whom",
        "whose",
        "which",
        "where",
        "when",
        "why",
        "how",
        "tell",
        "describe",
        "explain",
        "summarise",
        "summarize",
        "introduce",
        "show",
        "list",
        "give",
        "share",
        "know",
        "help",
        "brief",
        "outline",
        "walk",
        # Existential and auxiliary question forms. English asks whether
        # something is available without a wh-word -- "Is there a way to reach
        # him?", "Can anyone hire Ahmed?", "Are there any certifications?" --
        # and every one of those is a genuine question about the subject.
        #
        # This set only ever *adds* in-scope messages, and only for a message
        # that already names the subject (see QueryClassifier._subject_and_form).
        # "What is 2 + 2?" and "Recommend a restaurant near me" name nobody, so
        # they stay off-topic. An in-scope message that turns out to be
        # unanswerable gets the grounded no-information reply rather than a
        # canned refusal, which is the safer of the two outcomes.
        "is",
        "are",
        "was",
        "were",
        "am",
        "can",
        "could",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
    }
)

#: Question words that signal an overview request rather than a lookup. In
#: overview mode the composer quotes the lead of the routed document instead of
#: hunting for a matching sentence.
OVERVIEW_TERMS: Final[frozenset[str]] = frozenset(
    {
        "summarise",
        "summarize",
        "summary",
        "overview",
        "describe",
        "description",
        "introduction",
        "introduce",
        "introducing",
        "intro",
        "explain",
        "walkthrough",
        "walk",
        "profile",
        "portrait",
        "picture",
    }
)


@dataclass(frozen=True, slots=True)
class Routing:
    """The routing decision: what was asked, and which documents may answer it."""

    intent: Intent
    #: Document set the answer is drawn from, primary first.
    domains: tuple[Domain, ...]
    #: Query words that matched the *winning* intent's vocabulary, plus those of
    #: any intent admitted as a co-answer.
    terms: frozenset[str]
    #: Human-readable trace of the signals that fired, for logging and tests.
    trace: tuple[str, ...] = ()
    #: Every query word the router recognised, whichever intent it scored for.
    #: See :func:`route` for why this is not the same set as :attr:`terms`.
    vocabulary: frozenset[str] = frozenset()
    #: Domains whose documents may back this answer without independently having to
    #: match the question's vocabulary: those the question *named* alongside the
    #: primary one, plus a composite intent's declared expansion. The answer
    #: layer treats these differently from a simple intent's expansion; see
    #: :data:`COMPOSITE_INTENTS` and :func:`route`.
    spanned: tuple[Domain, ...] = ()

    @property
    def primary(self) -> Domain:
        """The domain the question is chiefly about."""
        return self.domains[0] if self.domains else Domain.UNKNOWN

    @property
    def secondary(self) -> tuple[Domain, ...]:
        """Bounded secondary domains for a multi-domain answer."""
        return tuple(domain for domain in self.domains[1:] if domain is not Domain.UNKNOWN)

    @property
    def shape(self) -> AnswerShape:
        """How the answer should be laid out."""
        return INTENT_SHAPE[self.intent][1]

    @property
    def is_composite(self) -> bool:
        """True when the answer legitimately spans more than one document."""
        return bool(self.secondary)


#: The subject's own name, excluded when judging whether a retrieved block is
#: about the question. It appears in nearly every chunk of the knowledge base.
_SUBJECT_ALIASES: Final[frozenset[str]] = frozenset({"ahmed", "ayaan", "mohammed"})


def query_words(query: str) -> frozenset[str]:
    """Every lowercase alphanumeric word in ``query``, stopwords included."""
    return frozenset(_WORD_SET_RE.findall(query.lower()))


def normalised_query(query: str) -> str:
    """``query`` lowercased and stripped of punctuation, word order preserved.

    Frames are matched against this rather than against a sorted word set, because
    "work(s) on" and "work with" are distinguished *only* by which word comes
    second. A set loses that, and both patterns would then match both questions.
    """
    return " ".join(_WORD_SET_RE.findall(query.lower()))


def _matched_surfaces(word: str, surfaces: frozenset[str]) -> bool:
    """True when ``word`` is one of ``surfaces``, allowing the known inflections."""
    if word in surfaces:
        return True
    if f"{word}s" in surfaces:
        return True
    # "studying"/"study", "focused"/"focus", "academically"/"academic": a bounded
    # prefix relation in either direction, which is exactly what terms_match does
    # and covers the inflections a concept group cannot be expected to enumerate.
    for surface in surfaces:
        if len(word) < 4 or len(surface) < 4:
            continue
        if abs(len(word) - len(surface)) > 3:
            continue
        shorter, longer = (word, surface) if len(word) <= len(surface) else (surface, word)
        if longer.startswith(shorter):
            return True
    return False


def _scored_intents(query: str) -> tuple[dict[Intent, float], dict[Intent, set[str]], list[str]]:
    """Score every intent from concept groups and frames. See :func:`route`."""
    words = query_words(query)
    normalised = normalised_query(query)
    scores: dict[Intent, float] = {}
    matched: dict[Intent, set[str]] = {}
    trace: list[str] = []

    for group in CONCEPT_GROUPS:
        hits = {word for word in words if _matched_surfaces(word, group.surfaces)}
        if not hits:
            continue
        # A group whose every hit is ambiguous has no way to tell which subject
        # matter was meant, so it does not get to start one. Corroboration is
        # per group, not per question: "employment" and "history" both live in
        # the employment group, so one corroborates the other, while "academic"
        # and "history" live in different groups and corroborate nothing.
        # See :data:`AMBIGUOUS_SURFACES`.
        if hits <= group.ambiguous:
            trace.append(f"uncorroborated:{group.name}({','.join(sorted(hits))})")
            continue
        scores[group.intent] = scores.get(group.intent, 0.0) + group.weight
        matched.setdefault(group.intent, set()).update(hits)
        trace.append(f"concept:{group.name}({','.join(sorted(hits))})")

    for frame in FRAMES:
        match = frame.pattern.search(normalised)
        if match is None:
            continue
        scores[frame.intent] = scores.get(frame.intent, 0.0) + FRAME_WEIGHT
        span_words = {word for word in query_words(match.group(0)) if word in words}
        matched.setdefault(frame.intent, set()).update(span_words)
        trace.append(f"frame:{frame.name}({match.group(0).strip()})")

    return scores, matched, trace


def _best_intent(scores: dict[Intent, float]) -> Intent:
    """Highest-scoring intent, ties broken by :data:`INTENT_PRIORITY`."""
    best = Intent.UNKNOWN
    best_score = 0.0
    for intent in INTENT_PRIORITY:
        score = scores.get(intent, 0.0)
        if score > best_score:
            best, best_score = intent, score
    return best


#: Intents that describe the subject *in general* and therefore never co-answer a
#: narrower question.
#:
#: "What's his academic background?" raises *academic* (education) and
#: *background* (identity) at equal weight. Admitting identity as a co-answer
#: looked defensible -- the background concept does point at the profile document
#: -- but the profile is not what an academic-background question asks for, and
#: the FAQ's "About Ahmed" entry then answered it with his name.
#:
#: These two intents say "tell me about him" in different words. A question that
#: already named a *specific* subject matter has said what it wants, and the
#: general documents must not dilute it.
CO_ANSWER_EXCLUDED_INTENTS: Final[frozenset[Intent]] = frozenset(
    {Intent.IDENTITY, Intent.BACKGROUND}
)


def _co_answer_intents(scores: dict[Intent, float], winner: Intent) -> tuple[Intent, ...]:
    """Intents that must join the winner because the question asks about them too.

    An intent qualifies when it scored at least :data:`MULTI_DOMAIN_RATIO` of the
    winner. That ratio is the whole mechanism for recognising a two-part question
    without a rule per question: "Tell me about Ahmed's projects and the
    technologies he uses" raises *building* and *capability* to the same score, so
    both are admitted. "What has Ahmed built?" mentions no capability word at all,
    so it stays a projects question.

    Intents are returned rather than their domains because the caller needs both:
    the domain to widen the document set, and the intent to recover the query
    words that scored for it. Returning domains alone silently lost those words,
    so a co-answered concept counted as neither routed nor attested and the
    question was declined for naming a second subject the router had just agreed
    to answer.

    A composite intent's own expansion is **not** added here; it is declared in
    :data:`INTENT_EXPANSION`. Two routes to multi-domain, deliberately kept apart:
    a composite intent knows what it spans, and a question that genuinely names
    two things is admitted because it named them.
    """
    best = scores.get(winner, 0.0)
    if best <= 0.0:
        return ()
    admitted: list[Intent] = []
    seen: set[Domain] = set()
    for intent in INTENT_PRIORITY:
        if intent is winner or intent in CO_ANSWER_EXCLUDED_INTENTS:
            continue
        if scores.get(intent, 0.0) >= best * MULTI_DOMAIN_RATIO:
            domain = INTENT_SHAPE[intent][0]
            if domain not in seen:
                seen.add(domain)
                admitted.append(intent)
    return tuple(admitted)


def route(query: str) -> Routing:
    """Route ``query`` to an intent and a bounded set of documents.

    Args:
        query: The user's question, verbatim.

    Returns:
        A :class:`Routing`. ``intent`` is :attr:`Intent.UNKNOWN` when nothing
        matched and the question carries content words of its own -- which the
        composer must treat as "no document can vouch for this", not as licence
        to answer from whatever retrieved highest.

    Two word sets come back, and they answer different questions.
    :attr:`Routing.terms` is what *shaped the decision* -- the winner's own
    vocabulary, plus any co-answer's. :attr:`Routing.vocabulary` is everything
    the router recognised, including words belonging to intents that lost.
    "What's his academic background?" scores education above background, so
    "academic" decides the route and "background" does not; but "background" is
    still a word this router knows, and demanding that some document contain it
    is demanding the corpus use the questioner's synonym. Only the first set may
    justify admitting a document on-topic, because only the first says anything
    about *which* document is relevant.

    :attr:`Routing.spanned` records the domains allowed to answer without
    matching the question's own words. A question that named two things at
    near-equal weight named both, and the answer layer should not then demand
    that the second domain's document happen to repeat its noun. Neither should
    a composite intent's expansion, because the composite intent is defined by
    spanning several documents -- see :data:`COMPOSITE_INTENTS`. A simple
    intent's expansion is deliberately *not* included: those documents are
    supporting context, and admitting them unconditionally is what used to
    append a restatement of the subject's name to an already complete answer.
    """
    words = query_words(query)
    if not words:
        return Routing(Intent.UNKNOWN, (), frozenset(), ("empty query",))

    scores, matched, trace = _scored_intents(query)
    intent = _best_intent(scores)
    terms = frozenset(matched.get(intent, set()))
    co_answers = _co_answer_intents(scores, intent)

    # The conservative fallback: a question whose only content words are the
    # subject's own name is asking about the subject. Nothing else in it pointed
    # anywhere, and about.md is where the subject is described.
    #
    # The content-word test is what keeps this from swallowing real questions:
    # "What is Ahmed's salary expectation?" carries "salary" and "expectation",
    # neither of which any concept group knows, so it stays UNKNOWN and is
    # declined instead of being answered with an introduction.
    if intent is Intent.UNKNOWN:
        own_words = [word for word in content_terms(query) if word not in _SUBJECT_ALIASES]
        if not own_words:
            intent = Intent.IDENTITY
            trace.append("fallback:subject-only")

    if intent is Intent.UNKNOWN:
        return Routing(Intent.UNKNOWN, (), frozenset(), tuple(trace))

    co_answer_domains = tuple(INTENT_SHAPE[extra][0] for extra in co_answers)
    domains: list[Domain] = [INTENT_SHAPE[intent][0]]
    for extra in (*INTENT_EXPANSION.get(intent, ()), *co_answer_domains):
        if extra not in domains and len(domains) <= MAX_EXPANSION_DOMAINS:
            domains.append(extra)
    terms = terms | frozenset(word for extra in co_answers for word in matched.get(extra, set()))

    if len(domains) > 1:
        trace.append("expansion:" + ",".join(domain.value for domain in domains))
    return Routing(
        intent,
        tuple(domains),
        terms,
        tuple(trace),
        frozenset(word for intent_matched in matched.values() for word in intent_matched),
        tuple(
            domain
            for domain in (
                *co_answer_domains,
                *(INTENT_EXPANSION.get(intent, ()) if intent in COMPOSITE_INTENTS else ()),
            )
            if domain in domains and domain != domains[0]
        ),
    )


def detect_domain(query: str) -> Domain:
    """Strict single-domain routing, used to label document *headings*.

    This is the old scoring rule -- the count of distinct query words matching a
    domain's vocabulary -- kept because its job is different: labelling the
    heading of a retrieved chunk so that a cross-cutting FAQ section cannot leak
    into an unrelated domain. There, an over-eager match is the risk, so this
    deliberately does not use frames, expansion or the identity fallback.
    """
    vocabulary: dict[Domain, set[str]] = {domain: set() for domain in DOMAIN_PRIORITY}
    for group in CONCEPT_GROUPS:
        for domain in group.domains:
            if domain in vocabulary:
                vocabulary[domain].update(group.surfaces)
    vocabulary[Domain.PROJECTS].update({"builds", "built", "make", "made"})

    words = query_words(query)
    if not words:
        return Domain.UNKNOWN

    best: Domain = Domain.UNKNOWN
    best_score = 0
    for domain in DOMAIN_PRIORITY:
        score = 0
        for word in words:
            if _matched_surfaces(word, frozenset(vocabulary[domain])):
                score += 1
        if score > best_score:
            best, best_score = domain, score
    return best


def routing_terms(query: str, domain: Domain) -> frozenset[str]:
    """Query words that matched ``domain``'s routing vocabulary.

    These are the system's own name for a subject matter, derived from what the
    documents are *about* rather than from the words they happen to contain. A
    concept that matched here is therefore an accepted paraphrase, and the answer
    layer must not go on to demand that the corpus use the same word: "fun" is
    how the question asks about interests, and requiring interests.md to contain
    the string "fun" is requiring a coincidence.

    Args:
        query: The user's question.
        domain: The domain to explain. :attr:`Domain.UNKNOWN` yields an empty set.

    Returns:
        The distinct query words present in that domain's vocabulary.
    """
    vocabulary: set[str] = set()
    for group in CONCEPT_GROUPS:
        if domain in group.domains:
            vocabulary.update(group.surfaces)
    words = query_words(query)
    return frozenset(word for word in words if _matched_surfaces(word, frozenset(vocabulary)))


def overview_framing_terms(query: str, intent: Intent | None = None) -> frozenset[str]:
    """The query words that ask for a description instead of naming a subject.

    "What is Ahmed's CGPA?" is a lookup and must quote the recorded value.
    "Summarise ResumeForge" is an overview and should quote the document lead. The
    difference is not only *whether* a document may be quoted wholesale -- it is
    also that these words name no subject matter at all. "summarise", "overview",
    "walk" and "history" say how the answer should look; they do not say what it
    is about, so a routed document cannot be expected to contain them. The answer
    layer uses this to keep asking for such a word from being read as a demand
    that education.md contain the string "academic history".

    ``history`` is a framing term only for education and employment. Treating it
    as a global overview word would make unrelated questions such as a project
    history or certification history consume an entire document.
    """
    words = query_words(query)
    framing = words & OVERVIEW_TERMS
    if "history" in words and intent in {Intent.EDUCATION, Intent.EXPERIENCE}:
        framing |= {"history"}
    # "Tell me about X" is an overview request even without the word "summary".
    if "tell" in words and "about" in words:
        framing |= {"tell", "about"}
    return frozenset(framing)


def is_overview_query(query: str, intent: Intent | None = None) -> bool:
    """True when the question asks for a description rather than a lookup.

    See :func:`overview_framing_terms`, which states the rule and names the words
    involved.
    """
    return bool(overview_framing_terms(query, intent))


def is_overview_intent(intent: Intent) -> bool:
    """True when ``intent`` wants a description of a document rather than a lookup.

    Two exclusions, both for the same reason: the intent names a *document*, not
    a request for its opening.

    * :attr:`Intent.PROJECT_DETAIL`. "Which technologies did ResumeForge use?" is a
      lookup *into* ResumeForge's document and must reach the stack, not the
      summary. Overview wording ("tell me about ResumeForge", "summarise
      ResumeForge") still reaches the lead through
      :func:`~app.services.answering.domains.is_overview_query`.
    * :attr:`Intent.IDENTITY`. "Who is Ahmed?" and "What is Ahmed's nickname?"
      are both identity questions, but only the first wants the document lead.
      The second wants the sentence that records a nickname, or the sentence that
      records that there is none -- forcing the lead answered it with his full
      name.

    :attr:`Intent.BACKGROUND` and :attr:`Intent.FOCUS` are composite by
    construction: they span documents, and the opening of each is what makes them
    answerable at all.
    """
    return intent in {Intent.BACKGROUND, Intent.FOCUS}


__all__ = [
    "CO_ANSWER_EXCLUDED_INTENTS",
    "CROSS_CUTTING_CATEGORY",
    "DOMAIN_CATEGORIES",
    "DOMAIN_PRIORITY",
    "MAX_EXPANSION_DOMAINS",
    "QUESTION_FORMS",
    "SUBJECT_REFERENCES",
    "WHOLE_DOCUMENT_INTENTS",
    "AnswerShape",
    "ConceptGroup",
    "Domain",
    "Frame",
    "Intent",
    "Routing",
    "detect_domain",
    "is_overview_intent",
    "is_overview_query",
    "normalised_query",
    "overview_framing_terms",
    "query_words",
    "route",
    "routing_terms",
]
