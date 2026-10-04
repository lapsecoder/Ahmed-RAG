"""The corpus map: which document backs which domain, and what projects exist.

Both facts are read from the knowledge base itself rather than hardcoded. Each
Markdown file declares a ``category:`` in its frontmatter, and project documents
declare ``category: project``. So adding ``knowledge_base/projects/newthing.md``
with the usual frontmatter is enough to make questions about it route and answer,
with no code change.

That matters for the "do not hardcode project answers" requirement: nothing in
this module knows what ResumeForge or MovieMind *say*, only which document they
live in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger
from app.services.answering.domains import (
    CROSS_CUTTING_CATEGORY,
    DOMAIN_CATEGORIES,
    Domain,
)
from app.services.knowledge_base import is_excluded, source_paths
from app.services.text import normalise_key

logger = get_logger("services.answering.corpus")

_CATEGORY_RE: re.Pattern[str] = re.compile(
    r"^category:\s*(?P<value>[A-Za-z0-9_-]+)\s*$", re.MULTILINE
)
_FRONTMATTER_RE: re.Pattern[str] = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
_H1_RE: re.Pattern[str] = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class CorpusProfile:
    """Which indexed files belong to which domain, and the project's names."""

    #: ``source_file`` (knowledge-base relative, e.g. ``projects/moviemind.md``)
    #: -> frontmatter category.
    categories: dict[str, str] = field(default_factory=dict)
    #: Normalised project name -> display name, e.g. ``"resumeforge"``.
    project_names: dict[str, str] = field(default_factory=dict)
    #: Display name -> ``source_file`` for each project document.
    project_files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_directory(
        cls, kb_dir: Path | str, exclude_globs: tuple[str, ...] = ()
    ) -> CorpusProfile:
        """Build the profile by reading the frontmatter of every indexed file.

        Never raises: a file that cannot be read or parsed simply does not
        contribute a category, which at worst narrows routing for that document.
        """
        categories: dict[str, str] = {}
        project_names: dict[str, str] = {}
        project_files: dict[str, str] = {}

        for relative in source_paths(kb_dir, list(exclude_globs)):
            path = Path(kb_dir) / relative
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.warning("could not read %s; excluding it from routing", relative)
                continue

            category_match = _CATEGORY_RE.search(_frontmatter_body(text))
            category = category_match.group("value").strip().lower() if category_match else ""
            if not category:
                logger.warning("%s has no frontmatter category; excluding from routing", relative)
                continue
            categories[relative] = category

            if category == "project":
                display = _display_name(text, relative)
                if not display:
                    continue
                key = normalise_key(display)
                if not key:
                    continue
                project_names.setdefault(key, display)
                project_files.setdefault(display, relative)

        logger.info(
            "corpus profile: %d categorised document(s), %d project(s)",
            len(categories),
            len(project_names),
        )
        return cls(
            categories=categories,
            project_names=project_names,
            project_files=project_files,
        )

    @property
    def categorised_files(self) -> frozenset[str]:
        """Every ``source_file`` that carries a usable category."""
        return frozenset(self.categories)

    def category(self, source_file: str) -> str | None:
        """Frontmatter category for ``source_file``, or ``None`` when unknown."""
        return self.categories.get(source_file)

    def is_categorised(self, source_file: str) -> bool:
        """True when ``source_file`` may be used as evidence."""
        return source_file in self.categories

    def files_for(self, domain: Domain) -> tuple[str, ...]:
        """Documents that may back an answer to a ``domain`` question.

        The domain's own documents come first, followed by the cross-cutting FAQ.
        An empty tuple means the domain has no documents and nothing can be
        answered for it.
        """
        wanted = DOMAIN_CATEGORIES.get(domain, ())
        primary = tuple(
            sorted(source for source, category in self.categories.items() if category in wanted)
        )
        if domain is Domain.UNKNOWN:
            return ()
        cross_cutting = tuple(
            sorted(
                source
                for source, category in self.categories.items()
                if category == CROSS_CUTTING_CATEGORY and source not in primary
            )
        )
        return primary + cross_cutting

    def files_for_domains(self, domains: Sequence[Domain]) -> tuple[str, ...]:
        """Documents permitted to answer a question spanning several domains.

        Each domain's own documents come first, in the order given, and the
        cross-cutting FAQ is appended once at the end rather than once per
        domain. Order matters: the composer prefers the primary document, so the
        caller controls which document leads by passing the primary domain first.

        Duplicates are removed, which is what keeps an expansion that re-names the
        primary domain -- "focus" spans goals and identity -- from listing one
        file twice.
        """
        ordered: list[str] = []
        for domain in domains:
            for source in self.files_for(domain):
                if source not in ordered:
                    ordered.append(source)
        return tuple(ordered)

    def domain_of(self, primary_files: frozenset[str]) -> Domain:
        """The domain that owns ``primary_files``, or :attr:`Domain.UNKNOWN`.

        The inverse of :meth:`primary_files_for`, for the rare caller that holds a
        document set rather than a domain.
        """
        for domain in DOMAIN_CATEGORIES:
            if domain is Domain.UNKNOWN:
                continue
            if frozenset(self.primary_files_for(domain)) == primary_files:
                return domain
        return Domain.UNKNOWN

    def primary_files_for(self, domain: Domain) -> tuple[str, ...]:
        """Documents for ``domain`` excluding the cross-cutting FAQ."""
        wanted = DOMAIN_CATEGORIES.get(domain, ())
        return tuple(
            sorted(source for source, category in self.categories.items() if category in wanted)
        )


def _frontmatter_body(text: str) -> str:
    """Return the frontmatter block of ``text``, or the whole text if absent."""
    match = _FRONTMATTER_RE.match(text)
    return match.group("body") if match else text


def _display_name(text: str, relative: str) -> str:
    """Best display name for a project document: H1 title, else file stem."""
    match = _H1_RE.search(_frontmatter_body(text) + "\n" + text)
    if match:
        title = match.group("title").strip()
        if title:
            return title
    stem = relative.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("-", " ").replace("_", " ").strip()


def build_profile(kb_dir: Path | str, exclude_globs: tuple[str, ...] = ()) -> CorpusProfile:
    """Convenience wrapper matching the container's call style."""
    return CorpusProfile.from_directory(kb_dir, exclude_globs)


__all__ = [
    "CorpusProfile",
    "build_profile",
    "is_excluded",
]
