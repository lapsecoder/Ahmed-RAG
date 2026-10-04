"""Markdown chunker: heading parsing, section metadata and genuine overlap."""

from __future__ import annotations

import itertools
import re

import pytest
from app.core.exceptions import ChunkingError
from app.services.chunker import (
    chunk_markdown,
    make_chunk_id,
    parse_sections,
)

SIMPLE = """# Profile

Intro line one.

Intro line two.

## Skills

Python, FastAPI and FAISS.

### Projects

Local RAG backend.

## Education

BSc.
"""


def unique_paragraph(count: int) -> str:
    """Build non-repeating sentences so shared text cannot come from duplication."""
    sentences = [
        f"Statement {index} describes a distinct engineering topic number {index}."
        for index in range(count)
    ]
    return " ".join(sentences)


def rebuild(chunks: list) -> str:
    """Reassemble the source text from overlapping chunks, dropping duplicates."""
    if not chunks:
        return ""
    text = chunks[0].text
    for previous, following in itertools.pairwise(chunks):
        text += following.text[longest_shared_edge(previous.text, following.text) :]
    return text


def longest_shared_edge(previous: str, following: str) -> int:
    """Length of the longest suffix of ``previous`` that prefixes ``following``."""
    best = 0
    limit = min(len(previous), len(following))
    for size in range(1, limit + 1):
        if previous[-size:] == following[:size]:
            best = size
    return best


# --------------------------------------------------------------------------- #
# Heading parsing
# --------------------------------------------------------------------------- #


def test_parses_all_three_heading_levels() -> None:
    sections = parse_sections(SIMPLE)
    assert [section.title for section in sections] == [
        "Profile",
        "Skills",
        "Projects",
        "Education",
    ]
    assert [section.level for section in sections] == [1, 2, 3, 2]


def test_heading_path_becomes_section_metadata() -> None:
    sections = {section.title: section for section in parse_sections(SIMPLE)}
    assert sections["Profile"].heading_path == ("Profile",)
    assert sections["Skills"].heading_path == ("Profile", "Skills")
    assert sections["Projects"].heading_path == ("Profile", "Skills", "Projects")
    assert sections["Projects"].name == "Profile > Skills > Projects"
    assert sections["Education"].name == "Profile > Education"


def test_h4_and_deeper_are_not_headings() -> None:
    markdown = "# Title\n\nBody.\n\n#### Too deep\n\nStill body.\n"
    sections = parse_sections(markdown)
    assert len(sections) == 1
    assert "Too deep" in sections[0].body


def test_headings_inside_code_fences_are_not_parsed() -> None:
    markdown = "# Real\n\n```python\n# not a heading\n## also not a heading\n```\n\nTail.\n"
    sections = parse_sections(markdown)
    assert len(sections) == 1
    assert sections[0].title == "Real"
    assert "not a heading" in sections[0].body


def test_sections_without_body_content_are_dropped() -> None:
    markdown = "# One\n\n## Two\n\n## Three\n\nReal content here.\n"
    sections = parse_sections(markdown)
    assert [section.title for section in sections] == ["Three"]


def test_content_before_any_heading_uses_document_section() -> None:
    chunks = chunk_markdown("Loose text with no headings at all.", "loose.md")
    assert len(chunks) == 1
    assert chunks[0].section == "Document"
    assert chunks[0].heading_level == 0


def test_section_line_and_character_offsets_are_tracked() -> None:
    sections = parse_sections(SIMPLE)
    education = sections[-1]
    assert education.start_line == 15
    assert SIMPLE.splitlines()[education.start_line - 1] == "## Education"


def test_parse_sections_rejects_non_string_input() -> None:
    with pytest.raises(ChunkingError):
        parse_sections(b"# bytes are not markdown")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Chunk metadata
# --------------------------------------------------------------------------- #


def test_every_chunk_carries_the_required_metadata() -> None:
    chunks = chunk_markdown(SIMPLE, "profile.md")
    assert chunks
    for chunk in chunks:
        assert chunk.source_file == "profile.md"
        assert chunk.section
        assert chunk.chunk_id
        assert chunk.text.strip()


def test_chunk_ids_are_unique_and_deterministic() -> None:
    first = chunk_markdown(SIMPLE, "profile.md")
    second = chunk_markdown(SIMPLE, "profile.md")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_chunk_id_changes_with_source_file() -> None:
    left = make_chunk_id("a.md", "S", 0, "text")
    right = make_chunk_id("b.md", "S", 0, "text")
    assert left != right


def test_chunks_never_span_sections() -> None:
    long_tail = "word " * 200
    markdown = f"# One\n\n{long_tail}\n\n# Two\n\n{long_tail}\n"
    chunks = chunk_markdown(markdown, "doc.md", max_chars=300, overlap_chars=50)
    sections = {chunk.section for chunk in chunks}
    assert sections == {"One", "Two"}


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #


def test_consecutive_chunks_share_genuine_overlap() -> None:
    paragraph = (
        "Ahmed builds retrieval augmented generation systems that run entirely on local "
        "hardware without any hosted model or paid API. "
    ) * 8
    markdown = f"# Projects\n\n{paragraph}\n"
    chunks = chunk_markdown(markdown, "doc.md", max_chars=400, overlap_chars=120)
    assert len(chunks) > 1, "test needs a document that produces several chunks"

    for previous, following in itertools.pairwise(chunks):
        shared = longest_shared_edge(previous.text, following.text)
        assert shared > 0, "consecutive chunks must share text"
        # The shared text is a real prefix of the next chunk and a real suffix
        # of the previous one, not a single incidental character.
        assert shared >= 40
        assert following.text.startswith(previous.text[-shared:])


def test_overlap_does_not_cross_section_boundaries() -> None:
    body = "Distinct sentence about the first section. " * 20
    markdown = f"# First\n\n{body}\n\n# Second\n\nCompletely different content here.\n"
    chunks = chunk_markdown(markdown, "doc.md", max_chars=300, overlap_chars=100)
    for previous, following in itertools.pairwise(chunks):
        if previous.section != following.section:
            assert longest_shared_edge(previous.text, following.text) == 0


def test_overlap_can_be_disabled() -> None:
    markdown = "# Projects\n\n" + unique_paragraph(20)
    chunks = chunk_markdown(markdown, "doc.md", max_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    for previous, following in itertools.pairwise(chunks):
        assert longest_shared_edge(previous.text, following.text) == 0


def test_overlap_starts_on_a_clean_boundary() -> None:
    markdown = "# Projects\n\n" + unique_paragraph(24)
    chunks = chunk_markdown(markdown, "doc.md", max_chars=400, overlap_chars=150)
    overlaps = [
        longest_shared_edge(previous.text, following.text)
        for previous, following in itertools.pairwise(chunks)
    ]
    assert overlaps and all(size > 0 for size in overlaps)
    for previous, following in itertools.pairwise(chunks):
        shared = longest_shared_edge(previous.text, following.text)
        head = following.text[:shared]
        # A clean boundary never begins in the middle of a word.
        assert head == head.strip()
        assert not head or head.split()[0].isalpha()


# --------------------------------------------------------------------------- #
# Size behaviour
# --------------------------------------------------------------------------- #


def test_chunks_respect_the_size_limit() -> None:
    markdown = "# Notes\n\n" + "\n\n".join(f"Paragraph {index}. " * 8 for index in range(20))
    chunks = chunk_markdown(markdown, "doc.md", max_chars=300, overlap_chars=60)
    assert len(chunks) > 5
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_chunks_reassemble_into_the_original_text() -> None:
    paragraph = unique_paragraph(30)
    markdown = f"# Notes\n\n{paragraph}\n"
    chunks = chunk_markdown(markdown, "doc.md", max_chars=250, overlap_chars=60)
    assert len(chunks) > 1
    assert re.sub(r"\s+", " ", rebuild(chunks)).strip() == re.sub(r"\s+", " ", markdown).strip()


def test_oversized_paragraph_is_split_on_sentence_boundaries() -> None:
    single = "This is one long sentence. " * 60
    markdown = f"# Notes\n\n{single}\n"
    chunks = chunk_markdown(markdown, "doc.md", max_chars=250, overlap_chars=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 250
    assert rebuild(chunks).count("sentence") == 60


def test_code_blocks_are_kept_intact_within_a_chunk() -> None:
    code = "\n".join(f"line_{index} = {index}" for index in range(20))
    markdown = f"# Notes\n\n```python\n{code}\n```\n"
    chunks = chunk_markdown(markdown, "doc.md", max_chars=4000, overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text.count("```") == 2
    assert "line_19 = 19" in chunks[0].text


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("markdown", ["", "   \n\n  \n", "# Only a heading\n"])
def test_empty_or_heading_only_documents_yield_no_chunks(markdown: str) -> None:
    assert chunk_markdown(markdown, "empty.md") == []


def test_invalid_chunk_parameters_are_rejected() -> None:
    with pytest.raises(ChunkingError):
        chunk_markdown("body", "f.md", max_chars=0)
    with pytest.raises(ChunkingError):
        chunk_markdown("body", "f.md", overlap_chars=-1)
    with pytest.raises(ChunkingError):
        chunk_markdown("body", "f.md", max_chars=100, overlap_chars=100)
    with pytest.raises(ChunkingError):
        chunk_markdown("body", "")


def test_char_offsets_point_at_the_chunk_text() -> None:
    markdown = "# Alpha\n\nThe first body paragraph.\n\n## Beta\n\nThe second paragraph.\n"
    chunks = chunk_markdown(markdown, "doc.md")
    for chunk in chunks:
        assert chunk.char_start >= 0
        assert chunk.char_end >= chunk.char_start
        assert markdown[chunk.char_start : chunk.char_start + len(chunk.text.strip())].strip()
