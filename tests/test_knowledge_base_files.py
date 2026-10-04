"""Markdown file handling: BOM, frontmatter, offsets, paths and exclude globs.

Everything here runs against temporary directories with synthetic Markdown.
Ahmed's real knowledge base is never read by the test suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.services.chunker import chunk_file, chunk_markdown, find_frontmatter
from app.services.knowledge_base import (
    DEFAULT_EXCLUDE_GLOBS,
    corpus_fingerprint,
    is_excluded,
    list_markdown_files,
    load_chunks,
    source_paths,
)

BOM = "\ufeff"


def write(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="")
    return path


@pytest.fixture
def knowledge_base(tmp_path: Path) -> Path:
    """A knowledge base with nested folders, a dot folder and a draft."""
    kb = tmp_path / "kb"
    write(kb / "profile.md", "# Profile\n\nAhmed builds local RAG systems.\n")
    write(kb / "projects" / "ahmed-rag.md", "# Ahmed RAG\n\nA FAISS backed service.\n")
    write(kb / "archive" / "profile.md", "# Archived profile\n\nOlder facts.\n")
    write(kb / "drafts" / "unfinished.md", "# Draft\n\nNot ready.\n")
    write(kb / ".hidden" / "secret.md", "# Secret\n\nHidden draft.\n")
    write(kb / ".scratch.md", "# Scratch\n\nDot file.\n")
    write(kb / "notes.txt", "not markdown")
    return kb


# --------------------------------------------------------------------------- #
# 1. UTF-8 BOM
# --------------------------------------------------------------------------- #


def test_a_bom_does_not_hide_the_first_heading(tmp_path: Path) -> None:
    path = write(tmp_path / "bom.md", "# Projects\n\nAhmed RAG is local.\n", encoding="utf-8-sig")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf"), "fixture must really carry a BOM"

    chunks = chunk_file(path, source_name="bom.md")

    assert [chunk.section for chunk in chunks] == ["Projects"]
    assert chunks[0].heading_path == ("Projects",)
    assert chunks[0].heading_level == 1


def test_a_bom_preserves_the_full_heading_hierarchy() -> None:
    with_bom = chunk_markdown(f"{BOM}# Projects\n\n## Ahmed RAG\n\nLocal service.\n", "doc.md")
    without = chunk_markdown("# Projects\n\n## Ahmed RAG\n\nLocal service.\n", "doc.md")
    assert with_bom[0].section == "Projects > Ahmed RAG"
    assert [c.section for c in with_bom] == [c.section for c in without]


def test_a_bom_is_never_embedded_in_the_chunk_text() -> None:
    chunks = chunk_markdown(f"{BOM}# Profile\n\nAhmed builds RAG.\n", "doc.md")
    assert BOM not in chunks[0].text


def test_bom_and_bomless_files_produce_identical_chunks(tmp_path: Path) -> None:
    body = "# Profile\n\nAhmed builds local RAG systems.\n\n## Skills\n\nPython.\n"
    plain = write(tmp_path / "plain.md", body)
    bommed = write(tmp_path / "bommed.md", body, encoding="utf-8-sig")

    from_plain = chunk_file(plain, source_name="doc.md")
    from_bommed = chunk_file(bommed, source_name="doc.md")

    assert [(c.chunk_id, c.text, c.section) for c in from_bommed] == [
        (c.chunk_id, c.text, c.section) for c in from_plain
    ]


def test_bom_offsets_refer_to_the_decoded_text() -> None:
    """A BOM is a byte-order mark, not text, so it is not part of the offsets."""
    body = f"{BOM}# Projects\n\nAhmed RAG is local.\n"
    chunk = chunk_markdown(body, "doc.md")[0]
    decoded = body.lstrip(BOM)
    assert decoded[chunk.char_start : chunk.char_end] == "# Projects\n\nAhmed RAG is local."
    assert chunk.start_line == 1
    # The same holds for a file read from disk with utf-8-sig.
    assert chunk.char_start == 0


# --------------------------------------------------------------------------- #
# 2. Frontmatter
# --------------------------------------------------------------------------- #


FRONTMATTER_DOC = """---
title: Ahmed
tags: [python, rag]
---

# Profile

Ahmed builds local RAG systems.
"""


def test_frontmatter_is_excluded_from_retrievable_content() -> None:
    chunks = chunk_markdown(FRONTMATTER_DOC, "profile.md")
    joined = " ".join(chunk.text for chunk in chunks)
    assert "title: Ahmed" not in joined
    assert "tags: [python, rag]" not in joined
    assert "---" not in joined
    assert [chunk.section for chunk in chunks] == ["Profile"]
    assert "Ahmed builds local RAG systems." in joined


def test_offsets_after_frontmatter_still_point_at_the_original_file() -> None:
    chunk = chunk_markdown(FRONTMATTER_DOC, "profile.md")[0]
    assert chunk.char_start == FRONTMATTER_DOC.index("# Profile")
    assert FRONTMATTER_DOC[chunk.char_start : chunk.char_end] == chunk.text
    assert chunk.start_line == 6
    assert FRONTMATTER_DOC.splitlines()[chunk.start_line - 1] == "# Profile"


def test_a_frontmatter_only_document_yields_no_chunks() -> None:
    assert chunk_markdown("---\ntitle: Ahmed\n---\n", "profile.md") == []


def test_an_unterminated_frontmatter_block_is_treated_as_content() -> None:
    document = "---\ntitle: Ahmed\n\n# Profile\n\nBody.\n"
    assert find_frontmatter(document) is None
    assert chunk_markdown(document, "profile.md")


def test_a_horizontal_rule_is_not_frontmatter() -> None:
    document = "# Profile\n\nBody text.\n\n---\n\nMore text.\n"
    assert find_frontmatter(document) is None
    # The rule is ordinary body content: it does not swallow the document and it
    # is still indexed, because a rule can be meaningful in a real document.
    chunks = chunk_markdown(document, "profile.md")
    assert [chunk.section for chunk in chunks] == ["Profile"]
    assert "---" in chunks[0].text
    assert "More text." in chunks[0].text


def test_frontmatter_detection_reports_line_numbers() -> None:
    assert find_frontmatter(FRONTMATTER_DOC) == (1, 4)
    assert find_frontmatter("\n\n---\na: 1\n---\n\n# T\n\nx\n") == (3, 5)
    assert find_frontmatter("# No frontmatter\n\ntext\n") is None


def test_frontmatter_is_excluded_for_files_read_from_disk(tmp_path: Path) -> None:
    path = write(tmp_path / "profile.md", FRONTMATTER_DOC)
    chunks = chunk_file(path, source_name="profile.md")
    assert all("tags:" not in chunk.text for chunk in chunks)


# --------------------------------------------------------------------------- #
# 3. Offsets refer to the original file
# --------------------------------------------------------------------------- #


DOC = """# Alpha

First paragraph of alpha.

## Beta

Second paragraph of beta.

### Gamma

Third paragraph of gamma.
"""


def test_char_offsets_locate_the_chunk_in_the_original_file() -> None:
    chunks = chunk_markdown(DOC, "doc.md")
    assert len(chunks) == 3
    for chunk in chunks:
        assert DOC[chunk.char_start : chunk.char_end] == chunk.text


def test_char_offsets_of_a_later_section_are_absolute() -> None:
    gamma = next(
        chunk for chunk in chunk_markdown(DOC, "doc.md") if chunk.section.endswith("Gamma")
    )
    assert gamma.char_start == DOC.index("### Gamma")
    assert gamma.char_start > 0, "a later section must not restart at offset 0"


def test_line_offsets_locate_the_chunk_in_the_original_file() -> None:
    lines = DOC.splitlines()
    for chunk in chunk_markdown(DOC, "doc.md"):
        first = lines[chunk.start_line - 1]
        last = lines[chunk.end_line - 1]
        assert first.strip() in chunk.text
        assert last.strip() in chunk.text


def test_line_offsets_are_per_chunk_not_per_section() -> None:
    long_body = "\n\n".join(f"Paragraph {index} of the same section." for index in range(12))
    document = f"# Section\n\n{long_body}\n"
    chunks = chunk_markdown(document, "doc.md", max_chars=160, overlap_chars=40)
    assert len(chunks) > 1
    starts = [chunk.start_line for chunk in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) > 1, "chunks of one section must not share a single line range"


def test_offsets_survive_a_long_multi_chunk_section() -> None:
    paragraph = "Ahmed builds local retrieval systems with FAISS and Ollama. " * 12
    document = f"# Projects\n\n{paragraph}\n"
    chunks = chunk_markdown(document, "doc.md", max_chars=400, overlap_chars=120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 <= chunk.char_start < chunk.char_end <= len(document)
        assert document[chunk.char_start] != "\n"
        assert document[chunk.char_end - 1] != "\n"
    # The first chunk owns the heading; later chunks start inside the paragraph.
    assert chunks[0].start_line == 1
    assert chunks[0].char_start == 0
    assert all(chunk.start_line == 3 for chunk in chunks[1:])
    assert all(chunk.char_start > 0 for chunk in chunks[1:])


# --------------------------------------------------------------------------- #
# 4. Source paths
# --------------------------------------------------------------------------- #


def test_nested_files_with_the_same_name_stay_distinguishable(knowledge_base: Path) -> None:
    chunks = load_chunks(knowledge_base, max_chars=400, overlap_chars=60)
    sources = {chunk.source_file for chunk in chunks}
    assert "profile.md" in sources
    assert "archive/profile.md" in sources
    assert len([source for source in sources if source.endswith("profile.md")]) == 2

    by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_file, []).append(chunk.chunk_id)
    top = next(chunk for chunk in chunks if chunk.source_file == "profile.md")
    archived = next(chunk for chunk in chunks if chunk.source_file == "archive/profile.md")
    assert top.chunk_id != archived.chunk_id
    assert top.citation != archived.citation


def test_source_paths_use_posix_separators_on_every_platform(knowledge_base: Path) -> None:
    sources = source_paths(knowledge_base)
    assert "projects/ahmed-rag.md" in sources
    assert all("\\" not in source for source in sources)


def test_chunk_ids_differ_for_identically_named_files_in_one_knowledge_base(
    tmp_path: Path,
) -> None:
    kb = tmp_path / "kb"
    write(kb / "a" / "profile.md", "# A\n\nAlpha content.\n")
    write(kb / "b" / "profile.md", "# A\n\nAlpha content.\n")
    chunks = load_chunks(kb)
    by_source: dict[str, set[str]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_file, set()).add(chunk.chunk_id)
    assert sorted(by_source) == ["a/profile.md", "b/profile.md"]
    assert not by_source["a/profile.md"] & by_source["b/profile.md"]


def test_discovery_is_sorted_by_relative_path(knowledge_base: Path) -> None:
    sources = source_paths(knowledge_base)
    assert sources == sorted(sources)


# --------------------------------------------------------------------------- #
# 5. Exclude globs
# --------------------------------------------------------------------------- #


def test_legitimate_nested_markdown_still_loads(knowledge_base: Path) -> None:
    sources = source_paths(knowledge_base)
    assert "profile.md" in sources
    assert "projects/ahmed-rag.md" in sources
    assert "archive/profile.md" in sources


def test_non_markdown_files_are_never_indexed(knowledge_base: Path) -> None:
    assert "notes.txt" not in source_paths(knowledge_base)


def test_dot_prefixed_files_and_directories_are_excluded(knowledge_base: Path) -> None:
    sources = source_paths(knowledge_base)
    assert not [source for source in sources if Path(source).name.startswith(".")]
    assert not [source for source in sources if ".hidden" in source]


def test_draft_and_export_directories_are_excluded_by_default(knowledge_base: Path) -> None:
    sources = source_paths(knowledge_base)
    assert "drafts/unfinished.md" not in sources
    assert not [source for source in sources if source.startswith("exports/")]


def test_exclude_globs_are_configurable(knowledge_base: Path) -> None:
    sources = source_paths(knowledge_base, ("drafts/**", "archive/**"))
    assert "drafts/unfinished.md" not in sources
    assert "archive/profile.md" not in sources
    assert "profile.md" in sources


def test_exclude_globs_can_be_disabled(knowledge_base: Path) -> None:
    sources = source_paths(knowledge_base, ())
    assert "drafts/unfinished.md" in sources
    assert ".hidden/secret.md" in sources
    assert ".scratch.md" in sources


def test_exclude_globs_are_configurable_through_settings(tmp_path: Path) -> None:
    from app.config import Settings

    kb = tmp_path / "kb"
    write(kb / "keep.md", "# Keep\n\nBody.\n")
    write(kb / "internal" / "secret.md", "# Secret\n\nHidden.\n")

    defaults = Settings(kb_dir=kb)
    assert source_paths(kb, defaults.kb_exclude_globs) == ["internal/secret.md", "keep.md"]

    custom = Settings(kb_dir=kb, kb_exclude_globs="internal")
    assert source_paths(kb, custom.kb_exclude_globs) == ["keep.md"]

    disabled = Settings(kb_dir=kb, kb_exclude_globs="")
    assert disabled.kb_exclude_globs == ()
    assert source_paths(kb, disabled.kb_exclude_globs) == ["internal/secret.md", "keep.md"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (".hidden/file.md", True),
        ("projects/.draft.md", True),
        ("drafts/a.md", True),
        ("notes/drafts/a.md", True),
        ("exports/report.md", True),
        ("deep/nested/exports/report.md", True),
        ("draft.tmp.md", True),
        ("backup.md~", True),
        ("projects/ahmed-rag.md", False),
        ("draft.md", False),
        ("drafts.md", False),
        ("archive/profile.md", False),
    ],
)
def test_is_excluded_matches_the_documented_defaults(path: str, expected: bool) -> None:
    assert is_excluded(path, DEFAULT_EXCLUDE_GLOBS) is expected


# --------------------------------------------------------------------------- #
# 6. Corpus fingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_is_stable_for_an_unchanged_corpus(knowledge_base: Path) -> None:
    assert corpus_fingerprint(knowledge_base) == corpus_fingerprint(knowledge_base)


def test_fingerprint_changes_when_content_changes(knowledge_base: Path) -> None:
    before = corpus_fingerprint(knowledge_base)
    write(knowledge_base / "profile.md", "# Profile\n\nAhmed builds local RAG systems too.\n")
    assert corpus_fingerprint(knowledge_base) != before


def test_fingerprint_changes_when_a_file_is_added_or_removed(knowledge_base: Path) -> None:
    before = corpus_fingerprint(knowledge_base)
    write(knowledge_base / "contact.md", "# Contact\n\nEmail address.\n")
    after_add = corpus_fingerprint(knowledge_base)
    assert after_add != before

    (knowledge_base / "contact.md").unlink()
    assert corpus_fingerprint(knowledge_base) == before


def test_fingerprint_changes_when_a_file_is_renamed(knowledge_base: Path) -> None:
    before = corpus_fingerprint(knowledge_base)
    (knowledge_base / "profile.md").rename(knowledge_base / "bio.md")
    assert corpus_fingerprint(knowledge_base) != before


def test_fingerprint_ignores_modification_times(knowledge_base: Path) -> None:
    """Staleness must be content driven, never mtime driven."""
    before = corpus_fingerprint(knowledge_base)
    target = knowledge_base / "profile.md"
    stat = target.stat()
    os.utime(target, (stat.st_atime + 9999, stat.st_mtime + 9999))
    assert corpus_fingerprint(knowledge_base) == before


def test_fingerprint_ignores_excluded_files(knowledge_base: Path) -> None:
    before = corpus_fingerprint(knowledge_base)
    write(knowledge_base / "drafts" / "another.md", "# Draft\n\nMore drafts.\n")
    assert corpus_fingerprint(knowledge_base) == before


def test_fingerprint_of_a_missing_knowledge_base_is_the_empty_digest(tmp_path: Path) -> None:
    import hashlib

    assert corpus_fingerprint(tmp_path / "absent") == hashlib.sha256().hexdigest()


def test_fingerprint_is_deterministic_across_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root, order in ((first, ["b.md", "a/c.md", "a.md"]), (second, ["a.md", "b.md", "a/c.md"])):
        for name in order:
            write(root / name, f"# {name}\n\nBody for {name}.\n")
    assert corpus_fingerprint(first) == corpus_fingerprint(second)


def test_list_markdown_files_returns_absolute_paths(knowledge_base: Path) -> None:
    for path in list_markdown_files(knowledge_base):
        assert path.is_absolute()
        assert path.suffix == ".md"
