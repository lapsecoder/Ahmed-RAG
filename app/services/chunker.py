"""Markdown chunker.

Guarantees:

* ``#`` / ``##`` / ``###`` headings are parsed into a real heading hierarchy and
  the full path becomes the chunk's ``section`` metadata.
* Fenced code blocks are treated as opaque: ``#`` lines inside a fence are not
  headings, and a fence is never split across chunks.
* A leading YAML frontmatter block is excluded from the retrievable content,
  while every offset still refers to the original file.
* A UTF-8 byte-order mark is stripped, so a file saved by a Windows editor keeps
  its heading structure.
* Chunks are paragraph-aligned and never exceed ``max_chars`` unless a single
  paragraph is unavoidably larger.
* Consecutive chunks inside the same section share a genuine textual overlap
  (a clean suffix/prefix of the previous chunk), so no context is lost at a
  boundary.
* ``start_line``/``end_line`` and ``char_start``/``char_end`` are offsets **into
  the original file**, never into a section-local copy of it. They are derived
  from the source text, not estimated.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.core.exceptions import ChunkingError
from app.models.document import DocumentChunk

#: Only the three heading levels required by the spec are treated as structure.
HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,3})\s+(\S.*?)\s*#*\s*$")
FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_SENTENCE_BREAK: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")

#: A ``---``/``...`` fence delimiting a YAML frontmatter block.
FRONTMATTER_DELIMITER_RE: Final[re.Pattern[str]] = re.compile(r"^(-{3,}|\.{3,})\s*$")

#: U+FEFF, emitted at the start of files written by many Windows editors.
BOM: Final[str] = "\ufeff"

DOCUMENT_SECTION: Final[str] = "Document"


@dataclass(frozen=True, slots=True)
class TextUnit:
    """A paragraph-sized slice with exact offsets into the original file."""

    text: str
    start: int
    end: int
    is_tail: bool = False


@dataclass(frozen=True, slots=True)
class _Line:
    """One physical line, located in both the file and its section body."""

    text: str
    start: int
    end: int
    number: int
    body_start: int = 0

    @property
    def blank(self) -> bool:
        """True when the line holds no characters."""
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Section:
    """A heading-delimited region of a Markdown document."""

    title: str
    level: int
    heading_path: tuple[str, ...]
    lines: tuple[_Line, ...]
    start_line: int
    end_line: int
    char_start: int
    char_end: int

    @property
    def name(self) -> str:
        """The section label stored on every chunk of this section."""
        return " > ".join(self.heading_path) if self.heading_path else DOCUMENT_SECTION

    @property
    def body(self) -> str:
        """The section's text, lines re-joined with ``\\n``.

        The trailing newline of each source line is dropped, exactly as it is
        when a chunk's text is assembled, so body offsets and chunk offsets
        describe the same coordinate space.
        """
        return "\n".join(line.text for line in self.lines)

    def file_offset(self, body_offset: int) -> int:
        """Translate a body offset into an offset in the original file.

        Args:
            body_offset: Position inside :attr:`body`.

        Returns:
            The matching character offset in the original document text. Offsets
            at a line boundary resolve to the end of the preceding line's text,
            which is the correct exclusive end for a span.
        """
        for line in self.lines:
            span = len(line.text)
            if line.body_start <= body_offset <= line.body_start + span:
                return line.start + (body_offset - line.body_start)
        return self.char_end

    def line_at(self, file_offset: int) -> int:
        """Return the 1-based file line containing ``file_offset``."""
        for line in self.lines:
            if line.start <= file_offset <= line.end:
                return line.number
        return self.start_line if file_offset < self.lines[0].start else self.end_line


def find_frontmatter(markdown: str) -> tuple[int, int] | None:
    """Locate a leading YAML frontmatter block.

    A frontmatter block starts at the first non-blank line when that line is a
    ``---`` (or ``...``) delimiter, and ends at the next such delimiter.

    Args:
        markdown: Raw document text, with or without a leading BOM.

    Returns:
        ``(first_line, last_line)`` as 1-based inclusive line numbers, or ``None``
        when the document does not start with frontmatter.
    """
    lines = markdown.splitlines()
    first = 0
    while first < len(lines) and not lines[first].strip():
        first += 1
    if first >= len(lines) or FRONTMATTER_DELIMITER_RE.match(lines[first].strip()) is None:
        return None
    for index in range(first + 1, len(lines)):
        if FRONTMATTER_DELIMITER_RE.match(lines[index].strip()) is not None:
            return first + 1, index + 1
    return None


def parse_sections(markdown: str) -> list[Section]:
    """Split ``markdown`` into sections delimited by ``#``/``##``/``###``.

    A leading BOM is stripped and a leading YAML frontmatter block is skipped;
    both are excluded from the retrievable content while every recorded offset
    still refers to the original file.

    Args:
        markdown: Raw document text.

    Returns:
        Sections in document order, each carrying its heading path, line range
        and character range. Sections that contain no substantive content
        (headings only) are dropped.

    Raises:
        ChunkingError: If ``markdown`` is not a string.
    """
    if not isinstance(markdown, str):
        raise ChunkingError(f"expected markdown text, got {type(markdown).__name__}")

    markdown = markdown.lstrip(BOM)
    skipped = find_frontmatter(markdown)
    skipped_lines = set(range(skipped[0], skipped[1] + 1)) if skipped else set()

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[_Line] = []
    offset = 0
    line_number = 0
    in_fence = False
    fence_char = ""

    def flush() -> None:
        section = _build_section(stack, buffer, offset)
        if section is not None:
            sections.append(section)

    for raw in markdown.splitlines(keepends=True):
        line_number += 1
        line = raw.rstrip("\r\n")
        line_end = offset + len(line)
        offset += len(raw)

        if line_number in skipped_lines:
            # Frontmatter stays out of the retrievable text, but the line
            # counter and file offset advance exactly as if it were kept.
            continue

        fence = FENCE_RE.match(line)
        if fence is not None:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, marker
            elif marker == fence_char:
                in_fence, fence_char = False, ""
            buffer.append(_Line(line, line_end - len(line), line_end, line_number))
            continue

        heading = None if in_fence else HEADING_RE.match(line)
        if heading is not None:
            flush()
            buffer.clear()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buffer.append(_Line(line, line_end - len(line), line_end, line_number))
            continue

        buffer.append(_Line(line, line_end - len(line), line_end, line_number))

    flush()
    return sections


def _build_section(
    stack: list[tuple[int, str]], lines: list[_Line], doc_end: int
) -> Section | None:
    if not lines:
        return None
    has_content = any(line.text.strip() and not HEADING_RE.match(line.text) for line in lines)
    if not has_content:
        return None
    # Body offsets are assigned here so a body offset and a file offset can be
    # translated exactly, one line at a time.
    located: list[_Line] = []
    body_start = 0
    for line in lines:
        located.append(
            _Line(
                text=line.text,
                start=line.start,
                end=line.end,
                number=line.number,
                body_start=body_start,
            )
        )
        body_start += len(line.text) + 1
    return Section(
        title=stack[-1][1] if stack else "",
        level=stack[-1][0] if stack else 0,
        heading_path=tuple(title for _, title in stack),
        lines=tuple(located),
        start_line=lines[0].number,
        end_line=lines[-1].number,
        char_start=lines[0].start,
        char_end=min(lines[-1].end, doc_end) if doc_end else lines[-1].end,
    )


def _paragraphs(section: Section) -> list[TextUnit]:
    """Split a section body into paragraph units with original-file offsets."""
    units: list[TextUnit] = []
    buffer: list[_Line] = []
    in_fence = False
    fence_char = ""

    for line in section.lines:
        fence = FENCE_RE.match(line.text)
        if fence is not None:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, marker
            elif marker == fence_char:
                in_fence, fence_char = False, ""
        buffer.append(line)
        if not in_fence and line.blank:
            unit = _unit_from_lines(section, buffer)
            if unit is not None:
                units.append(unit)
            buffer = []

    unit = _unit_from_lines(section, buffer)
    if unit is not None:
        units.append(unit)
    return units


def _unit_from_lines(section: Section, lines: list[_Line]) -> TextUnit | None:
    """Build one paragraph unit, converting body offsets into file offsets."""
    if not lines:
        return None
    text = "\n".join(line.text for line in lines)
    stripped = text.strip()
    if not stripped:
        return None
    lead = len(text) - len(text.lstrip())
    body_start = lines[0].body_start + lead
    start = section.file_offset(body_start)
    end = section.file_offset(body_start + len(stripped))
    return TextUnit(text=stripped, start=start, end=end)


def _split_long_unit(unit: TextUnit, max_chars: int) -> list[TextUnit]:
    """Break a paragraph that cannot fit, preferring sentence then word cuts."""
    text = unit.text
    if len(text) <= max_chars:
        return [unit]
    pieces: list[TextUnit] = []
    cursor = 0
    floor = max(max_chars // 3, 1)
    while cursor < len(text):
        limit = min(cursor + max_chars, len(text))
        cut = limit
        if limit < len(text):
            window = text[cursor:limit]
            sentence_end = 0
            for match in _SENTENCE_BREAK.finditer(window):
                if match.end() > floor:
                    sentence_end = match.end()
            if sentence_end:
                cut = cursor + sentence_end
            else:
                space = window.rfind(" ")
                cut = cursor + (space + 1 if space > floor else limit)
        raw = text[cursor:cut]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            pieces.append(
                TextUnit(
                    text=stripped,
                    start=unit.start + cursor + lead,
                    end=unit.start + cursor + lead + len(stripped),
                )
            )
        cursor = cut
    return pieces


def _overlap_tail(chunk_text: str, chunk_end: int, overlap_chars: int) -> TextUnit | None:
    """Return the trailing text of ``chunk_text`` to prepend to the next chunk."""
    if overlap_chars <= 0 or not chunk_text:
        return None
    window = chunk_text[-overlap_chars:]

    # Prefer starting the overlap on a clean sentence boundary, as long as that
    # still yields a substantial overlap.
    sentence = _SENTENCE_BREAK.search(window)
    tail = window[sentence.start() :].strip() if sentence is not None else ""
    if len(tail) < overlap_chars // 2:
        # Fall back to a word boundary, which keeps the overlap near the
        # requested size instead of shrinking to a few leftover characters.
        boundary = re.search(r"\s", window)
        if boundary is None:
            return None
        tail = window[boundary.end() :].strip()
    if not tail:
        return None
    tail = _trim_unbalanced_fences(tail)
    if not tail:
        return None
    return TextUnit(text=tail, start=chunk_end - len(tail), end=chunk_end, is_tail=True)


def _trim_unbalanced_fences(text: str) -> str:
    """Drop a trailing partial code fence so overlaps never break a code block."""
    for marker in ("```", "~~~"):
        if text.count(marker) % 2 == 1:
            cut = text.find(marker)
            if cut > 0:
                return text[:cut].rstrip()
    return text


@dataclass(slots=True)
class _Cursor:
    """A position inside a unit list, able to stop mid-unit."""

    units: list[TextUnit]
    index: int = 0
    offset: int = 0

    @property
    def done(self) -> bool:
        """True when every character has been consumed."""
        return self.index >= len(self.units)

    def current(self) -> TextUnit:
        """The unit the cursor points at."""
        return self.units[self.index]


def _take(cursor: _Cursor, budget: int) -> tuple[str, int, int]:
    """Consume up to ``budget`` characters, returning ``(text, start, end)``.

    Units are kept whole where possible. A unit that does not fit is consumed
    partially, preferring a word boundary, and the remainder is consumed by the
    next call. Progress is guaranteed: at least one character is always taken.
    """
    if cursor.done or budget <= 0:
        return "", cursor.current().start if not cursor.done else 0, cursor.current().start

    parts: list[str] = []
    total = 0
    start: int | None = None
    end = cursor.current().start

    while not cursor.done:
        unit = cursor.current()
        remaining = len(unit.text) - cursor.offset
        separator = 2 if parts else 0
        room = budget - total - separator

        if room <= 0:
            break

        if remaining <= room:
            piece = unit.text[cursor.offset :]
            if start is None:
                start = unit.start + cursor.offset
            parts.append(piece)
            total += separator + len(piece)
            end = unit.end
            cursor.index += 1
            cursor.offset = 0
            continue

        piece = unit.text[cursor.offset : cursor.offset + room]
        space = piece.rfind(" ")
        if space > room // 2:
            piece = piece[:space]
        piece = piece.rstrip()
        if not piece:
            break
        piece_start = unit.start + cursor.offset
        if start is None:
            start = piece_start
        parts.append(piece)
        end = piece_start + len(piece)
        cursor.offset += len(piece)
        break

    if not parts:
        # Guarantee forward progress even for pathological budgets.
        unit = cursor.current()
        piece = unit.text[cursor.offset : cursor.offset + 1]
        return piece, unit.start + cursor.offset, unit.start + cursor.offset + 1

    return (
        "".join(part if index == 0 else f"\n\n{part}" for index, part in enumerate(parts)),
        start or 0,
        end,
    )


def _chunk_units(units: Sequence[TextUnit], max_chars: int, overlap_chars: int) -> list[TextUnit]:
    """Pack units into chunks, guaranteeing a genuine overlap at every boundary."""
    if not units:
        return []
    cursor = _Cursor(list(units))
    chunks: list[TextUnit] = []

    text, start, end = _take(cursor, max_chars)
    if not text:
        return []

    while True:
        chunks.append(TextUnit(text=text, start=start, end=end))
        if cursor.done:
            break
        tail = _overlap_tail(text, end, overlap_chars)
        head = f"{tail.text}\n\n" if tail is not None else ""
        body, body_start, body_end = _take(cursor, max_chars - len(head))
        if not body:
            break
        text = f"{head}{body}"
        start = tail.start if tail is not None else body_start
        end = body_end
    return chunks


def make_chunk_id(source_file: str, section: str, ordinal: int, text: str) -> str:
    """Deterministic identifier: stable across rebuilds of the same document."""
    payload = f"{source_file}\x00{section}\x00{ordinal}\x00{text}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{Path(source_file).stem}-{ordinal:04d}-{digest}"


def chunk_markdown(
    markdown: str,
    source_file: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[DocumentChunk]:
    """Chunk a Markdown document into retrievable, self-describing units.

    Args:
        markdown: Raw document text. A leading BOM and a leading YAML
            frontmatter block are tolerated; neither is indexed.
        source_file: Knowledge-base-relative path recorded on every chunk.
        max_chars: Soft upper bound for a chunk's text.
        overlap_chars: Size of the genuine overlap shared with the next chunk
            in the same section.

    Returns:
        Chunks in document order. Empty when the document has no content. Every
        chunk's ``start_line``/``end_line``/``char_start``/``char_end`` are
        offsets into the original ``markdown`` text.

    Raises:
        ChunkingError: On invalid size parameters.
    """
    if max_chars <= 0:
        raise ChunkingError(f"max_chars must be positive, got {max_chars}")
    if overlap_chars < 0:
        raise ChunkingError(f"overlap_chars must be >= 0, got {overlap_chars}")
    if overlap_chars >= max_chars:
        raise ChunkingError(
            f"overlap_chars ({overlap_chars}) must be smaller than max_chars ({max_chars})"
        )
    if not source_file:
        raise ChunkingError("source_file must not be empty")

    chunks: list[DocumentChunk] = []
    ordinal = 0
    for section in parse_sections(markdown):
        units: list[TextUnit] = []
        for paragraph in _paragraphs(section):
            units.extend(_split_long_unit(paragraph, max_chars))
        for piece in _chunk_units(units, max_chars, overlap_chars):
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(source_file, section.name, ordinal, piece.text),
                    source_file=source_file,
                    section=section.name,
                    text=piece.text,
                    heading_path=section.heading_path,
                    heading_level=section.level,
                    start_line=section.line_at(piece.start),
                    end_line=section.line_at(max(piece.start, piece.end - 1)),
                    char_start=piece.start,
                    char_end=piece.end,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    return chunks


def chunk_file(
    path: Path | str,
    *,
    source_name: str | None = None,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[DocumentChunk]:
    """Read and chunk a Markdown file from disk.

    The file is decoded as ``utf-8-sig`` so a byte-order mark -- which many
    Windows editors still write -- never hides the document's first heading.

    Args:
        path: File to read.
        source_name: Value recorded as ``source_file``. Defaults to the file
            name; the knowledge base passes a knowledge-base-relative path so
            that identically named files in different folders stay distinct.
        max_chars: Soft upper bound for a chunk's text.
        overlap_chars: Overlap shared with the next chunk in the same section.
    """
    file_path = Path(path)
    try:
        markdown = file_path.read_text(encoding="utf-8-sig")
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise ChunkingError(f"cannot read {file_path}: {exc}") from exc
    return chunk_markdown(
        markdown,
        source_name if source_name is not None else file_path.name,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
