"""Normalisation views for injection detection.

The rule table in :mod:`app.security.injection` matches *words*. An attacker
who does not want to be matched therefore does not have to find a new rule to
defeat -- it is enough to change the bytes of a known attack while leaving its
words intact:

.. code-block:: textIgnore all previous instructions      <- matched
    I  G  N  O  R  E   A  L  L   P  R  E  V  I  O  U  S   I N S T R U C T I O N S
    Ig-nore all previous instructions
    Ignore all previous instructions      (Cyrillic "o", U+043E)
    1gn0re all prev1ous instruct10ns
    %69%67%6e%6f%72%65%20%61%6c%6c...
    aWdub3JlIGFsbCBwcmV2aW91cyBpbnN0cnVjdGlvbnM=

Adding one keyword per spelling of that string is a losing game: the next
spelling arrives without any code change. This module attacks the *class*
instead. It derives a small, ordered list of **views** -- alternate readings of
the same message that differ only in encoding noise -- and the unchanged rule
table is run over each. Obfuscation then stops being a bypass and becomes just
another view in which the existing rules fire.

Two properties keep this from weakening the detector:

1. **Additive only.** Views are tried in order and the first one that produces
   an unsuppressed match wins. A message the detector already catches is caught
   by the verbatim view and never sees an alternative reading, so no existing
   verdict can change.
2. **Never executed.** Decoded bytes are handed to the regex rule table and
   nowhere else. They are not retrieved, not composed, not rendered and never
   leave this module. A base64 blob cannot become an instruction because there
   is no interpreter downstream -- there is no language model at all.

Everything here is deterministic, offline and dependency-free: no model, no
network, no clock, no randomness.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
import urllib.parse
from typing import Final

#: Guard against pathological input: a view longer than this is truncated
#: before matching. The API already caps message length; this bounds the work
#: the rule table does regardless of caller.
MAX_VIEW_CHARS: Final[int] = 8_000

#: Unicode format/zero-width/bidi characters, which carry no meaning and are
#: inserted solely to break a literal match ("ig\u200bnore").
_FORMAT_CHARS: Final[str] = "".join(
    chr(code)
    for code in (
        0x00AD,  # soft hyphen
        0x200B,
        0x200C,
        0x200D,
        0x200E,
        0x200F,
        0x2060,
        0x2061,
        0x2062,
        0x2063,
        0x2064,
        0xFEFF,  # BOM / zero-width no-break space
    )
) + "".join(
    chr(code)
    for code in range(0x202A, 0x202F)  # bidi embedding/override
)

#: Letters from other scripts that render as, and are used to imitate, an ASCII
#: letter. This is a *character* map, not an attack vocabulary: it maps
#: code points, so it also folds the homoglyph in an innocent name or a product
#: name, which is the correct behaviour for a comparison.
#: Cherokee's letters are not in alphabetical order -- the syllabary is
#: vowel-first (A, E, I, O, U, V, GA, KA, GE, ...) precisely so that it can be
#: *read* as Latin. That is exactly why they work as homoglyphs, and exactly
#: why they must be mapped explicitly rather than by counting code points.
#: The order below is the syllabary's visual order, not its codepoint order.
#: Cherokee has no letter for N, so an all-Cherokee message cannot spell most
#: English words -- the realistic attack mixes scripts per letter.
_CHEROKEE_ORDER: Final[str] = "AEIOUVGAKAGEGIGOGUGVHAHEHIHOHUHVLALELILOLULVMA"

#: The two Cherokee blocks whose letters are drawn to look like Latin capitals:
#: the large form at U+13A0 and the small-cap form at U+AB70. The U+13C0 block
#: is deliberately excluded -- its letters are Cherokee syllables (NAH, NE),
#: not Latin imitations, and folding them would corrupt real Cherokee text.
_CHEROKEE_FOLD: Final[dict[str, str]] = {
    chr(base + offset): syllable
    for base in (0x13A0, 0xAB70)
    for offset, syllable in enumerate(_CHEROKEE_ORDER)
}

#: Letters from other scripts that render as, and are used to imitate, an ASCII
#: letter. This is a *character* map, not an attack vocabulary: it maps code
#: points, so it also folds the homoglyph in an innocent name or a product name,
#: which is the correct behaviour for a comparison.
#:
#: Every entry is an explicit ``0x`` code point rather than the character
#: itself. That is not pedantry: a table of confusables is exactly the kind of
#: source a reviewer cannot skim safely when the characters are literal, because
#: the eye normalises them while reading -- a Cyrillic "o" in the table is
#: indistinguishable from the Latin one it replaces. Written as code points the
#: table can be audited, diffed and checked against ``unicodedata.name``, and it
#: keeps this file pure ASCII so no tool can mistake a homoglyph for an
#: identifier.
_CONFUSABLE_PAIRS: Final[tuple[tuple[int, str], ...]] = (
    # Cyrillic lowercase and barred forms
    (0x0430, "a"),
    (0x0435, "e"),
    (0x043E, "o"),
    (0x0440, "p"),
    (0x0441, "c"),
    (0x0443, "y"),
    (0x0445, "x"),
    (0x0456, "i"),
    (0x0458, "j"),
    (0x0455, "s"),
    (0x04BB, "h"),
    (0x0501, "d"),
    (0x04CF, "l"),
    (0x0432, "b"),
    (0x043C, "m"),
    (0x043D, "h"),
    (0x0442, "t"),
    (0x043A, "k"),
    # Greek lowercase and capital forms
    (0x03BF, "o"),
    (0x03B1, "a"),
    (0x03B5, "e"),
    (0x03B9, "i"),
    (0x03BD, "v"),
    (0x03C1, "p"),
    (0x03C4, "t"),
    (0x03C5, "u"),
    (0x03C7, "x"),
    (0x03B3, "y"),
    (0x03BA, "k"),
    (0x0391, "A"),
    (0x0392, "B"),
    (0x0395, "E"),
    (0x0396, "Z"),
    (0x0397, "H"),
    (0x0399, "I"),
    (0x039A, "K"),
    (0x039C, "M"),
    (0x039D, "N"),
    (0x039F, "O"),
    (0x03A1, "P"),
    (0x03A4, "T"),
    (0x03A5, "Y"),
    (0x03A7, "X"),
    # Armenian
    (0x0585, "o"),
    (0x0561, "a"),
    (0x0565, "e"),
    (0x0580, "r"),
    (0x056D, "s"),
    # Latin small capitals and modifier letters used as homoglyphs
    (0x1D00, "A"),
    (0x1D07, "E"),
    (0x026A, "I"),
    (0x1D0F, "O"),
    (0x1D1B, "T"),
    (0x1D1C, "L"),
    (0x0274, "N"),
    (0x0299, "B"),
    (0x1D04, "C"),
    (0x1D05, "D"),
    (0x0493, "F"),
    (0x0262, "G"),
    (0x029C, "H"),
    (0x1D0A, "J"),
    (0x1D0B, "K"),
    (0x1D0D, "M"),
    (0x1D18, "P"),
    (0x01EB, "O"),
    (0x1D20, "V"),
    (0x1D21, "W"),
    (0x028F, "Y"),
    (0x1D22, "Z"),
    (0x0261, "g"),
    # Punctuation used as a letter
    (0x0040, "a"),
    (0x0024, "s"),
    (0x0021, "i"),
    (0x007C, "l"),
)

_CONFUSABLE_FOLD: Final[dict[str, str]] = {
    chr(code): letter for code, letter in _CONFUSABLE_PAIRS
} | _CHEROKEE_FOLD

#: Digit/symbol to letter, for "1gn0re"-style substitutions. Only applied to
#: tokens that mix letters with these characters, so real numbers are untouched.
_LEET_FOLD: Final[dict[str, str]] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "(": "c",
}

#: Noise characters an attacker inserts *between* words or inside them. They
#: carry no lexical content, so each is either folded to a space (the token
#: boundary was real) or removed (the break was inside a word).
_SEPARATOR_CHARS: Final[str] = ".,:;_~|\\/\\*+=<>^&-#@'’"  # noqa: RUF001

_SEPARATOR_BETWEEN_ALNUM: Final[re.Pattern[str]] = re.compile(
    rf"(?<=[A-Za-z0-9])[{re.escape(_SEPARATOR_CHARS)}]+(?=[A-Za-z0-9])"
)

#: A run of single-character words has been pulled apart ("I G N O R E").
#: Three or more one-character tokens in a row is not English prose -- "a b
#: test" is two -- so a longer run is treated as one obfuscated word.
_MIN_SPACED_WORD: Final[int] = 3

#: Literal backslash escapes written out rather than interpreted.
_ESCAPE_SEQUENCE: Final[re.Pattern[str]] = re.compile(r"\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})")

_PERCENT_ESCAPE: Final[re.Pattern[str]] = re.compile(r"%[0-9a-fA-F]{2}")

#: A base64 candidate long enough to carry an instruction. Short tokens are
#: common in ordinary text ("c2VjcmV0" as an id) and are not worth decoding.
_BASE64_TOKEN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

_PRINTABLE: Final[frozenset[str]] = frozenset(chr(code) for code in range(0x20, 0x7F)) | {
    "\n",
    "\t",
    "\r",
}

_ESCAPE_TRANSLATOR: Final[dict[str, str]] = {
    "\\n": " ",
    "\\r": " ",
    "\\t": " ",
    "\\0": " ",
    "\\b": " ",
    "\\f": " ",
}


#: Built once: ``str.maketrans`` is called per fold otherwise, and the table
#: never changes after import.
_CONFUSABLE_TABLE: Final[dict[int, str]] = str.maketrans(_CONFUSABLE_FOLD)


def _fold_confusables(text: str) -> str:
    """Replace lookalike code points with the ASCII letters they imitate."""
    return text.translate(_CONFUSABLE_TABLE)


def _fold_leet(text: str) -> str:
    """Fold letter/digit substitutions inside mixed tokens.

    Only tokens that contain at least one ASCII letter and at least one folded
    character are rewritten, so "2023" and "COVID-19" keep their digits.
    """
    out: list[str] = []
    for token in re.split(r"(\W+)", text):
        if not token or not token.isascii():
            out.append(token)
            continue
        has_letter = any(character.isalpha() for character in token)
        has_folded = any(character in _LEET_FOLD for character in token)
        if has_letter and has_folded:
            out.append("".join(_LEET_FOLD.get(character, character) for character in token))
        else:
            out.append(token)
    return "".join(out)


def _collapse_spacing(text: str) -> str:
    """Rejoin words that have been pulled apart one character at a time.

    Runs are found on whitespace-delimited tokens and rejoined independently, so
    ``I G N O R E A L L`` becomes ``IGNORE ALL`` rather than one 14-letter
    token that no word-boundary pattern could match.
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= _MIN_SPACED_WORD:
            out.append("".join(run))
        else:
            out.extend(run)
        run.clear()

    for index, token in enumerate(text.split(" ")):
        if len(token) == 1 and token.isalnum():
            if index > 0 and not out and not run:
                # Leading token of the message: nothing to join it to.
                run.append(token)
                continue
            run.append(token)
        else:
            flush()
            out.append(token)
    flush()
    return " ".join(part for part in out if part)


def _unescape_literals(text: str) -> str:
    """Interpret ``\\uXXXX``/``\\xXX`` escapes that were written out literally."""

    def replace(match: re.Match[str]) -> str:
        # Drop the leading backslash and the u/x marker, keeping every hex digit.
        digits = match.group(0)[2:]
        try:
            return chr(int(digits, 16))
        except (ValueError, OverflowError):  # pragma: no cover - regex guarantees hex
            return match.group(0)

    return _ESCAPE_SEQUENCE.sub(replace, text)


def _decode_percent(text: str) -> str | None:
    """Percent-decode ``text``, or return ``None`` when there is nothing to do."""
    if not _PERCENT_ESCAPE.search(text):
        return None
    decoded = urllib.parse.unquote(text)
    return decoded if decoded != text else None


def _decode_base64_tokens(text: str) -> list[str]:
    """Decode long base64-looking tokens into readable text.

    Returns the decoded strings only. Nothing here is executed, and a token
    that is not valid base64, or that decodes to binary, is skipped.
    """
    decoded: list[str] = []
    for match in _BASE64_TOKEN.finditer(text):
        token = match.group(0)
        padded = token + "=" * (-len(token) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            candidate = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not candidate.strip():
            continue
        printable = sum(1 for character in candidate if character in _PRINTABLE)
        # A blob of random bytes decodes to "valid" UTF-8 rarely; requiring most
        # of it to be printable is what separates a payload from noise.
        if printable / len(candidate) < 0.85:
            continue
        decoded.append(candidate)
    return decoded


def _clean(text: str) -> str:
    """NFKC-fold, drop format characters and normalise newlines to spaces."""
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("\r\n", "\n").replace("\r", "\n")
    folded = "".join(
        " " if character in _FORMAT_CHARS or unicodedata.category(character) == "Cf" else character
        for character in folded
    )
    return folded


def _clip(text: str) -> str:
    return text[:MAX_VIEW_CHARS]


def base_view(text: str) -> str:
    """The canonical reading: format characters gone, confusables folded."""
    return _clip(_fold_confusables(_clean(text)))


def views(text: str) -> tuple[tuple[str, str], ...]:
    """Return ``(name, text)`` pairs to run the rule table over, in order.

    The first entry is always ``("verbatim", text)``. Every later entry is a
    reading that differs from the first only in encoding noise, so a rule that
    fires on one fires on the other for the same underlying message.
    """
    ordered: list[tuple[str, str]] = [("verbatim", text)]
    seen: set[str] = {text}

    def add(name: str, candidate: str | None) -> None:
        if candidate is None:
            return
        candidate = _clip(candidate)
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append((name, candidate))

    folded = base_view(text)
    add("nfkc", folded)
    add("unescaped", _unescape_literals(folded))
    add("percent-decoded", _decode_percent(_unescape_literals(folded)))
    add("spaced-out", _collapse_spacing(folded))
    add("newlines-removed", folded.replace("\n", ""))
    add("separators-as-space", _SEPARATOR_BETWEEN_ALNUM.sub(" ", folded))
    add("separators-removed", _SEPARATOR_BETWEEN_ALNUM.sub("", folded))
    add("leet", _fold_leet(_SEPARATOR_BETWEEN_ALNUM.sub(" ", folded)))
    for index, decoded in enumerate(_decode_base64_tokens(folded)):
        add(f"base64-{index}", base_view(decoded))

    return tuple(ordered)
