/**
 * Turn a knowledge-base path into a name a visitor recognises.
 *
 * The backend cites sources by their knowledge-base-relative path, e.g.
 * `projects/resumeforge.md`. That is correct for a machine and wrong for a
 * reader: it leaks the repository's internal file layout and says nothing
 * about which document the answer actually came from.
 *
 * So the path is resolved against the document set *here*, at the UI boundary,
 * and only a display name is rendered. If a path cannot be resolved, the
 * fallback is deliberately generic - never the raw path.
 */

import type { SourceRef } from './api';

/**
 * Document id -> display name. Mirrors the knowledge base's own categories;
 * adding a document to the knowledge base means adding one line here.
 */
const DOCUMENT_NAMES: Readonly<Record<string, string>> = {
  'about.md': 'About',
  'availability.md': 'Availability',
  'certifications.md': 'Certifications',
  'contact.md': 'Contact',
  'education.md': 'Education',
  'experience.md': 'Experience',
  'faq.md': 'FAQ',
  'goals.md': 'Goals',
  'interests.md': 'Interests',
  'skills.md': 'Skills',
  'projects/resumeforge.md': 'ResumeForge',
  'projects/moviemind.md': 'MovieMind',
};

/** Anything with no mapping is presented as a document, never as a path. */
const FALLBACK = 'Knowledge base';

/** Traversal and separators that must never reach the rendered output. */
const UNSAFE = /[\\/\0]/g;

/**
 * Normalise a `source_file` to a lookup key.
 *
 * Strips any directory prefix before `projects/`, so both
 * `projects/resumeforge.md` and `resumeforge.md` resolve. A path containing a
 * traversal segment is rejected outright rather than normalised: there is no
 * legitimate citation that contains one.
 */
function normaliseKey(sourceFile: string): string | null {
  const cleaned = sourceFile.replace(UNSAFE, '/').trim().toLowerCase();
  if (cleaned === '' || cleaned.includes('..')) return null;
  const marker = cleaned.lastIndexOf('projects/');
  return marker === -1 ? cleaned : cleaned.slice(marker);
}

/** Human-facing document name for a cited source. */
export function documentLabel(sourceFile: string): string {
  const key = normaliseKey(sourceFile);
  if (key === null) return FALLBACK;
  return DOCUMENT_NAMES[key] ?? FALLBACK;
}

/**
 * Section label with the document's own name removed.
 *
 * The backend cites `ResumeForge > Overview`; showing both the document name
 * and its own name inside the section would read as a stutter.
 */
export function sectionLabel(section: string, document: string): string {
  const trimmed = section.replace(UNSAFE, ' ').trim();
  if (trimmed === '') return '';
  const tail = trimmed.includes('>') ? (trimmed.split('>').pop() ?? '') : trimmed;
  const clean = tail.trim();
  if (clean === '' || clean.toLowerCase() === document.toLowerCase()) return '';
  return clean;
}

export interface DisplaySource {
  document: string;
  section: string;
  /** Stable key for list rendering. Never a chunk id. */
  key: string;
}

/** Map raw sources into the minimal, display-safe shape the UI renders. */
export function displaySources(sources: readonly SourceRef[]): DisplaySource[] {
  const seen = new Set<string>();
  const out: DisplaySource[] = [];

  for (const source of sources) {
    const document = documentLabel(source.source_file);
    const section = sectionLabel(source.section, document);
    const key = `${document}::${section}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ document, section, key });
  }

  return out;
}

/**
 * True when text carries something the interface must never display.
 *
 * Used by the tests to assert that no internal path, score or delimiter can
 * reach the DOM through the source path, and as a cheap last-line check on
 * rendered source text.
 */
export function containsInternalDetail(text: string): boolean {
  return (
    /\.md\b/.test(text) ||
    /[\\]/.test(text) ||
    /\bfaiss\b/i.test(text) ||
    /\bbm25\b/i.test(text) ||
    /chunk[_ -]?id/i.test(text) ||
    /0\.\d{3,}/.test(text) ||
    /<<<|\[INST\]|<\|im_/i.test(text)
  );
}