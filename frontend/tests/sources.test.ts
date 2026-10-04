/**
 * Source sanitisation.
 *
 * The rule under test: the interface shows a document a visitor recognises and
 * never an internal path, score or identifier. These are the tests that keep
 * that true as the backend's citation format changes.
 */

import { describe, expect, it } from 'vitest';
import {
  containsInternalDetail,
  displaySources,
  documentLabel,
  sectionLabel,
} from '../src/lib/sources';
import type { SourceRef } from '../src/lib/api';

function source(over: Partial<SourceRef> = {}): SourceRef {
  return {
    chunk_id: 'projects/resumeforge.md#0',
    source_file: 'projects/resumeforge.md',
    section: 'ResumeForge > Overview',
    similarity: 0.83,
    ...over,
  };
}

describe('documentLabel', () => {
  it('maps project paths to their product names', () => {
    expect(documentLabel('projects/resumeforge.md')).toBe('ResumeForge');
    expect(documentLabel('projects/moviemind.md')).toBe('MovieMind');
  });

  it('maps knowledge-base documents to readable names', () => {
    expect(documentLabel('skills.md')).toBe('Skills');
    expect(documentLabel('education.md')).toBe('Education');
    expect(documentLabel('certifications.md')).toBe('Certifications');
  });

  it('is case-insensitive', () => {
    expect(documentLabel('Projects/ResumeForge.MD')).toBe('ResumeForge');
  });

  it('falls back to a generic label rather than echoing the path', () => {
    expect(documentLabel('secret/notes.md')).toBe('Knowledge base');
    expect(documentLabel('')).toBe('Knowledge base');
  });

  it('refuses a traversal path outright', () => {
    expect(documentLabel('../../etc/passwd')).toBe('Knowledge base');
    expect(documentLabel('projects/../../secrets.md')).toBe('Knowledge base');
  });
});

describe('sectionLabel', () => {
  it('takes the leaf of a heading path', () => {
    expect(sectionLabel('ResumeForge > Overview', 'ResumeForge')).toBe('Overview');
  });

  it('drops the section when it only repeats the document name', () => {
    expect(sectionLabel('ResumeForge', 'ResumeForge')).toBe('');
  });

  it('handles an empty section', () => {
    expect(sectionLabel('', 'Skills')).toBe('');
  });
});

describe('displaySources', () => {
  it('returns document names, never file paths', () => {
    const shown = displaySources([source(), source({ source_file: 'projects/moviemind.md', section: 'MovieMind > Overview' })]);
    expect(shown.map((s) => s.document)).toEqual(['ResumeForge', 'MovieMind']);
  });

  it('de-duplicates repeated citations of the same section', () => {
    const shown = displaySources([
      source(),
      source({ chunk_id: 'projects/resumeforge.md#4' }),
      source({ section: 'ResumeForge > Limits' }),
    ]);
    expect(shown).toHaveLength(2);
  });

  it('produces nothing that carries an internal detail', () => {
    const shown = displaySources([
      source(),
      source({ source_file: 'contact.md', section: 'Contact > Email' }),
    ]);
    for (const item of shown) {
      expect(containsInternalDetail(item.document)).toBe(false);
      expect(containsInternalDetail(item.section)).toBe(false);
      expect(containsInternalDetail(item.key)).toBe(false);
    }
  });
});

describe('containsInternalDetail', () => {
  it.each([
    'skills.md',
    'C:\\kb\\skills.md',
    'projects/resumeforge.md',
    'FAISS index',
    'bm25 score 0.82',
    'chunk_id abc',
    'similarity 0.9123',
    '<<<CONTEXT_CHUNK',
    '[INST]',
  ])('flags %s', (text) => {
    expect(containsInternalDetail(text)).toBe(true);
  });

  it.each(['ResumeForge', 'Overview', 'Certifications', 'MovieMind', 'Availability'])(
    'does not flag the friendly name %s',
    (text) => {
      expect(containsInternalDetail(text)).toBe(false);
    },
  );
});