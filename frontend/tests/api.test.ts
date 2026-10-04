/**
 * API client contract.
 *
 * The client treats every response as untrusted input. These tests pin that
 * behaviour, including the cases where a malformed or hostile body must be
 * rejected rather than rendered.
 */

import { describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  ask,
  outcomeToState,
  parseChatResponse,
  type ChatResponse,
} from '../src/lib/api';

const VALID: ChatResponse = {
  response: "- Ahmed builds ResumeForge.",
  classification: 'in_scope',
  outcome: 'answered',
  sources: [
    { chunk_id: 'projects/resumeforge.md#0', source_file: 'projects/resumeforge.md', section: 'ResumeForge > Overview', similarity: 0.71 },
  ],
  retrieval_scores: [0.71],
  llm_used: false,
  injection_rule_ids: [],
};

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
}

describe('outcomeToState', () => {
  it('maps every backend outcome onto exactly one UI state', () => {
    expect(outcomeToState('answered')).toBe('answered');
    expect(outcomeToState('no_context')).toBe('no_context');
    expect(outcomeToState('off_topic')).toBe('off_topic');
    // Both refusals collapse to one calm state; the reason is never surfaced.
    expect(outcomeToState('blocked_injection')).toBe('blocked');
    expect(outcomeToState('blocked_output')).toBe('blocked');
  });
});

describe('parseChatResponse', () => {
  it('accepts a well-formed answer and keeps its sources', () => {
    const parsed = parseChatResponse(VALID);
    expect(parsed.response).toContain('ResumeForge');
    expect(parsed.sources).toHaveLength(1);
    expect(parsed.sources[0]?.source_file).toBe('projects/resumeforge.md');
  });

  it.each([
    ['a non-object body', 'not json at all'],
    ['an array body', [1, 2, 3]],
    ['a null body', null],
  ])('rejects %s', (_label, body) => {
    expect(() => parseChatResponse(body)).toThrow(ApiError);
  });

  it('rejects an empty response string', () => {
    expect(() => parseChatResponse({ ...VALID, response: '   ' })).toThrow(
      /empty response/i,
    );
  });

  it('rejects an unrecognised outcome rather than guessing', () => {
    expect(() => parseChatResponse({ ...VALID, outcome: 'exploded' })).toThrow(
      ApiError,
    );
  });

  it('rejects an unrecognised classification', () => {
    expect(() => parseChatResponse({ ...VALID, classification: 'maybe' })).toThrow(
      ApiError,
    );
  });

  it('rejects a non-array sources field', () => {
    expect(() => parseChatResponse({ ...VALID, sources: 'skills.md' })).toThrow(
      ApiError,
    );
  });

  it('drops malformed source entries instead of failing the whole answer', () => {
    const parsed = parseChatResponse({
      ...VALID,
      sources: [null, { source_file: 42 }, { source_file: 'skills.md', section: 'Skills' }],
    });
    expect(parsed.sources).toHaveLength(1);
    expect(parsed.sources[0]?.source_file).toBe('skills.md');
  });

  it('never coerces a non-string field into text', () => {
    const parsed = parseChatResponse({
      ...VALID,
      response: '- real answer',
      injection_rule_ids: ['ignore_previous_instructions', 7],
    });
    expect(parsed.injection_rule_ids).toEqual(['ignore_previous_instructions']);
  });
});

describe('request payload contract', () => {
  /**
   * The backend's `ChatRequest` has exactly one field, `message`. Sending
   * `query` instead is a 422 from FastAPI, which the interface can only
   * surface as a generic "unavailable" -- so the failure looks like a network
   * problem rather than a payload problem, and is easy to misdiagnose.
   *
   * These tests pin the key name, the key set, and the absence of the wrong
   * name, so a regression fails here rather than in a browser.
   */
  async function sentBody(message = 'Who are you?'): Promise<Record<string, unknown>> {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(VALID));
    await ask(message, { baseUrl: '/api', fetchImpl: fetchImpl as unknown as typeof fetch });
    const [, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    return JSON.parse(String(init.body)) as Record<string, unknown>;
  }

  it('sends the question under the key "message"', async () => {
    const body = await sentBody('Who are you?');
    expect(body.message).toBe('Who are you?');
  });

  it('never sends the key "query"', async () => {
    const body = await sentBody('Who are you?');
    expect(body).not.toHaveProperty('query');
    expect(Object.keys(body)).not.toContain('query');
  });

  it('sends exactly one field, so the backend cannot reject it as unexpected', async () => {
    const body = await sentBody();
    // `ChatRequest` is `extra="forbid"`: an extra key is a 422, not a warning.
    expect(Object.keys(body)).toEqual(['message']);
  });

  it('passes the question through unchanged', async () => {
    const awkward = '  What is his CGPA?  ';
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(VALID));
    await ask(awkward, { baseUrl: '/api', fetchImpl: fetchImpl as unknown as typeof fetch });
    const [, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    // `ask` is a transport and does not reinterpret its argument; trimming is
    // the controller's decision, made before this point.
    expect(JSON.parse(String(init.body)).message).toBe(awkward);
  });

  it('targets the chat route', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(VALID));
    await ask('hello', { baseUrl: '/api', fetchImpl: fetchImpl as unknown as typeof fetch });
    expect(fetchImpl.mock.calls[0]?.[0]).toBe('/api/chat');
  });
});

describe('ask', () => {
  it('posts the message and returns a validated response', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(VALID));
    const result = await ask('What projects has Ahmed built?', {
      baseUrl: '/api',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });

    expect(result.outcome).toBe('answered');
    const [, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).toEqual({
      message: 'What projects has Ahmed built?',
    });
  });

  it('raises a timeout when the backend does not answer in time', async () => {
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    );
    const promise = ask('hello', {
      timeoutMs: 5,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await expect(promise).rejects.toThrow(/too long/i);
  });

  it('raises a network error when the backend is unreachable', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(
      ask('hello', { fetchImpl: fetchImpl as unknown as typeof fetch }),
    ).rejects.toThrow(/could not reach/i);
  });

  it.each([400, 404, 422, 500, 503])(
    'raises a calm error for HTTP %i without leaking the status',
    async (status) => {
      const fetchImpl = vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ detail: 'internal traceback here', code: 'internal_error' }, { status }),
        );
      await expect(
        ask('hello', { fetchImpl: fetchImpl as unknown as typeof fetch }),
      ).rejects.toThrow(/unavailable right now/i);
    },
  );

  it('never surfaces the server detail string to the visitor', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ detail: 'Traceback: File "app/api/routes.py"', code: 'internal_error' }, { status: 500 }),
      );
    try {
      await ask('hello', { fetchImpl: fetchImpl as unknown as typeof fetch });
      expect.unreachable('should have thrown');
    } catch (error) {
      expect((error as Error).message).not.toMatch(/traceback|routes\.py/i);
    }
  });

  it('raises a malformed error when the body is not JSON', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(new Response('<html>oops</html>', { status: 200 }));
    await expect(
      ask('hello', { fetchImpl: fetchImpl as unknown as typeof fetch }),
    ).rejects.toThrow(/could not read/i);
  });
});