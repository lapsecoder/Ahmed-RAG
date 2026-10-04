/**
 * Conversation behaviour, without a browser.
 *
 * The controller owns turn order, empty-input rejection, the pending flag and
 * clear. Rendering is verified separately in `dom.test.ts`.
 */

import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '../src/lib/api';
import {
  ChatController,
  canSend,
  splitStatements,
  type Transport,
} from '../src/lib/chat';
import type { SourceRef } from '../src/lib/api';

const SOURCE: SourceRef = {
  chunk_id: 'projects/resumeforge.md#0',
  source_file: 'projects/resumeforge.md',
  section: 'ResumeForge > Overview',
  similarity: 0.8,
};

function transportReturning(result: {
  state: 'answered' | 'no_context' | 'off_topic' | 'blocked' | 'error';
  answer: string;
  sources?: SourceRef[];
}): Transport {
  return vi.fn().mockResolvedValue({ sources: [], ...result });
}

describe('canSend', () => {
  it('rejects empty and whitespace-only input', () => {
    expect(canSend('')).toBe(false);
    expect(canSend('   ')).toBe(false);
    expect(canSend('\n\t ')).toBe(false);
  });

  it('accepts a real question', () => {
    expect(canSend('Who is Ahmed?')).toBe(true);
  });
});

describe('splitStatements', () => {
  it('splits a dash-prefixed answer into statements', () => {
    expect(splitStatements('- first fact.\n- second fact.')).toEqual([
      'first fact.',
      'second fact.',
    ]);
  });

  it('returns prose responses whole', () => {
    expect(splitStatements('I cannot help with that.')).toEqual([
      'I cannot help with that.',
    ]);
  });

  it('returns nothing for empty text', () => {
    expect(splitStatements('   ')).toEqual([]);
  });
});

describe('ChatController', () => {
  it('records the user turn before the assistant turn and preserves order', async () => {
    const controller = new ChatController({
      transport: transportReturning({ state: 'answered', answer: '- An answer.' }),
    });

    await controller.submit('What projects has Ahmed built?');
    await controller.submit('Where did he study?');

    const roles = controller.turns.map((t) => t.role);
    expect(roles).toEqual(['user', 'assistant', 'user', 'assistant']);
    expect(controller.turns[0]?.text).toBe('What projects has Ahmed built?');
    expect(controller.turns[1]?.text).toBe('- An answer.');
    expect(controller.turns[2]?.text).toBe('Where did he study?');
  });

  it('refuses empty input without changing state', async () => {
    const transport = transportReturning({ state: 'answered', answer: 'x' });
    const controller = new ChatController({ transport });

    expect(await controller.submit('   ')).toBe(false);
    expect(controller.turns).toHaveLength(0);
    expect(transport).not.toHaveBeenCalled();
  });

  it('shows a pending assistant turn while the request is in flight', async () => {
    let release!: (value: { state: 'answered'; answer: string; sources: SourceRef[] }) => void;
    const controller = new ChatController({
      transport: () => new Promise((resolve) => (release = resolve)),
    });

    const inFlight = controller.submit('What are his skills?');
    expect(controller.pending).toBe(true);
    expect(controller.turns).toHaveLength(2);
    expect(controller.turns[1]?.pending).toBe(true);

    release({ state: 'answered', answer: '- Python, SQL.', sources: [] });
    await inFlight;

    expect(controller.pending).toBe(false);
    expect(controller.turns[1]?.pending).toBe(false);
  });

  it('ignores a second submit while one is already in flight', async () => {
    let release!: (value: { state: 'answered'; answer: string; sources: SourceRef[] }) => void;
    const controller = new ChatController({
      transport: () => new Promise((resolve) => (release = resolve)),
    });

    const first = controller.submit('first question');
    expect(await controller.submit('second question')).toBe(false);
    expect(controller.turns).toHaveLength(2);

    release({ state: 'answered', answer: 'ok', sources: [] });
    await first;
  });

  it('carries sources through to the assistant turn', async () => {
    const controller = new ChatController({
      transport: transportReturning({ state: 'answered', answer: '- Built it.', sources: [SOURCE] }),
    });
    await controller.submit('What projects has Ahmed built?');
    expect(controller.turns[1]?.sources).toEqual([SOURCE]);
  });

  it.each(['no_context', 'off_topic', 'blocked'] as const)(
    'records the %s state',
    async (state) => {
    const controller = new ChatController({
      transport: transportReturning({ state, answer: 'server text' }),
    });
    await controller.submit('something');
      expect(controller.turns[1]?.state).toBe(state);
    },
  );

  it('turns a transport failure into a readable error state', async () => {
    const controller = new ChatController({
      transport: vi.fn().mockRejectedValue(new ApiError('network', 'Could not reach the assistant.')),
    });
    await controller.submit('hello');
    expect(controller.turns[1]?.state).toBe('error');
    expect(controller.turns[1]?.text).toMatch(/could not reach/i);
  });

  it('turns an unexpected rejection into a generic error, not a raw throw', async () => {
    const controller = new ChatController({
      transport: vi.fn().mockRejectedValue(new Error('ENOENT /home/user/.env')),
    });
    await controller.submit('hello');
    expect(controller.turns[1]?.state).toBe('error');
    expect(controller.turns[1]?.text).not.toMatch(/ENOENT|\/home\//);
  });

  it('clears the whole conversation', async () => {
    const controller = new ChatController({
      transport: transportReturning({ state: 'answered', answer: '- yes' }),
    });
    await controller.submit('one');
    await controller.submit('two');
    expect(controller.turns).toHaveLength(4);

    controller.clear();
    expect(controller.turns).toHaveLength(0);
  });

  it('notifies subscribers on every state change', async () => {
    const controller = new ChatController({
      transport: transportReturning({ state: 'answered', answer: '- yes' }),
    });
    const listener = vi.fn();
    controller.subscribe(listener);

    await controller.submit('one');
    controller.clear();

    // Three: the optimistic append, the resolved answer, and the clear.
    expect(listener).toHaveBeenCalledTimes(3);
  });

  it('trims the submitted message', async () => {
    const transport = transportReturning({ state: 'answered', answer: 'ok' });
    const controller = new ChatController({ transport });
    await controller.submit('  Who is Ahmed?  ');
    expect(transport).toHaveBeenCalledWith('Who is Ahmed?');
  });
});