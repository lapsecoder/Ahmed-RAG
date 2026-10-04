/**
 * Conversation state for Ask Ahmed.
 *
 * Deliberately DOM-free: this module owns the turns, the pending flag and the
 * validation rules, and knows nothing about elements. The view layer in
 * `scripts/chat.ts` renders whatever state this exposes. That split is what
 * makes the behaviour testable without a browser.
 */

import { ApiError, ask, type SourceRef, type UiState } from './api';

export interface Turn {
  id: number;
  role: 'user' | 'assistant';
  /** Rendered text. For an assistant turn this is the answer or the state copy. */
  text: string;
  state?: UiState;
  sources?: SourceRef[];
  pending?: boolean;
}

export type Transport = (message: string) => Promise<{
  state: UiState;
  answer: string;
  sources: SourceRef[];
}>;

export interface ChatOptions {
  transport?: Transport;
}

/** Turns that should be rendered. A pending assistant turn is included so the
 * user sees that the request is in flight. */
export function visibleTurns(turns: readonly Turn[]): Turn[] {
  return turns.filter((turn) => !(turn.pending && !turn.text));
}

/** Whether the composer should accept this input. */
export function canSend(text: string): boolean {
  return text.trim().length > 0;
}

/**
 * Split a backend answer into statements for rendering.
 *
 * An `answered` response arrives as newline-separated sentences, each prefixed
 * with "- ". Non-answered responses are plain prose and are returned whole.
 * This is presentation only; it applies no judgement about content.
 */
export function splitStatements(text: string): string[] {
  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const bulleted = lines.filter((line) => line.startsWith('- '));
  if (bulleted.length === 0) {
    const whole = text.trim();
    return whole === '' ? [] : [whole];
  }
  return bulleted.map((line) => line.slice(2).trim()).filter((line) => line !== '');
}

/**
 * The calm, human-readable framing for each non-answer state.
 *
 * These are UI copy. They say what happened and nothing about how the backend
 * works: no classifier names, no rule ids, no thresholds, no scores.
 */
export const STATE_COPY: Record<Exclude<UiState, 'answered' | 'error'>, { title: string; note: string }> = {
  no_context: {
    title: 'Not in the knowledge base',
    note: "There's nothing recorded about that yet. Try asking about projects, skills, education or availability.",
  },
  off_topic: {
    title: 'Outside this assistant',
    note: 'Ask Ahmed is grounded in his own portfolio, so it only covers his work, projects, skills, education and interests.',
  },
  blocked: {
    title: "I can't help with that",
    note: 'This assistant answers questions about Ahmed from a fixed set of documents and does not change how it works on request.',
  },
};

export const ERROR_COPY: { title: string; note: string } = {
  title: 'Something went wrong',
  note: 'The assistant could not be reached. Please try again in a moment.',
};

export class ChatController {
  #turns: Turn[] = [];
  #nextId = 1;
  #pending = false;
  #transport: Transport;
  #listeners = new Set<() => void>();

  constructor(options: ChatOptions = {}) {
    const transport = options.transport;
    this.#transport =
      transport ??
      (async (message: string) => {
        const result = await ask(message);
        return { state: result.outcome === 'answered' ? 'answered' : mapOutcome(result.outcome), answer: result.response, sources: result.sources };
      });
  }

  get turns(): Turn[] {
    return this.#turns;
  }

  get pending(): boolean {
    return this.#turns.some((turn) => turn.pending === true);
  }

  subscribe(listener: () => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  clear(): void {
    this.#turns = [];
    this.#pending = false;
    this.#notify();
  }

  /**
   * Submit a message.
   *
   * Returns false without touching state when the input is blank or a request
   * is already in flight, so the caller can simply do nothing.
   */
  async submit(raw: string): Promise<boolean> {
    if (!canSend(raw) || this.#pending) return false;

    const message = raw.trim();
    const userTurn: Turn = { id: this.#nextId++, role: 'user', text: message };
    const pendingTurn: Turn = {
      id: this.#nextId++,
      role: 'assistant',
      text: '',
      state: 'answered',
      pending: true,
    };

    this.#turns = [...this.#turns, userTurn, pendingTurn];
    this.#pending = true;
    this.#notify();

    let result: { state: UiState; answer: string; sources: SourceRef[] };
    try {
      result = await this.#transport(message);
    } catch (error) {
      result =
        error instanceof ApiError
          ? { state: 'error', answer: error.message, sources: [] }
          : {
              state: 'error',
              answer: 'The assistant could not be reached. Please try again in a moment.',
              sources: [],
            };
    }

    this.#turns = this.#turns.map((turn) =>
      turn.id === pendingTurn.id
        ? {
            ...turn,
            text: result.answer,
            state: result.state,
            sources: result.sources,
            pending: false,
          }
        : turn,
    );
    this.#pending = false;
    this.#notify();
    return true;
  }

  #notify(): void {
    for (const listener of this.#listeners) listener();
  }
}

function mapOutcome(outcome: string): UiState {
  switch (outcome) {
    case 'blocked_injection':
    case 'blocked_output':
      return 'blocked';
    case 'no_context':
      return 'no_context';
    case 'off_topic':
      return 'off_topic';
    default:
      return 'answered';
  }
}