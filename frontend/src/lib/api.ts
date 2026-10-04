/**
 * Typed client for the Ahmed-RAG chat endpoint.
 *
 * The backend is authoritative. This module does not decide whether a message
 * is an injection, is in scope, or is safe to show: it sends a message and
 * renders whatever comes back. There is deliberately no client-side security
 * logic here, and none may be added.
 */

export type QueryClassification = 'in_scope' | 'off_topic' | 'injection';

export type ChatOutcome =
  | 'answered'
  | 'no_context'
  | 'off_topic'
  | 'blocked_injection'
  | 'blocked_output';

export interface SourceRef {
  chunk_id: string;
  source_file: string;
  section: string;
  similarity: number;
}

export interface ChatResponse {
  response: string;
  classification: QueryClassification;
  outcome: ChatOutcome;
  sources: SourceRef[];
  retrieval_scores: number[];
  llm_used: boolean;
  injection_rule_ids: string[];
}

export interface ChatRequest {
  message: string;
}

/**
 * The five states the UI distinguishes.
 *
 * `error` is not a backend outcome - it is the UI's own state for a request
 * that never produced a usable answer (network failure, timeout, bad status,
 * unparseable body).
 */
export type UiState = 'answered' | 'no_context' | 'off_topic' | 'blocked' | 'error';

export interface UiResult {
  state: UiState;
  answer: string;
  sources: SourceRef[];
}

/** Thrown for any request that did not yield a usable response. */
export class ApiError extends Error {
  readonly kind: 'network' | 'timeout' | 'http' | 'malformed';

  constructor(kind: ApiError['kind'], message: string) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
  }
}

const DEFAULT_BASE = '/api';
const DEFAULT_TIMEOUT_MS = 20_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const OUTCOMES: ReadonlySet<string> = new Set<ChatOutcome>([
  'answered',
  'no_context',
  'off_topic',
  'blocked_injection',
  'blocked_output',
]);

const CLASSIFICATIONS: ReadonlySet<string> = new Set<QueryClassification>([
  'in_scope',
  'off_topic',
  'injection',
]);

/**
 * Map a backend outcome onto the state the UI renders.
 *
 * `blocked_injection` and `blocked_output` both mean the same thing to a
 * visitor - the request was not answered - and both map to `blocked`. The
 * reason is never shown: `injection_rule_ids` names the internal rule that
 * fired, which is security internals the interface must not expose.
 */
export function outcomeToState(outcome: ChatOutcome): UiState {
  switch (outcome) {
    case 'answered':
      return 'answered';
    case 'no_context':
      return 'no_context';
    case 'off_topic':
      return 'off_topic';
    case 'blocked_injection':
    case 'blocked_output':
      return 'blocked';
  }
}

/**
 * Validate an untrusted JSON body into a `ChatResponse`.
 *
 * Anything unrecognised is rejected rather than coerced. A chat UI that
 * renders whatever it happens to receive is a UI that will eventually render
 * whatever a broken or hostile server sends.
 */
export function parseChatResponse(body: unknown): ChatResponse {
  if (!isRecord(body)) {
    throw new ApiError('malformed', 'The assistant sent a response we could not read.');
  }
  const { response, classification, outcome } = body;

  if (typeof response !== 'string' || response.trim() === '') {
    throw new ApiError('malformed', 'The assistant sent an empty response.');
  }
  if (typeof outcome !== 'string' || !OUTCOMES.has(outcome)) {
    throw new ApiError('malformed', 'The assistant sent an unexpected response.');
  }
  if (typeof classification !== 'string' || !CLASSIFICATIONS.has(classification)) {
    throw new ApiError('malformed', 'The assistant sent an unexpected response.');
  }

  const rawSources = body.sources;
  if (rawSources !== undefined && !Array.isArray(rawSources)) {
    throw new ApiError('malformed', 'The assistant sent an unexpected response.');
  }

  const sources: SourceRef[] = [];
  for (const entry of rawSources ?? []) {
    // A malformed *source* is skipped, not fatal. Losing one citation is a far
    // better failure than discarding an otherwise valid answer because one
    // entry in an array was the wrong shape.
    if (!isRecord(entry)) continue;
    // Anything that is not a plain string is dropped rather than rendered.
    const sourceFile = typeof entry.source_file === 'string' ? entry.source_file : '';
    const section = typeof entry.section === 'string' ? entry.section : '';
    if (sourceFile === '') continue;
    sources.push({
      chunk_id: typeof entry.chunk_id === 'string' ? entry.chunk_id : '',
      source_file: sourceFile,
      section,
      similarity: typeof entry.similarity === 'number' ? entry.similarity : 0,
    });
  }

  return {
    response,
    classification: classification as QueryClassification,
    outcome: outcome as ChatOutcome,
    sources,
    retrieval_scores: Array.isArray(body.retrieval_scores)
      ? body.retrieval_scores.filter((n): n is number => typeof n === 'number')
      : [],
    llm_used: body.llm_used === true,
    injection_rule_ids: Array.isArray(body.injection_rule_ids)
      ? body.injection_rule_ids.filter((n): n is string => typeof n === 'string')
      : [],
  };
}

export interface AskOptions {
  baseUrl?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
}

/**
 * Ask one question.
 *
 * Throws {@link ApiError} with a message that is safe to show a visitor. The
 * server's `detail` string is deliberately *not* forwarded: it can contain
 * internal configuration and exception text.
 */
export async function ask(
  message: string,
  options: AskOptions = {},
): Promise<ChatResponse> {
  const {
    baseUrl = import.meta.env?.PUBLIC_API_BASE ?? DEFAULT_BASE,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    signal,
    fetchImpl = fetch,
  } = options;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort);

  let response: Response;
  try {
    response = await fetchImpl(`${baseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ message } satisfies ChatRequest),
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) {
      throw new ApiError('timeout', 'The assistant took too long to respond.');
    }
    if (signal?.aborted) {
      throw new ApiError('network', 'The request was cancelled.');
    }
    throw new ApiError(
      'network',
      'Could not reach the assistant. It may be starting up or offline.',
    );
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }

  if (!response.ok) {
    // 4xx and 5xx alike collapse to one calm message. The status is useful in
    // the console for a developer and useless - and potentially revealing - in
    // the interface.
    console.warn(`[ask-ahmed] backend returned ${response.status}`);
    throw new ApiError(
      'http',
      'The assistant is unavailable right now. Please try again in a moment.',
    );
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError('malformed', 'The assistant sent a response we could not read.');
  }

  return parseChatResponse(body);
}