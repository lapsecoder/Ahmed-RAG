/**
 * View layer for Ask Ahmed.
 *
 * Owns the DOM and nothing else: it renders the state published by
 * `ChatController` and forwards user intent back to it. All decision-making
 * lives in `lib/`; this file makes no judgements about the assistant's answer.
 *
 * Notably absent: any client-side injection detection, scope check or
 * classification. The backend is authoritative for all of that, and duplicating
 * it here would create two definitions of "safe" that could disagree.
 */

import {
  ChatController,
  ERROR_COPY,
  STATE_COPY,
  canSend,
  splitStatements,
  type Turn,
} from '../lib/chat';
import { displaySources } from '../lib/sources';

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) {
    throw new Error(`Ask Ahmed: missing required element ${selector}`);
  }
  return element;
}

function renderSources(turn: Turn): HTMLElement | null {
  const sources = displaySources(turn.sources ?? []);
  if (sources.length === 0) return null;

  const container = document.createElement('div');
  container.className = 'sources';

  const label = document.createElement('span');
  label.className = 'sources-label';
  label.textContent = 'Sources';
  container.append(label);

  for (const source of sources) {
    const chip = document.createElement('span');
    chip.className = 'source';
    // textContent, never innerHTML: source names are data.
    chip.append(document.createTextNode(source.document));
    if (source.section !== '') {
      const section = document.createElement('span');
      section.className = 'source-section';
      section.textContent = ` · ${source.section}`;
      chip.append(section);
    }
    container.append(chip);
  }

  return container;
}

function renderTurnBody(turn: Turn): HTMLElement {
  const body = document.createElement('div');
  body.className = 'turn-body';

  if (turn.pending) {
    const pending = document.createElement('span');
    pending.className = 'dots';
    pending.setAttribute('role', 'status');
    pending.setAttribute('aria-label', 'Looking for an answer');
    for (let index = 0; index < 3; index += 1) {
      pending.append(document.createElement('span'));
    }
    body.append(pending);
    return body;
  }

  const state = turn.state ?? 'answered';

  if (state === 'answered') {
    const statements = splitStatements(turn.text);
    if (statements.length === 0) {
      body.textContent = turn.text;
    } else if (statements.length === 1) {
      body.textContent = statements[0] ?? '';
    } else {
      const list = document.createElement('ul');
      list.className = 'answer-lines';
      for (const statement of statements) {
        const item = document.createElement('li');
        item.textContent = statement;
        list.append(item);
      }
      body.append(list);
    }
  } else {
    const copy = state === 'error' ? ERROR_COPY : STATE_COPY[state];
    const block = document.createElement('div');
    block.className = 'state';
    block.dataset.state = state;

    const title = document.createElement('p');
    title.className = 'state-title';
    title.textContent = copy.title;

    const note = document.createElement('p');
    note.textContent = state === 'error' ? turn.text || copy.note : copy.note;

    block.append(title, note);
    body.append(block);
  }

  const sources = renderSources(turn);
  if (sources !== null) body.append(sources);

  return body;
}

function renderTurn(turn: Turn): HTMLElement {
  const article = document.createElement('article');
  article.className = 'turn';
  article.dataset.role = turn.role;
  article.dataset.turnId = String(turn.id);

  const speaker = document.createElement('p');
  speaker.className = 'speaker';
  speaker.textContent = turn.role === 'user' ? 'You' : 'Ahmed';

  article.append(speaker, renderTurnBody(turn));
  return article;
}

function autoGrow(textarea: HTMLTextAreaElement): void {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
}

function main(): void {
  const form = requireElement<HTMLFormElement>('[data-composer]');
  const input = requireElement<HTMLTextAreaElement>('[data-input]');
  const send = requireElement<HTMLButtonElement>('[data-send]');
  const thread = requireElement<HTMLElement>('[data-thread]');
  const empty = requireElement<HTMLElement>('[data-empty]');
  const clear = requireElement<HTMLButtonElement>('[data-clear]');
  const scrollRegion = requireElement<HTMLElement>('[data-scroll-region]');
  const jump = requireElement<HTMLButtonElement>('[data-jump]');

  const controller = new ChatController();

  const syncSendState = () => {
    send.disabled = !canSend(input.value) || controller.pending;
  };

  const scrollToNewest = () => {
    scrollRegion.scrollTop = scrollRegion.scrollHeight;
  };

  const render = () => {
    thread.replaceChildren(...controller.turns.map(renderTurn));
    empty.hidden = controller.turns.length > 0;
    clear.disabled = controller.turns.length === 0;
    input.disabled = controller.pending;
    syncSendState();

    // Only follow the conversation when the reader is already near the end.
    // Yanking the viewport away from someone reading earlier is worse than
    // making them scroll once.
    const distanceFromBottom =
      scrollRegion.scrollHeight - scrollRegion.scrollTop - scrollRegion.clientHeight;
    const atBottom = distanceFromBottom < 80;
    if (atBottom || controller.turns.length <= 2) {
      requestAnimationFrame(scrollToNewest);
      jump.hidden = true;
    } else {
      jump.hidden = false;
    }
  };

  controller.subscribe(render);
  render();

  const submit = async (message: string) => {
    const accepted = await controller.submit(message);
    if (accepted) {
      input.value = '';
      autoGrow(input);
    }
    syncSendState();
    input.focus();
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    void submit(input.value);
  });

  input.addEventListener('input', () => {
    autoGrow(input);
    syncSendState();
  });

  // Enter sends; Shift+Enter inserts a newline. The keydown handler exists
  // because a form submit on Enter would also fire on the implicit submission
  // path, which would send a message containing a newline.
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!send.disabled) form.requestSubmit();
    }
  });

  clear.addEventListener('click', () => {
    controller.clear();
    input.value = '';
    autoGrow(input);
    input.focus();
  });

  jump.addEventListener('click', () => {
    scrollToNewest();
    jump.hidden = true;
    input.focus();
  });

  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-suggestion]')) {
    button.addEventListener('click', () => {
      const question = button.dataset.suggestion ?? '';
      if (question !== '') void submit(question);
    });
  }

  if (input.value !== '') autoGrow(input);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main, { once: true });
} else {
  main();
}