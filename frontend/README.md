# Ask Ahmed — UI

A standalone chat interface for the Ahmed-RAG backend. A visitor asks a
question; the backend decides whether it can be answered from the approved
knowledge base and returns either a grounded answer with citations, or a calm
refusal.

## Running it

Two processes. The backend first:

```bash
# from the repository root
ahmed-rag            # or: uvicorn app.main:app --port 8000
```

Then the UI:

```bash
cd frontend
npm install
npm run dev          # http://localhost:4321
```

The UI calls `/api/chat` on its own origin. In development a small Vite
middleware forwards that to `http://127.0.0.1:8000`, so the browser never makes
a cross-origin request and the backend needs no CORS middleware. Point it
elsewhere with `AHMED_RAG_API_URL`:

```bash
AHMED_RAG_API_URL=http://192.168.1.20:8000 npm run dev
```

To check the built output locally:

```bash
npm run build
npm run preview       # http://localhost:4321
```

`preview` proxies to the backend too (via Vite's `preview.proxy`; Astro drops
user `vite.plugins` when it builds the preview server, so the dev-only
`configureServer` hook does not run there). Without that, preview loads the
page and then 404s on every `/api/chat` request.

`dev` and `preview` share port 4321 — only one runs at a time. Astro sets the
preview port from `server.port`, ignoring `vite.preview.port`, so change
`server.port` to move both.

For production, `astro build` emits static files. Serve them from the same
origin as the API (one reverse proxy in front of both), or set
`PUBLIC_API_BASE` at build time.

## Gates

```bash
npm run build     # astro build
npx tsc --noEmit  # typecheck
npm test          # vitest, jsdom
```

`astro check` is not used: the portfolio pins TypeScript 7, which `astro check`
does not yet support. `tsc --noEmit` covers the TypeScript sources.

## How it relates to the portfolio

The visual language is not invented here. `src/styles/tokens.css` and
`src/styles/global.css` are **byte-identical copies** of the portfolio's files,
so the palette, Fraunces/Inter/JetBrains Mono stack, `.pill`, `.card`,
`.arrow-link` and `.eyebrow` conventions are literally the same code. Only
`src/styles/chat.css` is new, and it adds no colours, fonts or radii — it uses
only the tokens already defined.

There is no Tailwind. The portfolio does not use it, so this does not either.

No UI framework. The portfolio is Astro and so is this; the only client code is
a few hundred lines of vanilla TypeScript.

## Security posture

The backend is the security boundary and stays authoritative.

* **No client-side security logic.** There is no injection detection, scope
  check or classification in this codebase, and adding any would create a
  second, weaker definition of "safe" that could disagree with the backend's.
  Every message is sent verbatim; whatever comes back is what is rendered.
* **Refusals reveal nothing.** The backend returns `injection_rule_ids`, which
  names the internal rule that fired. The UI deliberately ignores it, along
  with `retrieval_scores` and `classification`.
* **Citations are sanitised at this boundary.** The backend cites documents by
  knowledge-base-relative path (`projects/resumeforge.md`). `src/lib/sources.ts`
  resolves that to a display name (`ResumeForge`) and the raw path is never
  rendered. Unknown or traversal-shaped paths fall back to a generic label
  rather than being echoed.
* **Untrusted response bodies are validated, not trusted.**
  `parseChatResponse` rejects anything unrecognised rather than coercing it.
* **Errors never leak internals.** The server's `detail` string is not
  forwarded to the interface; it can contain exception and configuration text.

## Tests

`tests/` runs in jsdom against the **built** page in `dist/`, so the DOM under
test is the DOM that ships. Run `npm run build` before `npm test`.

| File | Covers |
| --- | --- |
| `api.test.ts` | **request payload contract** (`message`, never `query`), response validation, outcome mapping, timeout, HTTP errors, malformed bodies |
| `sources.test.ts` | document-name resolution, traversal rejection, no paths/scores/id reach the UI |
| `chat.test.ts` | turn order, empty input, pending state, clear, error mapping |
| `dom.test.ts` | rendering, all four response states, loading, keyboard, accessibility, no client-side security |

The backend side of the contract is pinned by
`../tests/test_ask_ahmed_contract.py`, so a backend change that would break this
interface fails in pytest rather than in a browser.