import { defineConfig } from 'astro/config';

/**
 * Ask Ahmed — dev-server API proxy.
 *
 * The browser only ever talks to its own origin (`/api/...`). In development
 * this middleware forwards those calls to the FastAPI backend.
 *
 * Why not ask the backend for CORS? Because it does not need to have any. A
 * same-origin dev flow means the browser never issues a cross-origin request,
 * so the backend needs no CORS middleware and no origin allow-list. That
 * matters here: the backend is the security boundary for this whole system, and
 * a permissive `allow_origins` on it is attack surface added for the
 * convenience of a development tool.
 *
 * Astro 7 removed its own `server.proxy` option, so this is done through
 * Vite's `configureServer` hook, which Astro still builds on.
 *
 * It is development-only. In production the static build is served from the
 * same origin as the API (behind one reverse proxy), or `PUBLIC_API_BASE` is
 * set at build time.
 */
/**
 * Ask Ahmed — dev-server API proxy.
 *
 * The browser only ever talks to its own origin (`/api/...`). In development
 * this middleware forwards those calls to the FastAPI backend.
 *
 * Why not ask the backend for CORS? Because it does not need to have any. A
 * same-origin flow means the browser never issues a cross-origin request, so
 * the backend needs no CORS middleware and no origin allow-list. That matters
 * here: the backend is the security boundary for this whole system, and a
 * permissive `allow_origins` on it is attack surface added for the convenience
 * of a development tool.
 *
 * Astro 7 removed its own `server.proxy` option, so this is done through
 * Vite's `configureServer` hook, which Astro still builds on.
 *
 * Note for `astro preview`: Astro deliberately drops user `vite.plugins` when
 * it builds the preview server (see `core/preview/static-preview-server.js`),
 * so this hook does not run there. Preview gets its proxy from the standard
 * Vite `preview.proxy` option below instead.
 *
 * It is development-only. In production the static build is served from the
 * same origin as the API (behind one reverse proxy), or `PUBLIC_API_BASE` is
 * set at build time.
 */
function apiProxy(target) {
  return {
    name: 'ask-ahmed-api-proxy',
    configureServer(server) {
      server.middlewares.use('/api', async (req, res, next) => {
        // Reject anything that is not the chat API rather than proxying blind.
        if (!req.url || !req.url.startsWith('/chat')) {
          next();
          return;
        }

        const upstream = `${target}/api${req.url}`;
        const chunks = [];
        for await (const chunk of req) chunks.push(chunk);
        const body = chunks.length > 0 ? Buffer.concat(chunks) : undefined;

        try {
          const upstreamResponse = await fetch(upstream, {
            method: req.method,
            headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
            body,
          });

          const text = await upstreamResponse.text();
          res.statusCode = upstreamResponse.status;
          res.setHeader('Content-Type', 'application/json');
          res.end(text);
        } catch (error) {
          server.config.logger.error(`[api-proxy] ${upstream} failed: ${error}`);
          // Mirror the backend's own error envelope so the client sees one
          // shape whether the failure happened upstream or in transit.
          res.statusCode = 502;
          res.setHeader('Content-Type', 'application/json');
          res.end(
            JSON.stringify({
              detail: 'The assistant backend could not be reached.',
              code: 'backend_unreachable',
            }),
          );
        }
      });
    },
  };
}

const apiTarget = process.env.AHMED_RAG_API_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
  server: {
    port: 4321,
  },
  vite: {
    plugins: [apiProxy(apiTarget)],
    // `astro preview` serves the built output without our plugin, so the proxy
    // has to be declared through Vite's own preview config as well. Without
    // it, preview loads the page and then 404s on every /api/chat request.
    //
    // Note: do NOT set `preview.port` here. Astro builds the preview server by
    // `mergeConfig(userViteConfig, astroPreviewConfig)` with its own config
    // applied *second*, and that config sets `preview.port` from `server.port`
    // -- so any port set here is silently discarded and preview always lands
    // on `server.port`, i.e. the same 4321 as `astro dev`. Set `server.port`
    // to move both.
    preview: {
      proxy: {
        '/api': { target: apiTarget, changeOrigin: true },
      },
    },
  },
  devToolbar: { enabled: false },
});