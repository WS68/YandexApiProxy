# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands

- `npm test` — run tests (`node --test`, auto-discovers `src/**/*.test.js`)
- Run a single test file: `node --test src/embeddings.test.js`
- Run a single test by name: `node --test --test-name-pattern="mapWithConcurrency" src/embeddings.test.js`
- `npm run dev` — server with `--watch`; `npm start` — plain run
- No linter/formatter is configured; no build step (plain ESM JavaScript)

## Environment (non-obvious)

- `.env` is NOT auto-loaded. Use `node --env-file=.env src/server.js` (Node 20.6+) or export vars in the shell.
- `YANDEX_API_KEY` and `YANDEX_FOLDER_ID` are required; the process exits at startup if missing (`buildConfig()` throws).
- Default port is 9988, not 3000.

## Architecture

- Two-file core: `src/embeddings.js` is pure, framework-agnostic logic (validation, fan-out, merge, fetch calls); `src/server.js` is the Express layer. Keep new logic in `embeddings.js` with `fetch` injected (`fetchImpl` param) so tests never touch the network — all tests use fake `fetch` implementations.
- Batch fan-out: an `input` array of N strings becomes N upstream calls (one string each) via `mapWithConcurrency()`, then `mergeResponses()` re-indexes `data` 0..N-1 and sums `usage` token counters.
- Error hierarchy in `src/embeddings.js` maps to HTTP responses in `src/server.js`: `ValidationError` → 400, `UpstreamNetworkError` → 502, `UpstreamError` → upstream status forwarded verbatim. Any new failure mode should follow this throw-in-core / map-in-server pattern.
- Client `model` and `encoding_format` are always overwritten (model → `emb://<folderId>/text-embeddings-v2-doc/latest`, encoding_format → `"float"`); token-array inputs are rejected with 400.
- `server.js` only starts the listener when run directly (checks `import.meta.url === pathToFileURL(process.argv[1]).href`) — safe to import `createApp`/`start` elsewhere.

## Style

- ESM everywhere (`"type": "module"`); Node built-in imports prefixed `node:` (e.g. `node:test`, `node:url`).
- JSDoc type annotations on all exported functions in `src/embeddings.js`; keep this convention.
- Tests use `node:test` `test()` + `node:assert/strict` (e.g. `assert.deepEqual`, `assert.rejects` with error-class matchers).
- Logs are prefixed `[yandex-proxy]` via `console.log`/`console.error`.
- Underscore-prefixed unused params (`_req`, `_next`) to bypass no-unused checks.
