# Project Coding Rules (Non-Obvious Only)

- All proxy logic goes in `src/embeddings.js` as pure functions with `fetchImpl` injected as the last parameter (default `globalThis.fetch`) — never import/call fetch directly inside server.js logic; tests rely on fake fetch injection.
- New failure modes = new Error subclass in `src/embeddings.js`, then a matching `err instanceof ...` branch in `src/server.js`'s route handler. Validation errors must use `ValidationError(message, param)` so the 400 response can echo `param`.
- Keep JSDoc (`@param`/`@returns`/`@throws`/`@template`) on every exported function in `src/embeddings.js`.
- Tests: `node:test` + `node:assert/strict`; network-touching functions are tested via fake fetch objects shaped like `{ ok, status, text: async () => JSON.stringify(body) }` (see `jsonResponse` helper in `src/embeddings.test.js`).
- Concurrency-bounded async work must use `mapWithConcurrency()` from `src/embeddings.js` — it preserves result order and is already unit-tested.
- Logs must keep the `[yandex-proxy]` prefix; use `console.log`/`console.error` (no logger dependency).
