# Project Debug Rules (Non-Obvious Only)

- Server silently does nothing when imported — it only `listen()`s when run directly (`import.meta.url === pathToFileURL(process.argv[1]).href` in `src/server.js`). Use `start()` explicitly when debugging via a harness.
- Missing `YANDEX_API_KEY`/`YANDEX_FOLDER_ID` crashes at startup by design (`buildConfig()` throws). `.env` is NOT auto-loaded — forgetting `--env-file=.env` is the #1 "server won't start" cause.
- Batch requests fail whole-or-nothing: if any of the N fan-out upstream calls throws, the entire request fails and the first thrown error wins (no partial responses) — check `[yandex-proxy] yandex status:` logs to see which per-item call failed.
- Malformed upstream JSON doesn't throw; it is wrapped as `{ raw: <text> }` in `callUpstream()` — inspect `err.body.raw` when upstream errors look "empty".
- Upstream HTTP error status is forwarded verbatim to the client (e.g. Yandex 429 → client 429); only network/timeout failures become 502.
- JSON body-parser errors are converted to OpenAI-style JSON in the final error middleware — `entity.parse.failed` → 400, `entity.too.large` → 413 (body limit is 2mb).
