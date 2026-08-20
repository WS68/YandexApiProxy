# Project Architecture Rules (Non-Obvious Only)

- Strict two-layer split: `src/embeddings.js` (pure, framework-free, fetch-injected) vs `src/server.js` (Express, HTTP mapping only). Any logic added to server.js that isn't request/response shaping is misplaced.
- Batch fan-out is core to the design: Yandex accepts one string per call, so N-string batches become N upstream calls through `mapWithConcurrency()` (worker-pool, order-preserving), merged by `mergeResponses()` (re-index `data` 0..N-1, sum `usage`). Changing input handling must preserve this contract.
- Error protocol between layers: core throws typed errors (`ValidationError`/`UpstreamError`/`UpstreamNetworkError`), server maps them to OpenAI-shaped JSON envelopes (400 / forwarded upstream status / 502). Keep the hierarchy exhaustive — server's final fallback is 500.
- `buildConfig(env = process.env)` takes env explicitly so tests can pass fake env objects; config is resolved once at startup and threaded through — no `process.env` reads scattered in handlers.
- Config contract: `{ apiKey, folderId, baseUrl (trailing slashes stripped), concurrency, timeoutMs }`; timeouts use `AbortSignal.timeout` per upstream call.
- Single dependency (express); everything else is Node built-ins. Zero-dependency core is intentional — avoid adding runtime deps for core logic.
- Response shape contract is OpenAI's embeddings schema (`object: "list"`, `data[].index`, `usage.prompt_tokens/total_tokens`); `completion_tokens` only included when upstream provides it.
