# Project Documentation Rules (Non-Obvious Only)

- `README.md` is accurate and canonical (env vars table, fan-out explanation, error table); `src/embeddings.js`/`src/server.js` header comments mirror it — update all when behavior changes.
- "Proxy" here means a single-endpoint transformer, not a generic forwarder: only `POST /v1/embeddings` and `POST /embeddings` are handled; everything else (including other OpenAI endpoints) gets a 404 by design.
- The client's `model` and API key are intentionally ignored/overwritten — any dummy value works client-side. This surprises users; it's documented in README "Features".
- `encoding_format: "base64"` is unsupported: always forced to `"float"` upstream regardless of client request.
- Vectors are 1024-dimensional float arrays from Yandex `text-embeddings-v2-doc`.
- `GET /health` exists for liveness checks but is easy to miss (only endpoint besides embeddings).
