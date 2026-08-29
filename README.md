# YandexProxy — OpenAI-compatible embeddings proxy

A small, single-purpose Node.js proxy that exposes an **OpenAI-compatible** embeddings
endpoint on `http://localhost:9988` and forwards requests to the provider selected in
[`config.json`](config.json).

The checked-in default is `openai`, using `https://routerai.ru/api/v1` and
`qwen/qwen3-embedding-8b`. Set `mode` to `yandex` to use Yandex credentials and model.

```json
{
  "mode": "openai",
  "baseUrl": "https://routerai.ru/api/v1",
  "model": "qwen/qwen3-embedding-8b"
}
```

This lets you point any OpenAI SDK / tool at `http://localhost:9988/v1` and get
embeddings from Yandex `text-embeddings-v2-doc` without changing your application code.

## Features

- Listens on `http://localhost:9988` (configurable via `PORT`).
- Accepts `POST /v1/embeddings` and `POST /embeddings` (any other path → 404).
- Injects the selected provider's API key as a Bearer token
  (the client's own key is ignored, so any dummy value works client-side).
- Rewrites `model` to `emb://$YANDEX_FOLDER_ID/text-embeddings-v2-doc/latest`.
- Forces `encoding_format: "float"` in the outgoing body.
- Forwards `dimensions` if present — must be `256`, `512` or `768`, otherwise the
  request is rejected with a `400` — and defaults to `768` when absent.
- **Batch processing:** an `input` array of N strings is processed as N upstream
  calls (one string each, strictly sequential) and the responses are merged back
  into a single OpenAI-shaped response with re-indexed `data` and summed `usage`.
- **429 retry:** upstream rate-limit responses (`429`) are retried with
  exponential backoff (`1s → 2s → 4s → 8s → 16s → 32s → 60s`, capped); if a retry
  following the full 60s wait still returns `429`, the whole request fails with `429`.

## Requirements

- Node.js **18+** (uses built-in `fetch` and `AbortSignal.timeout`).
- A Yandex Cloud account with the **Foundation Models API** enabled and an API key.

## Setup

```bash
# 1. install dependencies
npm install

# 2. create .env (see .env.example) or export the vars
export OPENAI_API_KEY="your_routerai_key" # for mode=openai
export YANDEX_API_KEY="your_api_key"
export YANDEX_FOLDER_ID="b1gxxxxxxxxxxxxxxxxx"
export PORT=9988   # optional, default 9988

# 3. run
npm start
```

> The proxy reads variables from the **process environment**. To load them from a
> `.env` file use `node --env-file=.env src/server.js` (Node 20.6+).

Startup validation checks `OPENAI_API_KEY` for `openai`, and `YANDEX_API_KEY` plus
`YANDEX_FOLDER_ID` for `yandex`.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `YANDEX_API_KEY` | ✅ | — | Yandex Cloud API key (service account key or OAuth token) |
| `OPENAI_API_KEY` | openai mode | — | API key sent to the configured OpenAI-compatible endpoint |
| `YANDEX_FOLDER_ID` | ✅ | — | Folder id used to build `emb://<id>/text-embeddings-v2-doc/latest` |
| `PORT` | — | `9988` | Port the proxy listens on |
| `YANDEX_BASE_URL` | — | `https://ai.api.cloud.yandex.net/v1` | Upstream base URL (trailing slash stripped) |
| `YANDEX_TIMEOUT_MS` | — | `60000` | Upstream request timeout |

## Dimensions

Yandex accepts `256`, `512`, or `768` and defaults to `768`. OpenAI mode accepts
`256`, `512`, `768`, `1024`, `1536`, `2048`, `3072`, or `4096` and defaults to `1536`.

## Usage

### cURL

```bash
# single string
curl http://localhost:9988/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"any-model-name","input":"hello world"}'

# batch (array of strings) — split into N upstream calls, merged response
curl http://localhost:9988/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"any-model-name","input":["first text","second text","third text"]}'
```

### OpenAI JavaScript SDK

```js
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'any-dummy-key', // the proxy injects the real Yandex key
  baseURL: 'http://localhost:9988/v1',
});

const res = await client.embeddings.create({
  model: 'any-model-name', // replaced with the Yandex model id
  input: ['hello', 'world', 'from', 'yandex'],
});

console.log(res.data.map((d) => d.index));      // [0, 1, 2, 3]
console.log(res.data[0].embedding.length);      // 1024 (float32 vector)
console.log(res.usage.prompt_tokens);
```

### Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-dummy-key",          # the proxy injects the real Yandex key
    base_url="http://localhost:9988/v1",
)

res = client.embeddings.create(
    model="any-model-name",           # replaced with the Yandex model id
    input=["hello", "world"],
)

print([d.index for d in res.data])    # [0, 1]
```

## How batch processing works

Given `input: ["a", "b", "c"]`, the proxy:

1. Validates the body (strings only; token arrays are rejected with a 400).
2. Makes **3 upstream calls** to `/v1/embeddings`, each with `input: "a"`, `"b"`, `"c"`,
   strictly sequentially (each call is awaited before the next starts).
3. Merges the responses: `data` entries are re-indexed `0..N-1` in original order and
   `usage` token counters are summed.
4. Returns a single OpenAI-shaped response.

Upstream `429` responses are retried with exponential backoff (`1s` doubling, capped
at `60s`). If a retry following the full 60s wait still returns `429`, the whole
request fails with a `429` (worst case ~2 minutes per upstream call).

A single-string `input` is forwarded as-is (one upstream call).

## Error handling

| Situation | Response |
|---|---|
| Invalid body / unsupported `input` | `400` OpenAI-style error |
| Unsupported `dimensions` (not 256/512/768) | `400` OpenAI-style error |
| Upstream returns an HTTP error | Upstream status + error body forwarded |
| Upstream `429` rate limit | Retried with backoff `1s → 60s` (capped); if still `429` after the full 60s wait, `429` returned to the client |
| Upstream unreachable / timeout | `502` `upstream_network_error` |
| Any other path | `404` |

`GET /health` returns `{"status":"ok"}` for liveness checks.

## Development

```bash
npm test          # run unit tests (node:test, no extra deps)
npm run dev       # start with --watch
```

## Notes / limitations

- Only `POST /v1/embeddings` (and `/embeddings`) is proxied; other OpenAI endpoints
  are intentionally not forwarded (single-purpose proxy).
- `encoding_format` is always forced to `"float"` — `"base64"` is not supported.
- `dimensions` is validated against `256`, `512` and `768` and defaults to `768`
  when the client does not send it; any other value is rejected with a `400`.
- Yandex document embeddings produce 1024-dimensional float vectors.
