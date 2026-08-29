# YandexProxy — OpenAI-compatible embeddings proxy

A small, single-purpose Node.js proxy that exposes an **OpenAI-compatible** embeddings
endpoint on `http://localhost:9988` and forwards requests to the provider selected in
[`config.json`](config.json).

The checked-in default is `openai`, using `https://routerai.ru/api/v1` and
`qwen/qwen3-embedding-4b`. Set `mode` to `yandex` to use Yandex credentials and model.

```json
{
  "mode": "openai",
  "baseUrl": "https://routerai.ru/api/v1",
  "model": "qwen/qwen3-embedding-4b",
  "maxBatchStrings": 60
}
```

This lets you point any OpenAI SDK / tool at `http://localhost:9988/v1` and get
embeddings from the configured OpenAI-compatible provider without changing your
application code. In `yandex` mode, requests are sent to Yandex's
`text-embeddings-v2-doc` model.

## Features

- Listens on `http://localhost:9988` (configurable via `PORT`).
- Accepts `POST /v1/embeddings` and `POST /embeddings` (any other path → 404).
- Injects the selected provider's API key as a Bearer token
  (the client's own key is ignored, so any dummy value works client-side).
- Rewrites `model` to the configured model in `openai` mode, or to
  `emb://$YANDEX_FOLDER_ID/text-embeddings-v2-doc/latest` in `yandex` mode.
- Forces `encoding_format: "float"` in the outgoing body.
- Forwards `dimensions` if present. In `openai` mode it must be one of
  `256`, `512`, `768`, `1024`, `1536`, `2048`, `3072` or `4096`; in `yandex`
  mode it must be `256`, `512` or `768`. Invalid values are rejected with a
  `400`. Defaults are `1536` and `768`, respectively.
- **Batch processing:** an `input` array is split into sequential upstream calls
  containing at most `maxBatchStrings` strings each (default `60` in the checked-in
  configuration). A one-string
  chunk is sent as a string; larger chunks are sent as arrays. Responses are
  merged back into a single OpenAI-shaped response with re-indexed `data` and
  summed `usage`.
- **429 retry:** upstream rate-limit responses (`429`) are retried with
  exponential backoff (`1s → 2s → 4s → 8s → 16s → 32s → 60s`, capped); if a retry
  following the full 60s wait still returns `429`, the whole request fails with `429`.

## Requirements

- Node.js **18+** (uses built-in `fetch` and `AbortSignal.timeout`).
- An API key for the configured OpenAI-compatible provider. Yandex Cloud
  credentials are additionally required when `mode` is `yandex`.

## Setup

```bash
# 1. install dependencies
npm install

# 2. create .env (see .env.example) or export the vars
export OPENAI_API_KEY="your_routerai_key" # for mode=openai
# For mode=yandex, use these instead:
# export YANDEX_API_KEY="your_api_key"
# export YANDEX_FOLDER_ID="b1gxxxxxxxxxxxxxxxxx"
export PORT=9988   # optional, default 9988

# 3. run (the checked-in config uses openai mode)
npm start
```

> The proxy reads variables from the **process environment**. To load them from a
> `.env` file use `node --env-file=.env src/server.js` (Node 20.6+).

Startup validation checks `OPENAI_API_KEY` for `openai`, and `YANDEX_API_KEY` plus
`YANDEX_FOLDER_ID` for `yandex`.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `YANDEX_API_KEY` | yandex mode | — | Yandex Cloud API key (service account key or OAuth token) |
| `OPENAI_API_KEY` | openai mode | — | API key sent to the configured OpenAI-compatible endpoint |
| `YANDEX_FOLDER_ID` | yandex mode | — | Folder id used to build `emb://<id>/text-embeddings-v2-doc/latest` |
| `PORT` | — | `9988` | Port the proxy listens on |
| `YANDEX_BASE_URL` | — | `https://ai.api.cloud.yandex.net/v1` | Default upstream base URL when `mode` is `yandex`; `config.json.baseUrl` takes precedence |
| `YANDEX_TIMEOUT_MS` | — | `60000` | Upstream request timeout |

The `maxBatchStrings` setting in `config.json` must be a positive integer. It
controls the maximum number of input strings in each outgoing request.

## Dimensions

In `openai` mode, `dimensions` accepts `256`, `512`, `768`, `1024`, `1536`,
`2048`, `3072`, or `4096` and defaults to `1536`. In `yandex` mode, it accepts
`256`, `512`, or `768` and defaults to `768`.

## Usage

### cURL

```bash
# single string
curl http://localhost:9988/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"any-model-name","input":"hello world"}'

# batch (array of strings) — split according to maxBatchStrings, merged response
curl http://localhost:9988/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"any-model-name","input":["first text","second text","third text"]}'
```

### OpenAI JavaScript SDK

```js
import OpenAI from 'openai';

const client = new OpenAI({
   apiKey: 'any-dummy-key', // the proxy injects the configured provider key
  baseURL: 'http://localhost:9988/v1',
});

const res = await client.embeddings.create({
   model: 'any-model-name', // replaced with the configured model
  input: ['hello', 'world', 'from', 'the proxy'],
});

console.log(res.data.map((d) => d.index));      // [0, 1, 2, 3]
console.log(res.data[0].embedding.length);      // provider/model-dependent
console.log(res.usage.prompt_tokens);
```

### Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-dummy-key",          # the proxy injects the configured provider key
    base_url="http://localhost:9988/v1",
)

res = client.embeddings.create(
    model="any-model-name",           # replaced with the configured model
    input=["hello", "world"],
)

print([d.index for d in res.data])    # [0, 1]
```

## How batch processing works

Given `input: ["a", "b", "c"]` and `maxBatchStrings: 2`, the proxy:

1. Validates the body (strings only; token arrays are rejected with a 400).
2. Makes **2 upstream calls** to `/v1/embeddings`, with `input: ["a", "b"]` and
   then `input: "c"`, strictly sequentially (each call is awaited before the next starts).
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
| Unsupported `dimensions` for the selected mode | `400` OpenAI-style error |
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
- `dimensions` is mode-dependent: `openai` accepts `256`, `512`, `768`, `1024`,
  `1536`, `2048`, `3072`, or `4096` and defaults to `1536`; `yandex` accepts
  `256`, `512`, or `768` and defaults to `768`.
- The returned vector dimension is determined by the selected provider/model and
  the requested `dimensions` value.
