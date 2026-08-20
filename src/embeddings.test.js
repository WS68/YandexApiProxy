/**
 * Unit tests for the pure proxy logic in embeddings.js.
 * Runs with `node --test src/` (no extra dependencies).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  buildConfig,
  buildUpstreamBody,
  callUpstream,
  mapWithConcurrency,
  mergeResponses,
  normalizeInput,
  proxyEmbeddings,
  UpstreamError,
  UpstreamNetworkError,
  ValidationError,
  yandexModel,
} from './embeddings.js';

// ---------------------------------------------------------------------------
// buildConfig
// ---------------------------------------------------------------------------

test('buildConfig validates required env vars', () => {
  assert.throws(() => buildConfig({}), /YANDEX_API_KEY is required/);
  assert.throws(
    () => buildConfig({ YANDEX_API_KEY: 'key' }),
    /YANDEX_FOLDER_ID is required/
  );
});

test('buildConfig reads env vars and applies defaults', () => {
  const config = buildConfig({
    YANDEX_API_KEY: ' test-key ',
    YANDEX_FOLDER_ID: ' folder-1 ',
  });
  assert.equal(config.apiKey, 'test-key');
  assert.equal(config.folderId, 'folder-1');
  assert.equal(config.baseUrl, 'https://ai.api.cloud.yandex.net/v1');
  assert.equal(config.concurrency, 4);
  assert.equal(config.timeoutMs, 60000);
});

test('buildConfig honors overrides and trims trailing slash', () => {
  const config = buildConfig({
    YANDEX_API_KEY: 'k',
    YANDEX_FOLDER_ID: 'f',
    YANDEX_BASE_URL: 'https://example.com/v1/',
    YANDEX_CONCURRENCY: '8',
    YANDEX_TIMEOUT_MS: '5000',
  });
  assert.equal(config.baseUrl, 'https://example.com/v1');
  assert.equal(config.concurrency, 8);
  assert.equal(config.timeoutMs, 5000);
});

test('yandexModel builds the Yandex document model id', () => {
  assert.equal(
    yandexModel('b1g7abc'),
    'emb://b1g7abc/text-embeddings-v2-doc/latest'
  );
});

// ---------------------------------------------------------------------------
// normalizeInput
// ---------------------------------------------------------------------------

test('normalizeInput accepts a single string', () => {
  assert.deepEqual(normalizeInput('hello'), ['hello']);
});

test('normalizeInput accepts a string array', () => {
  assert.deepEqual(normalizeInput(['a', 'b', 'c']), ['a', 'b', 'c']);
});

test('normalizeInput rejects non-string/non-array input', () => {
  assert.throws(() => normalizeInput(42), ValidationError);
  assert.throws(() => normalizeInput(null), ValidationError);
  assert.throws(() => normalizeInput({}), ValidationError);
});

test('normalizeInput rejects empty array and empty strings', () => {
  assert.throws(() => normalizeInput([]), ValidationError);
  assert.throws(() => normalizeInput(['  ']), ValidationError);
  assert.throws(() => normalizeInput('   '), ValidationError);
});

test('normalizeInput rejects token arrays (numbers)', () => {
  assert.throws(() => normalizeInput([[1, 2, 3]]), ValidationError);
  assert.throws(() => normalizeInput(['a', [1, 2]]), ValidationError);
});

// ---------------------------------------------------------------------------
// buildUpstreamBody
// ---------------------------------------------------------------------------

test('buildUpstreamBody rewrites model, forces encoding_format float, keeps input', () => {
  const body = buildUpstreamBody(
    { model: 'text-embedding-3-small', input: 'hi', user: 'u1' },
    'folder-1'
  );
  assert.equal(body.model, 'emb://folder-1/text-embeddings-v2-doc/latest');
  assert.equal(body.encoding_format, 'float');
  assert.equal(body.input, 'hi');
  assert.equal(body.user, 'u1');
});

test('buildUpstreamBody overrides a client-provided encoding_format', () => {
  const body = buildUpstreamBody(
    { model: 'x', input: ['a'], encoding_format: 'base64' },
    'f'
  );
  assert.equal(body.encoding_format, 'float');
});

test('buildUpstreamBody rejects missing input', () => {
  assert.throws(() => buildUpstreamBody({ model: 'x' }, 'f'), ValidationError);
  assert.throws(() => buildUpstreamBody(null, 'f'), ValidationError);
});

// ---------------------------------------------------------------------------
// callUpstream
// ---------------------------------------------------------------------------

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  };
}

test('callUpstream sends the correct URL, headers and body', async () => {
  let captured;
  const fakeFetch = async (url, opts) => {
    captured = { url, opts };
    return jsonResponse(200, { data: [], usage: {} });
  };

  const config = {
    baseUrl: 'https://ai.api.cloud.yandex.net/v1',
    apiKey: 'secret-key',
    timeoutMs: 30000,
  };

  await callUpstream({ model: 'm', input: 'x' }, config, fakeFetch);

  assert.equal(captured.url, 'https://ai.api.cloud.yandex.net/v1/embeddings');
  assert.equal(captured.opts.method, 'POST');
  assert.equal(captured.opts.headers['Content-Type'], 'application/json');
  assert.equal(captured.opts.headers.Authorization, 'Bearer secret-key');
  assert.deepEqual(JSON.parse(captured.opts.body), { model: 'm', input: 'x' });
});

test('callUpstream throws UpstreamError with status and body on HTTP error', async () => {
  const fakeFetch = async () =>
    jsonResponse(429, { error: { message: 'rate limited' } });
  await assert.rejects(
    () =>
      callUpstream({}, { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 }, fakeFetch),
    (err) =>
      err instanceof UpstreamError &&
      err.status === 429 &&
      err.body.error.message === 'rate limited'
  );
});

test('callUpstream throws UpstreamNetworkError when fetch itself fails', async () => {
  const fakeFetch = async () => {
    throw new TypeError('fetch failed', { cause: new Error('ECONNREFUSED') });
  };
  await assert.rejects(
    () =>
      callUpstream({}, { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 }, fakeFetch),
    UpstreamNetworkError
  );
});

// ---------------------------------------------------------------------------
// mapWithConcurrency
// ---------------------------------------------------------------------------

test('mapWithConcurrency preserves order and bounds concurrency', async () => {
  let active = 0;
  let maxActive = 0;
  const seen = [];

  const fn = async (item, index) => {
    active += 1;
    maxActive = Math.max(maxActive, active);
    await new Promise((r) => setTimeout(r, 5));
    active -= 1;
    seen.push(index);
    return item.toUpperCase();
  };

  const results = await mapWithConcurrency(
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    3,
    fn
  );

  assert.deepEqual(results, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);
  assert.equal(maxActive, 3);
});

// ---------------------------------------------------------------------------
// mergeResponses
// ---------------------------------------------------------------------------

test('mergeResponses re-indexes data and sums usage', () => {
  const merged = mergeResponses([
    {
      data: [{ object: 'embedding', embedding: [0.1], index: 0 }],
      usage: { prompt_tokens: 5, total_tokens: 5 },
      model: 'm',
    },
    {
      data: [{ object: 'embedding', embedding: [0.2], index: 0 }],
      usage: { prompt_tokens: 7, total_tokens: 7 },
      model: 'm',
    },
    {
      data: [{ object: 'embedding', embedding: [0.3], index: 0 }],
      usage: { prompt_tokens: 3, total_tokens: 3 },
      model: 'm',
    },
  ]);

  assert.equal(merged.object, 'list');
  assert.equal(merged.model, 'm');
  assert.equal(merged.data.length, 3);
  assert.deepEqual(
    merged.data.map((d) => d.index),
    [0, 1, 2]
  );
  assert.equal(merged.usage.prompt_tokens, 15);
  assert.equal(merged.usage.total_tokens, 15);
  assert.equal(merged.usage.completion_tokens, undefined);
});

test('mergeResponses sums completion_tokens when present', () => {
  const merged = mergeResponses([
    { data: [{}], usage: { completion_tokens: 2 } },
    { data: [{}], usage: { completion_tokens: 3 } },
  ]);
  assert.equal(merged.usage.completion_tokens, 5);
});

test('mergeResponses throws on empty input', () => {
  assert.throws(() => mergeResponses([]), /at least one response/);
});

// ---------------------------------------------------------------------------
// proxyEmbeddings (integration of split + fan-out + merge with fake fetch)
// ---------------------------------------------------------------------------

test('proxyEmbeddings fans out array input and merges results', async () => {
  const calls = [];
  const fakeFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    calls.push(body.input);
    return jsonResponse(200, {
      data: [{ object: 'embedding', embedding: [body.input.length], index: 0 }],
      usage: { prompt_tokens: body.input.length, total_tokens: body.input.length },
      model: body.model,
    });
  };

  const config = {
    baseUrl: 'https://ai.api.cloud.yandex.net/v1',
    apiKey: 'k',
    folderId: 'folder-9',
    concurrency: 2,
    timeoutMs: 1000,
  };

  const result = await proxyEmbeddings(
    { model: 'any', input: ['ab', 'cd', 'ef'] },
    config,
    fakeFetch
  );

  // one upstream call per input string
  assert.deepEqual(calls, ['ab', 'cd', 'ef']);
  // merged response
  assert.equal(result.data.length, 3);
  assert.deepEqual(
    result.data.map((d) => d.index),
    [0, 1, 2]
  );
  assert.deepEqual(
    result.data.map((d) => d.embedding),
    [[2], [2], [2]]
  );
  assert.equal(result.usage.prompt_tokens, 6);
  assert.equal(result.usage.total_tokens, 6);
  // every upstream call carried the rewritten model
  assert.equal(result.model, 'emb://folder-9/text-embeddings-v2-doc/latest');
});

test('proxyEmbeddings forwards a single string unchanged (no fan-out)', async () => {
  let callCount = 0;
  let sentBody = null;
  const fakeFetch = async (_url, opts) => {
    callCount += 1;
    sentBody = JSON.parse(opts.body);
    return jsonResponse(200, {
      data: [{ object: 'embedding', embedding: [0.5], index: 0 }],
      usage: { prompt_tokens: 1, total_tokens: 1 },
      model: 'm',
    });
  };

  const config = {
    baseUrl: 'https://b',
    apiKey: 'k',
    folderId: 'f',
    concurrency: 4,
    timeoutMs: 1000,
  };

  const result = await proxyEmbeddings({ model: 'x', input: 'solo' }, config, fakeFetch);
  assert.equal(callCount, 1);
  assert.equal(sentBody.input, 'solo');
  assert.equal(sentBody.encoding_format, 'float');
  assert.equal(result.data.length, 1);
});

test('proxyEmbeddings propagates a single upstream failure', async () => {
  const fakeFetch = async () =>
    jsonResponse(500, { error: { message: 'yandex exploded' } });

  const config = {
    baseUrl: 'https://b',
    apiKey: 'k',
    folderId: 'f',
    concurrency: 2,
    timeoutMs: 1000,
  };

  await assert.rejects(
    () => proxyEmbeddings({ model: 'x', input: ['a', 'b'] }, config, fakeFetch),
    (err) => err instanceof UpstreamError && err.status === 500
  );
});
