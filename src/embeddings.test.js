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
  callUpstreamWithRetry,
  createSuccessStats,
  flushUpstreamSuccess,
  localTime,
  logError,
  logInfo,
  mergeResponses,
  normalizeDimensions,
  normalizeInput,
  proxyEmbeddings,
  recordUpstreamSuccess,
  UpstreamError,
  UpstreamNetworkError,
  UpstreamResponseError,
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
  assert.equal(config.timeoutMs, 60000);
  assert.equal(config.maxBatchStrings, 1);
});

test('buildConfig honors overrides and trims trailing slash', () => {
  const config = buildConfig({
    YANDEX_API_KEY: 'k',
    YANDEX_FOLDER_ID: 'f',
    YANDEX_BASE_URL: 'https://example.com/v1/',
    YANDEX_TIMEOUT_MS: '5000',
  });
  assert.equal(config.baseUrl, 'https://example.com/v1');
  assert.equal(config.timeoutMs, 5000);
});

test('buildConfig supports openai mode from file config', () => {
  const config = buildConfig(
    { OPENAI_API_KEY: ' openai-key ', YANDEX_TIMEOUT_MS: '5000' },
    { mode: 'openai', baseUrl: 'https://routerai.ru/api/v1/', model: 'qwen/qwen3-embedding-8b' }
  );
  assert.equal(config.mode, 'openai');
  assert.equal(config.apiKey, 'openai-key');
  assert.equal(config.baseUrl, 'https://routerai.ru/api/v1');
  assert.equal(config.model, 'qwen/qwen3-embedding-8b');
  assert.equal(config.timeoutMs, 5000);
});

test('buildConfig reads a positive maxBatchStrings value from file config', () => {
  const config = buildConfig(
    { OPENAI_API_KEY: 'key' },
    { mode: 'openai', model: 'm', maxBatchStrings: 3 }
  );
  assert.equal(config.maxBatchStrings, 3);
});

test('buildConfig rejects invalid maxBatchStrings values', () => {
  for (const maxBatchStrings of [0, -1, 1.5, '2', null, true]) {
    assert.throws(
      () => buildConfig(
        { OPENAI_API_KEY: 'key' },
        { mode: 'openai', model: 'm', maxBatchStrings }
      ),
      /maxBatchStrings must be a positive integer/
    );
  }
});

test('buildConfig in openai mode requires OPENAI_API_KEY (no TypeError on missing key)', () => {
  // Regression: the key expression previously parsed as
  // `mode === 'openai' ? env.OPENAI_API_KEY : (env.YANDEX_API_KEY || '')`,
  // so openai mode lost the fallback and .trim() threw a TypeError when the
  // key was missing, killing startup before the friendly check below ran.
  assert.throws(
    () => buildConfig({}, { mode: 'openai', model: 'm' }),
    /OPENAI_API_KEY is required/
  );
  // A present YANDEX_API_KEY must not satisfy the openai-mode requirement.
  assert.throws(
    () => buildConfig({ YANDEX_API_KEY: 'yandex-key' }, { mode: 'openai', model: 'm' }),
    /OPENAI_API_KEY is required/
  );
});

test('buildConfig in openai mode trims and rejects whitespace-only OPENAI_API_KEY', () => {
  assert.throws(
    () => buildConfig({ OPENAI_API_KEY: '   ' }, { mode: 'openai', model: 'm' }),
    /OPENAI_API_KEY is required/
  );
});

test('yandexModel builds the Yandex document model id', () => {
  assert.equal(
    yandexModel('b1g7abc'),
    'emb://b1g7abc/text-embeddings-v2-doc/latest'
  );
});

// ---------------------------------------------------------------------------
// localTime / logInfo / logError (logging helpers)
// ---------------------------------------------------------------------------

test('localTime formats HH:mm:ss with zero padding', () => {
  const date = new Date(2026, 0, 5, 9, 7, 3); // local timezone
  assert.equal(localTime(date), '09:07:03');
});

test('localTime pads single-digit hours, minutes and seconds', () => {
  const date = new Date(2026, 5, 15, 3, 4, 5); // local timezone
  assert.equal(localTime(date), '03:04:05');
});

test('localTime defaults to the current local time', () => {
  const expected = new Date();
  const got = localTime();
  const pad = (n) => String(n).padStart(2, '0');
  assert.equal(
    got,
    `${pad(expected.getHours())}:${pad(expected.getMinutes())}:${pad(expected.getSeconds())}`
  );
});

test('logInfo inserts a local HH:mm:ss timestamp after the prefix', () => {
  const originalLog = console.log;
  const lines = [];
  console.log = (msg, ...args) => lines.push([msg, ...args]);
  try {
    logInfo('[yandex-proxy] request: 1 input string(s)');
  } finally {
    console.log = originalLog;
  }
  assert.equal(lines.length, 1);
  assert.match(lines[0][0], /^\[yandex-proxy\] \d{2}:\d{2}:\d{2} request: 1 input string\(s\)$/);
});

test('logInfo normalizes a message that already carries the prefix', () => {
  const originalLog = console.log;
  const lines = [];
  console.log = (msg, ...args) => lines.push([msg, ...args]);
  try {
    logInfo('[yandex-proxy] listening on http://localhost:9988');
  } finally {
    console.log = originalLog;
  }
  // The prefix appears exactly once, followed by the timestamp.
  assert.equal((lines[0][0].match(/\[yandex-proxy\]/g) || []).length, 1);
  assert.match(lines[0][0], /^\[yandex-proxy\] \d{2}:\d{2}:\d{2} listening on http:\/\/localhost:9988$/);
});

test('logError forwards extra args to console.error', () => {
  const originalError = console.error;
  const lines = [];
  const sentinel = new Error('boom');
  console.error = (msg, ...args) => lines.push([msg, ...args]);
  try {
    logError('[yandex-proxy] unexpected error:', sentinel);
  } finally {
    console.error = originalError;
  }
  assert.equal(lines.length, 1);
  assert.match(lines[0][0], /^\[yandex-proxy\] \d{2}:\d{2}:\d{2} unexpected error:$/);
  assert.equal(lines[0][1], sentinel);
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
// normalizeDimensions
// ---------------------------------------------------------------------------

test('normalizeDimensions defaults to 768 when absent or null', () => {
  assert.equal(normalizeDimensions(undefined), 768);
  assert.equal(normalizeDimensions(null), 768);
});

test('normalizeDimensions accepts 256, 512 and 768', () => {
  assert.equal(normalizeDimensions(256), 256);
  assert.equal(normalizeDimensions(512), 512);
  assert.equal(normalizeDimensions(768), 768);
});

test('normalizeDimensions rejects unsupported or non-integer values', () => {
  for (const bad of [0, 128, 300, 1024, 256.5, NaN, '512', true, [], {}]) {
    assert.throws(() => normalizeDimensions(bad), ValidationError);
  }
});

test('normalizeDimensions errors carry the param "dimensions"', () => {
  try {
    normalizeDimensions(300);
    assert.fail('expected ValidationError');
  } catch (err) {
    assert.ok(err instanceof ValidationError);
    assert.equal(err.param, 'dimensions');
  }
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

test('buildUpstreamBody injects dimensions 768 by default', () => {
  const body = buildUpstreamBody({ model: 'x', input: 'a' }, 'f');
  assert.equal(body.dimensions, 768);
});

test('buildUpstreamBody uses the OpenAI model and defaults dimensions to 1536', () => {
  const body = buildUpstreamBody(
    { model: 'client-model', input: 'a' },
    { mode: 'openai', model: 'qwen/qwen3-embedding-8b' }
  );
  assert.equal(body.model, 'qwen/qwen3-embedding-8b');
  assert.equal(body.dimensions, 1536);
  assert.equal(body.encoding_format, 'float');
});

test('buildUpstreamBody accepts OpenAI dimensions and rejects unsupported values', () => {
  for (const dimensions of [256, 512, 768, 1024, 1536, 2048, 3072, 4096]) {
    assert.equal(
      buildUpstreamBody({ input: 'a', dimensions }, { mode: 'openai', model: 'm' }).dimensions,
      dimensions
    );
  }
  assert.throws(
    () => buildUpstreamBody({ input: 'a', dimensions: 5000 }, { mode: 'openai', model: 'm' }),
    (err) => err instanceof ValidationError && err.param === 'dimensions'
  );
});

test('buildUpstreamBody forwards a valid client dimensions value', () => {
  assert.equal(buildUpstreamBody({ input: 'a', dimensions: 256 }, 'f').dimensions, 256);
  assert.equal(buildUpstreamBody({ input: 'a', dimensions: 512 }, 'f').dimensions, 512);
});

test('buildUpstreamBody rejects an unsupported dimensions value', () => {
  assert.throws(
    () => buildUpstreamBody({ model: 'x', input: 'a', dimensions: 300 }, 'f'),
    (err) => err instanceof ValidationError && err.param === 'dimensions'
  );
  assert.throws(
    () => buildUpstreamBody({ model: 'x', input: 'a', dimensions: '512' }, 'f'),
    ValidationError
  );
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
    return jsonResponse(200, { data: [{}], usage: {} });
  };

  const config = {
    baseUrl: 'https://ai.api.cloud.yandex.net/v1',
    apiKey: 'secret-key',
    timeoutMs: 30000,
  };

  await callUpstream({ model: 'm', input: 'x' }, config, null, fakeFetch);

  assert.equal(captured.url, 'https://ai.api.cloud.yandex.net/v1/embeddings');
  assert.equal(captured.opts.method, 'POST');
  assert.equal(captured.opts.headers['Content-Type'], 'application/json');
  assert.equal(captured.opts.headers.Authorization, 'Bearer secret-key');
  assert.deepEqual(JSON.parse(captured.opts.body), { model: 'm', input: 'x' });
});

test('callUpstream sends OpenAI-compatible endpoint credentials', async () => {
  let captured;
  const fakeFetch = async (url, opts) => {
    captured = { url, opts };
    return jsonResponse(200, { data: [{}], usage: {} });
  };

  await callUpstream(
    { model: 'qwen/qwen3-embedding-8b', input: 'x' },
    { mode: 'openai', baseUrl: 'https://routerai.ru/api/v1', apiKey: 'secret', timeoutMs: 1000 },
    null,
    fakeFetch
  );

  assert.equal(captured.url, 'https://routerai.ru/api/v1/embeddings');
  assert.equal(captured.opts.headers.Authorization, 'Bearer secret');
  assert.deepEqual(JSON.parse(captured.opts.body), {
    model: 'qwen/qwen3-embedding-8b',
    input: 'x',
  });
});

test('callUpstream throws UpstreamError with status and body on HTTP error', async () => {
  const fakeFetch = async () =>
    jsonResponse(429, { error: { message: 'rate limited' } });
  await assert.rejects(
    () =>
      callUpstream({}, { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 }, null, fakeFetch),
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
      callUpstream({}, { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 }, null, fakeFetch),
    UpstreamNetworkError
  );
});

test('callUpstream rejects a response with the wrong embedding count', async () => {
  await assert.rejects(
    () => callUpstream({ input: ['a', 'b'] }, { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 }, null,
      async () => jsonResponse(200, { data: [{}], usage: {} })),
    (err) => err instanceof UpstreamResponseError && err.expected === 2 && err.actual === 1
  );
});

// ---------------------------------------------------------------------------
// callUpstreamWithRetry (429 backoff)
// ---------------------------------------------------------------------------

test('callUpstreamWithRetry retries 429 with exponential backoff then succeeds', async () => {
  let calls = 0;
  const delays = [];
  const fakeFetch = async () => {
    calls += 1;
    if (calls <= 2) {
      return jsonResponse(429, { error: { message: 'rate limited' } });
    }
    return jsonResponse(200, { data: [{}], usage: {} });
  };
  const sleepImpl = async (ms) => {
    delays.push(ms);
  };

  const result = await callUpstreamWithRetry(
    { model: 'm', input: 'x' },
    { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 },
    null,
    fakeFetch,
    sleepImpl
  );

  assert.equal(calls, 3);
  assert.deepEqual(delays, [1000, 2000]);
  assert.deepEqual(result, { data: [{}], usage: {} });
});

test('callUpstreamWithRetry gives up with 429 after 1s,2s,4s,...,60s schedule', async () => {
  const delays = [];
  let calls = 0;
  const fakeFetch = async () => {
    calls += 1;
    return jsonResponse(429, { error: { message: 'still limited' } });
  };
  const sleepImpl = async (ms) => {
    delays.push(ms);
  };

  await assert.rejects(
    () =>
      callUpstreamWithRetry(
        { model: 'm', input: 'x' },
        { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 },
        null,
        fakeFetch,
        sleepImpl
      ),
    (err) => err instanceof UpstreamError && err.status === 429
  );

  // 7 sleeps (1s, 2s, 4s, 8s, 16s, 32s, 60s), then the 8th call's 429 throws.
  assert.deepEqual(delays, [1000, 2000, 4000, 8000, 16000, 32000, 60000]);
  assert.equal(calls, 8);
});

test('callUpstreamWithRetry does not retry non-429 upstream errors', async () => {
  let calls = 0;
  const fakeFetch = async () => {
    calls += 1;
    return jsonResponse(500, { error: { message: 'boom' } });
  };
  const sleepImpl = async () => {
    assert.fail('should not sleep for a non-429 error');
  };

  await assert.rejects(
    () =>
      callUpstreamWithRetry(
        { model: 'm', input: 'x' },
        { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 },
        null,
        fakeFetch,
        sleepImpl
      ),
    (err) => err instanceof UpstreamError && err.status === 500
  );
  assert.equal(calls, 1);
});

// ---------------------------------------------------------------------------
// success statistics (per-request accumulator, 1s window aggregation)
// ---------------------------------------------------------------------------

test('recordUpstreamSuccess opens a window on first success without logging', () => {
  const stats = createSuccessStats();
  const logs = [];
  recordUpstreamSuccess(stats, 1000, (m) => logs.push(m));
  recordUpstreamSuccess(stats, 1100, (m) => logs.push(m));
  assert.deepEqual(logs, []);
});

test('recordUpstreamSuccess logs the total and resets once the 1s window elapses', () => {
  const stats = createSuccessStats();
  const logs = [];
  recordUpstreamSuccess(stats, 1000, (m) => logs.push(m));
  recordUpstreamSuccess(stats, 1100, (m) => logs.push(m));
  recordUpstreamSuccess(stats, 1200, (m) => logs.push(m)); // 200ms — still inside window
  recordUpstreamSuccess(stats, 2500, (m) => logs.push(m)); // 1500ms — window elapsed
  assert.equal(logs.length, 1);
  assert.equal(logs[0], '[yandex-proxy] yandex ok: 3 successfully tokenized string(s) in 1500ms');
});

test('flushUpstreamSuccess prints the pending count and resets the window', () => {
  const stats = createSuccessStats();
  const logs = [];
  recordUpstreamSuccess(stats, 1000, () => {});
  recordUpstreamSuccess(stats, 1100, () => {});
  flushUpstreamSuccess(stats, 1500, (m) => logs.push(m));
  assert.equal(logs.length, 1);
  assert.equal(logs[0], '[yandex-proxy] yandex ok: 2 successfully tokenized string(s) in 500ms');

  // accumulator is fully reset — the next flush prints nothing
  const moreLogs = [];
  flushUpstreamSuccess(stats, 2000, (m) => moreLogs.push(m));
  assert.deepEqual(moreLogs, []);
});

test('flushUpstreamSuccess with zero count prints nothing', () => {
  const stats = createSuccessStats();
  const logs = [];
  flushUpstreamSuccess(stats, 5000, (m) => logs.push(m));
  assert.deepEqual(logs, []);
});

test('per-request accumulators are isolated from each other', () => {
  const statsA = createSuccessStats();
  const statsB = createSuccessStats();
  const logsA = [];
  const logsB = [];

  recordUpstreamSuccess(statsA, 1000, (m) => logsA.push(m));
  recordUpstreamSuccess(statsB, 1100, (m) => logsB.push(m));
  recordUpstreamSuccess(statsA, 1200, (m) => logsA.push(m));
  recordUpstreamSuccess(statsB, 1300, (m) => logsB.push(m));

  // Each accumulator counts only its own successes.
  flushUpstreamSuccess(statsA, 1500, (m) => logsA.push(m));
  flushUpstreamSuccess(statsB, 1500, (m) => logsB.push(m));
  assert.deepEqual(logsA, ['[yandex-proxy] yandex ok: 2 successfully tokenized string(s) in 500ms']);
  assert.deepEqual(logsB, ['[yandex-proxy] yandex ok: 2 successfully tokenized string(s) in 400ms']);
});

test('callUpstream accumulates 200s silently and keeps the status log for non-200s', async () => {
  const stats = createSuccessStats();
  const logs = [];
  const originalLog = console.log;
  console.log = (m) => logs.push(m);
  try {
    await callUpstream(
      { model: 'm', input: 'x' },
      { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 },
      stats,
      async () => jsonResponse(200, { data: [{}], usage: {} })
    );
    assert.deepEqual(logs, []);

    await assert.rejects(
      () =>
        callUpstream(
          {},
          { baseUrl: 'b', apiKey: 'k', timeoutMs: 1000 },
          stats,
          async () => jsonResponse(500, { error: { message: 'boom' } })
        ),
      (err) => err instanceof UpstreamError && err.status === 500
    );
    assert.equal(logs.length, 1);
    assert.match(
      logs[0],
       /^\[yandex-proxy\] \d{2}:\d{2}:\d{2} yandex status: 500$/
    );

    // The 200 was accumulated, not logged — the flush reports it exactly once
    // (window opened at Date.now(); flush uses that same real clock, so the
    // elapsed time is whatever passed since the 200 — always > 0).
    const flushLogs = [];
    flushUpstreamSuccess(stats, Date.now(), (m) => flushLogs.push(m));
    assert.equal(flushLogs.length, 1);
    assert.match(flushLogs[0], /^\[yandex-proxy\] yandex ok: 1 successfully tokenized string\(s\) in \d+ms$/);
  } finally {
    console.log = originalLog;
  }
});

test('proxyEmbeddings flushes pending success stats on completion', async () => {
  const logs = [];
  const originalLog = console.log;
  console.log = (m) => logs.push(m);
  try {
    const fakeFetch = async (_url, opts) => {
      const body = JSON.parse(opts.body);
      return jsonResponse(200, {
        data: [{ object: 'embedding', embedding: [1], index: 0 }],
        usage: { prompt_tokens: 1, total_tokens: 1 },
        model: body.model,
      });
    };
    await proxyEmbeddings(
      { model: 'x', input: ['a', 'b', 'c'] },
      { baseUrl: 'b', apiKey: 'k', folderId: 'f', timeoutMs: 1000 },
      fakeFetch
    );
    // All three 200s are aggregated and reported (exactly once each) by the
    // end-of-request flush, no matter whether the window elapsed mid-batch.
    const counts = logs.map((m) => Number((m.match(/yandex ok: (\d+)/) || [])[1] ?? 0));
    assert.equal(counts.reduce((a, b) => a + b, 0), 3);
  } finally {
    console.log = originalLog;
  }
});

test('proxyEmbeddings flushes pending stats even when the request fails', async () => {
  const logs = [];
  const originalLog = console.log;
  console.log = (m) => logs.push(m);
  try {
    let calls = 0;
    const fakeFetch = async (_url, _opts) => {
      calls += 1;
      if (calls === 1) {
        return jsonResponse(200, {
          data: [{ object: 'embedding', embedding: [1], index: 0 }],
          usage: { prompt_tokens: 1, total_tokens: 1 },
          model: 'm',
        });
      }
      return jsonResponse(500, { error: { message: 'boom' } });
    };

    await assert.rejects(
      () =>
        proxyEmbeddings(
          { model: 'x', input: ['a', 'b'] },
          { baseUrl: 'b', apiKey: 'k', folderId: 'f', timeoutMs: 1000 },
          fakeFetch
        ),
      (err) => err instanceof UpstreamError && err.status === 500
    );
    // The single success before the failure is still reported by the flush.
    const counts = logs.map((m) => Number((m.match(/yandex ok: (\d+)/) || [])[1] ?? 0));
    assert.equal(counts.reduce((a, b) => a + b, 0), 1);
  } finally {
    console.log = originalLog;
  }
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
// proxyEmbeddings (integration of split + sequential loop + merge with fake fetch)
// ---------------------------------------------------------------------------

test('proxyEmbeddings processes array input sequentially and merges results', async () => {
  const calls = [];
  const seenDimensions = [];
  const fakeFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    calls.push(body.input);
    seenDimensions.push(body.dimensions);
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
    timeoutMs: 1000,
  };

  const result = await proxyEmbeddings(
    { model: 'any', input: ['ab', 'cd', 'ef'] },
    config,
    fakeFetch
  );

  // one upstream call per input string, each carrying the default dimensions
  assert.deepEqual(calls, ['ab', 'cd', 'ef']);
  assert.deepEqual(seenDimensions, [768, 768, 768]);
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

test('proxyEmbeddings splits input into configured batches and sends singleton tail as string', async () => {
  const calls = [];
  const fakeFetch = async (_url, opts) => {
    const body = JSON.parse(opts.body);
    calls.push(body.input);
    const items = Array.isArray(body.input) ? body.input : [body.input];
    return jsonResponse(200, {
      data: items.map((text, index) => ({
        object: 'embedding',
        embedding: [text],
        index,
      })),
      usage: { prompt_tokens: items.length, total_tokens: items.length },
      model: body.model,
    });
  };

  const result = await proxyEmbeddings(
    { model: 'x', input: ['a', 'b', 'c', 'd', 'e'] },
    {
      baseUrl: 'https://b',
      apiKey: 'k',
      folderId: 'f',
      timeoutMs: 1000,
      maxBatchStrings: 2,
    },
    fakeFetch
  );

  assert.deepEqual(calls, [['a', 'b'], ['c', 'd'], 'e']);
  assert.deepEqual(result.data.map((item) => item.embedding), [['a'], ['b'], ['c'], ['d'], ['e']]);
  assert.deepEqual(result.data.map((item) => item.index), [0, 1, 2, 3, 4]);
  assert.equal(result.usage.prompt_tokens, 5);
  assert.equal(result.usage.total_tokens, 5);
});

test('proxyEmbeddings makes strictly sequential upstream calls (no overlap)', async () => {
  let active = 0;
  let maxActive = 0;
  const callOrder = [];
  const fakeFetch = async (_url, opts) => {
    const body = JSON.parse(opts.body);
    active += 1;
    maxActive = Math.max(maxActive, active);
    callOrder.push(body.input);
    await new Promise((r) => setTimeout(r, 5));
    active -= 1;
    return jsonResponse(200, {
      data: [{ object: 'embedding', embedding: [1], index: 0 }],
      usage: { prompt_tokens: 1, total_tokens: 1 },
      model: body.model,
    });
  };

  const config = {
    baseUrl: 'https://b',
    apiKey: 'k',
    folderId: 'f',
    timeoutMs: 1000,
  };

  const result = await proxyEmbeddings(
    { model: 'x', input: ['one', 'two', 'three'] },
    config,
    fakeFetch
  );

  assert.equal(callOrder.length, 3);
  assert.deepEqual(callOrder, ['one', 'two', 'three']);
  assert.equal(maxActive, 1, 'upstream calls must never overlap');
  assert.equal(result.data.length, 3);
});

test('proxyEmbeddings forwards a single string unchanged (single upstream call)', async () => {
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
    timeoutMs: 1000,
  };

  await assert.rejects(
    () => proxyEmbeddings({ model: 'x', input: ['a', 'b'] }, config, fakeFetch),
    (err) => err instanceof UpstreamError && err.status === 500
  );
});

test('proxyEmbeddings retries a mismatched batch, then falls back to sequential singletons', async () => {
  const calls = [];
  const fakeFetch = async (_url, opts) => {
    const input = JSON.parse(opts.body).input;
    calls.push(input);
    if (calls.length <= 2) return jsonResponse(200, { data: [{}], usage: {} });
    const text = Array.isArray(input) ? input[0] : input;
    return jsonResponse(200, { data: [{ embedding: [text] }], usage: { total_tokens: 1 } });
  };
  const result = await proxyEmbeddings(
    { input: ['a', 'b'] },
    { baseUrl: 'b', apiKey: 'k', folderId: 'f', timeoutMs: 1000, maxBatchStrings: 2 },
    fakeFetch
  );
  assert.deepEqual(calls, [['a', 'b'], ['a', 'b'], 'a', 'b']);
  assert.deepEqual(result.data.map((item) => item.embedding), [['a'], ['b']]);
});

test('proxyEmbeddings fails after a singleton cardinality mismatch', async () => {
  let calls = 0;
  await assert.rejects(
    () => proxyEmbeddings(
      { input: ['a', 'b'] },
      { baseUrl: 'b', apiKey: 'k', folderId: 'f', timeoutMs: 1000, maxBatchStrings: 2 },
      async () => {
        calls += 1;
        return jsonResponse(200, { data: [], usage: {} });
      }
    ),
    UpstreamResponseError
  );
  assert.equal(calls, 4);
});
