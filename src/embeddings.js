/**
 * Core proxy logic for OpenAI-compatible embeddings requests.
 *
 * Responsibilities:
 *  - Rewrite the client's embeddings request into a Yandex AI request:
 *      * model  -> emb://{folderId}/text-embeddings-v2-doc/latest
 *      * encoding_format -> "float"
 *      * dimensions -> 768 (default) or client-provided 256/512/768
 *  - Sequentially process batched inputs (array of strings): one upstream call
 *    per string, one at a time, in order.
 *  - Merge per-item responses into a single OpenAI-shaped response.
 *  - Retry upstream HTTP 429 responses with exponential backoff (1s, doubling,
 *    capped at 60s); if a retry following a full 60s wait still returns 429,
 *    the whole request fails with 429.
 *
 * The module is pure and framework-agnostic: `fetch` is injected so the
 * merge/split logic can be unit tested without real network calls.
 */

export class UpstreamError extends Error {
  constructor(status, body) {
    super(`Upstream request failed with status ${status}`);
    this.name = 'UpstreamError';
    this.status = status;
    this.body = body;
  }
}

export class UpstreamNetworkError extends Error {
  constructor(message) {
    super(message);
    this.name = 'UpstreamNetworkError';
  }
}

export class ValidationError extends Error {
  constructor(message, param = null) {
    super(message);
    this.name = 'ValidationError';
    this.param = param;
  }
}

export class UpstreamResponseError extends Error {
  constructor(expected, actual) {
    super(`Upstream response returned ${actual} embedding(s) for ${expected} input string(s)`);
    this.name = 'UpstreamResponseError';
    this.expected = expected;
    this.actual = actual;
  }
}

/**
 * Formats a date as HH:mm:ss (24-hour, zero-padded) using the process's local
 * timezone. Shared by every log line so all timestamps use the same format.
 * Pure and injectable so it can be tested deterministically.
 * @param {Date} [date=new Date()] instant to format
 * @returns {string} local time as HH:mm:ss
 */
export function localTime(date = new Date()) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/**
 * Prints an info log line with the local time inserted right after the
 * `[yandex-proxy]` prefix, e.g. `[yandex-proxy] 09:47:22 request: ...`. If the
 * message already carries the `[yandex-proxy]` prefix it is normalized so the
 * timestamp is inserted exactly once.
 * @param {string} message log message (prefix optional)
 * @param {...unknown} args extra values forwarded to console.log
 * @returns {void}
 */
export function logInfo(message, ...args) {
  const clean = String(message).replace(/^\[yandex-proxy\]\s*/, '');
  console.log(`[yandex-proxy] ${localTime()} ${clean}`, ...args);
}

/**
 * Prints an error log line with the local time inserted right after the
 * `[yandex-proxy]` prefix.
 * @param {string} message log message (prefix optional)
 * @param {...unknown} args extra values forwarded to console.error
 * @returns {void}
 */
export function logError(message, ...args) {
  const clean = String(message).replace(/^\[yandex-proxy\]\s*/, '');
  console.error(`[yandex-proxy] ${localTime()} ${clean}`, ...args);
}

const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_BATCH_STRINGS = 1;
const DEFAULT_DIMENSIONS = 768;
const ALLOWED_DIMENSIONS = [256, 512, 768];
const OPENAI_DEFAULT_DIMENSIONS = 1536;
const OPENAI_ALLOWED_DIMENSIONS = [256, 512, 768, 1024, 1536, 2048, 3072, 4096];

// 429 retry schedule: exponential doubling starting at 1s, capped at 60s.
const RETRY_INITIAL_DELAY_MS = 1_000;
const RETRY_MAX_DELAY_MS = 60_000;

// Success statistics: aggregate upstream 200 responses into a 1s window and
// print the total once the window elapses, instead of logging every call.
const SUCCESS_STATS_WINDOW_MS = 1_000;

// The accumulator lives per request, not per module: createSuccessStats()
// allocates fresh state for each proxyEmbeddings() call, and that state is
// threaded through callUpstream() (per-call recording) down to the
// end-of-request flush. The clock and logger are injectable so the behavior is
// deterministic under test, and concurrent requests never share or corrupt
// each other's counters.

/**
 * Builds the config object consumed by createProxy().
 * @param {object} env process.env
 * @returns {object}
 */
export function buildConfig(env = process.env, fileConfig = { mode: 'yandex' }) {
  const mode = fileConfig?.mode || 'yandex';
  if (mode !== 'yandex' && mode !== 'openai') {
    throw new Error('config mode must be either yandex or openai');
  }

  const apiKey = (mode === 'openai' ? env.OPENAI_API_KEY : env.YANDEX_API_KEY || '').trim();
  const folderId = (env.YANDEX_FOLDER_ID || '').trim();
  const baseUrl = (fileConfig.baseUrl || (env.YANDEX_BASE_URL || 'https://ai.api.cloud.yandex.net/v1')).replace(/\/+$/, '');
  const timeoutMs = Number.parseInt(env.YANDEX_TIMEOUT_MS, 10) || DEFAULT_TIMEOUT_MS;
  const hasMaxBatchStrings = Object.prototype.hasOwnProperty.call(fileConfig, 'maxBatchStrings');
  const maxBatchStrings = hasMaxBatchStrings
    ? fileConfig.maxBatchStrings
    : DEFAULT_MAX_BATCH_STRINGS;

  if (!Number.isInteger(maxBatchStrings) || maxBatchStrings < 1) {
    throw new Error('config maxBatchStrings must be a positive integer');
  }

  if (!apiKey) {
    throw new Error(`Environment variable ${mode === 'openai' ? 'OPENAI_API_KEY' : 'YANDEX_API_KEY'} is required`);
  }
  if (mode === 'yandex' && !folderId) {
    throw new Error('Environment variable YANDEX_FOLDER_ID is required');
  }

  if (mode === 'openai' && !(fileConfig.model || '').trim()) {
    throw new Error('config model is required for openai mode');
  }

  return { mode, apiKey, folderId, baseUrl, model: fileConfig.model, timeoutMs, maxBatchStrings };
}

/**
 * The full Yandex model identifier for document embeddings.
 * @param {string} folderId
 * @returns {string}
 */
export function yandexModel(folderId) {
  return `emb://${folderId}/text-embeddings-v2-doc/latest`;
}

/**
 * Validates and normalizes the `input` field of an embeddings request.
 * Returns the list of strings to embed.
 *
 * @param {unknown} input
 * @returns {string[]}
 * @throws {ValidationError}
 */
export function normalizeInput(input) {
  if (typeof input === 'string') {
    if (!input.trim()) {
      throw new ValidationError('input string must not be empty', 'input');
    }
    return [input];
  }

  if (!Array.isArray(input)) {
    throw new ValidationError('input must be a string or an array of strings', 'input');
  }

  if (input.length === 0) {
    throw new ValidationError('input array must not be empty', 'input');
  }

  if (input.every((item) => Array.isArray(item))) {
    throw new ValidationError(
      'token arrays are not supported: Yandex embeddings accept strings only',
      'input'
    );
  }

  if (!input.every((item) => typeof item === 'string')) {
    throw new ValidationError(
      'input must contain only strings (numbers/token arrays are not supported)',
      'input'
    );
  }

  if (input.some((item) => !item.trim())) {
    throw new ValidationError('input array must not contain empty strings', 'input');
  }

  return [...input];
}

/**
 * Validates and normalizes the `dimensions` field of an embeddings request.
 * Absent (`undefined`) or `null` resolves to the Yandex default (768).
 * Otherwise the value must be an integer in [256, 512, 768].
 *
 * @param {unknown} dimensions client-provided value (may be undefined)
 * @returns {number} normalized dimension size
 * @throws {ValidationError} when `dimensions` is present but invalid
 */
export function normalizeDimensions(dimensions) {
  if (dimensions === undefined || dimensions === null) {
    return DEFAULT_DIMENSIONS;
  }

  if (
    typeof dimensions !== 'number' ||
    !Number.isInteger(dimensions) ||
    !ALLOWED_DIMENSIONS.includes(dimensions)
  ) {
    throw new ValidationError(
      `dimensions must be one of ${ALLOWED_DIMENSIONS.join(', ')}`,
      'dimensions'
    );
  }

  return dimensions;
}

/**
 * Transforms a client embeddings request into the upstream Yandex request body.
 * @param {object} body client request body (already parsed JSON)
 * @param {string} folderId
 * @returns {object}
 * @throws {ValidationError} when `input` is missing/invalid or `dimensions` is invalid
 */
export function buildUpstreamBody(body, configOrFolderId) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new ValidationError('request body must be a JSON object', 'body');
  }

  // normalizeInput validates presence and shape of `input`
  normalizeInput(body.input);
  const config = typeof configOrFolderId === 'string'
    ? { mode: 'yandex', folderId: configOrFolderId }
    : configOrFolderId;
  const isOpenAI = config?.mode === 'openai';
  const allowed = isOpenAI ? OPENAI_ALLOWED_DIMENSIONS : ALLOWED_DIMENSIONS;
  const defaultDimensions = isOpenAI ? OPENAI_DEFAULT_DIMENSIONS : DEFAULT_DIMENSIONS;
  let dimensions = body.dimensions;
  if (dimensions === undefined || dimensions === null) {
    dimensions = defaultDimensions;
  } else if (typeof dimensions !== 'number' || !Number.isInteger(dimensions) || !allowed.includes(dimensions)) {
    throw new ValidationError(`dimensions must be one of ${allowed.join(', ')}`, 'dimensions');
  }

  return {
    ...body,
    model: isOpenAI ? config.model : yandexModel(config.folderId),
    encoding_format: 'float',
    dimensions,
  };
}

/**
 * Creates a fresh, per-request upstream-success accumulator. Each call to
 * proxyEmbeddings() allocates its own accumulator and threads it through the
 * sequential upstream calls, so success statistics never leak across requests.
 * @returns {{ count: number, windowStart: number | null }}
 */
export function createSuccessStats() {
  return { count: 0, windowStart: null };
}

/**
 * Records one successful (HTTP 200) upstream response into the given
 * per-request accumulator. The first success opens a new aggregation window
 * silently; further successes within 1s accumulate. Once a success arrives
 * after the window has been open for more than 1s, the accumulated count is
 * logged and the window restarts including this response. Zero values are
 * never printed.
 *
 * @param {{ count: number, windowStart: number | null }} stats per-request accumulator from createSuccessStats()
 * @param {number} [now=Date.now()] current timestamp in ms, injectable for tests
 * @param {(message: string) => void} [log=logInfo] logger, injectable for tests
 * @returns {void}
 */
export function recordUpstreamSuccess(stats, tokenizedStrings = 1, now = Date.now(), log = logInfo, mode = 'yandex') {
  // Backward-compatible injectable form: (stats, now, log).
  if (typeof tokenizedStrings === 'number' && tokenizedStrings >= 1000) {
    mode = 'yandex';
    log = typeof now === 'function' ? now : logInfo;
    now = tokenizedStrings;
    tokenizedStrings = 1;
  } else if (typeof tokenizedStrings !== 'number' || tokenizedStrings < 1) {
    tokenizedStrings = 1;
  }
  const { count, windowStart } = stats;

  if (windowStart === null) {
    // First success in a new window: start accumulating without any output.
    stats.count = tokenizedStrings;
    stats.windowStart = now;
    return;
  }

  if (now - windowStart > SUCCESS_STATS_WINDOW_MS) {
    // The window elapsed: report what accumulated, then start a fresh window
    // that already contains this response.
    log(`[yandex-proxy] ${mode === 'openai' ? 'open api' : 'yandex'} ok: ${count} successfully tokenized string(s) in ${now - windowStart}ms`);
    stats.count = tokenizedStrings;
    stats.windowStart = now;
    return;
  }

  stats.count = count + tokenizedStrings;
}

/**
 * Flushes the per-request accumulator, reporting any successes that have not
 * yet been printed (e.g. because the 1s window never elapsed). Called at the
 * end of each incoming request (including failures). A zero count prints
 * nothing — the accumulator is simply reset to a fresh state.
 *
 * @param {{ count: number, windowStart: number | null }} stats per-request accumulator from createSuccessStats()
 * @param {number} [now=Date.now()] current timestamp in ms, injectable for tests
 * @param {(message: string) => void} [log=logInfo] logger, injectable for tests
 * @returns {void}
 */
export function flushUpstreamSuccess(stats, now = Date.now(), log = logInfo, mode = 'yandex') {
  if (typeof now === 'function') {
    mode = 'yandex';
    log = now;
    now = Date.now();
  }
  const { count, windowStart } = stats;
  if (count > 0) {
    log(`[yandex-proxy] ${mode === 'openai' ? 'open api' : 'yandex'} ok: ${count} successfully tokenized string(s) in ${now - windowStart}ms`);
  }
  stats.count = 0;
  stats.windowStart = null;
}

/**
 * Runs a single upstream embeddings call and returns the parsed JSON response.
 * HTTP 200 responses are recorded into the per-request success accumulator
 * (`stats`); pass `null` to skip accumulation entirely.
 * @param {object} upstreamBody body to send to Yandex
 * @param {object} config
 * @param {{ count: number, windowStart: number | null } | null} stats per-request accumulator, or null to skip recording
 * @param {typeof fetch} [fetchImpl]
 * @returns {Promise<object>}
 * @throws {UpstreamError|UpstreamNetworkError|UpstreamResponseError}
 */
export async function callUpstream(
  upstreamBody,
  config,
  stats = null,
  fetchImpl = globalThis.fetch
) {
  const url = `${config.baseUrl}/embeddings`;

  let res;
  try {
    res = await fetchImpl(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(config.timeoutMs),
    });
  } catch (err) {
    const cause = err?.cause ?? err;
    const message = `Upstream request failed: ${cause?.message ?? String(err)}`;
    logError(`[yandex-proxy] ${config.mode === 'openai' ? 'open api' : 'yandex'} network error: ${message}`);
    throw new UpstreamNetworkError(message);
  }

  if (res.status === 200) {
    // Success: accumulate into this request's 1s stats window instead of
    // logging the call.
    // Statistics are recorded only after the response cardinality is checked.
  } else {
    // Non-200: keep the per-call status line so errors stay visible immediately.
    logInfo(`[yandex-proxy] ${config.mode === 'openai' ? 'open api' : 'yandex'} status: ${res.status}`);
  }

  const rawText = await res.text();
  let json;
  try {
    json = JSON.parse(rawText);
  } catch {
    json = { raw: rawText };
  }

  if (!res.ok) {
    const inputs = Array.isArray(upstreamBody.input) ? upstreamBody.input : [upstreamBody.input];
    const inputLengths = inputs.map((input) => typeof input === 'string' ? input.length : null);
    // Status alone cannot distinguish an invalid request shape from a source
    // chunk that exceeds the provider's per-input limit. Log only metadata
    // about source text, plus the provider's response, to diagnose failures
    // without writing indexed source content to the terminal.
    logError(
      `[yandex-proxy] ${config.mode === 'openai' ? 'open api' : 'yandex'} rejected upstream batch: status ${res.status}, input string(s): ${inputs.length}, character lengths: ${JSON.stringify(inputLengths)}, response: ${JSON.stringify(json)}`
    );
    throw new UpstreamError(res.status, json);
  }

  const expected = Array.isArray(upstreamBody.input) ? upstreamBody.input.length : 1;
  const actual = Array.isArray(json?.data) ? json.data.length : 0;
  if (actual !== expected) {
    // Do not print json.data: it contains full embedding vectors. The response
    // shape is enough to identify provider-side cardinality failures.
    logError(
      `[yandex-proxy] ${config.mode === 'openai' ? 'open api' : 'yandex'} invalid upstream response: expected ${expected} embedding(s), received ${actual}, response keys: ${JSON.stringify(Object.keys(json ?? {}))}`
    );
    throw new UpstreamResponseError(expected, actual);
  }
  if (stats) {
    recordUpstreamSuccess(stats, expected, Date.now(), logInfo, config.mode);
  }

  return json;
}

function validateUpstreamResponse(response, expected) {
  const actual = Array.isArray(response?.data) ? response.data.length : 0;
  if (actual !== expected) {
    throw new UpstreamResponseError(expected, actual);
  }
}

/**
 * Runs a single upstream call, retrying HTTP 429 responses with exponential
 * backoff. Delays double from 1s up to a 60s cap (1s, 2s, 4s, 8s, 16s, 32s,
 * 60s). If a retry that followed a full 60s wait still returns 429, the last
 * UpstreamError (status 429) is thrown — the caller fails the whole request.
 *
 * Non-429 errors (UpstreamError with any other status) and UpstreamNetworkError
 * propagate immediately without retrying.
 *
 * @param {object} upstreamBody body to send to Yandex
 * @param {object} config
 * @param {{ count: number, windowStart: number | null } | null} [stats] per-request accumulator, or null to skip recording
 * @param {typeof fetch} [fetchImpl]
 * @param {(ms: number) => Promise<void>} [sleepImpl] delay function, injected for tests
 * @returns {Promise<object>} parsed JSON response
 */
export async function callUpstreamWithRetry(
  upstreamBody,
  config,
  stats = null,
  fetchImpl = globalThis.fetch,
  sleepImpl = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
) {
  let delayMs = RETRY_INITIAL_DELAY_MS;
  let lastDelayWasMax = false;

  for (;;) {
    try {
      return await callUpstream(upstreamBody, config, stats, fetchImpl);
    } catch (err) {
      if (!(err instanceof UpstreamError) || err.status !== 429) {
        throw err;
      }
      // A 429 that follows the full 60s wait means we are rate-limited even
      // after backoff: fail the whole request with the 429.
      if (lastDelayWasMax) {
        throw err;
      }
      // 429: wait, then retry (1s, 2s, 4s, ..., capped at 60s).
      await sleepImpl(delayMs);
      lastDelayWasMax = delayMs >= RETRY_MAX_DELAY_MS;
      delayMs = Math.min(delayMs * 2, RETRY_MAX_DELAY_MS);
    }
  }
}

/**
 * Merges N single-item embeddings responses into one OpenAI-shaped response:
 *  - `data` re-indexed 0..N-1 in original order
 *  - `usage` token counters summed
 * @param {object[]} responses
 * @returns {object}
 */
export function mergeResponses(responses) {
  if (!Array.isArray(responses) || responses.length === 0) {
    throw new Error('mergeResponses requires at least one response');
  }

  const first = responses[0];
  let index = 0;
  const data = [];

  for (const response of responses) {
    const items = Array.isArray(response?.data) ? response.data : [];
    for (const item of items) {
      data.push({ ...item, index });
      index += 1;
    }
  }

  const usage = { prompt_tokens: 0, total_tokens: 0 };
  let hasCompletionTokens = false;
  for (const response of responses) {
    const u = response?.usage;
    if (!u) continue;
    usage.prompt_tokens += u.prompt_tokens ?? 0;
    usage.total_tokens += u.total_tokens ?? 0;
    if (u.completion_tokens != null) {
      usage.completion_tokens = (usage.completion_tokens ?? 0) + u.completion_tokens;
      hasCompletionTokens = true;
    }
  }
  if (!hasCompletionTokens) {
    delete usage.completion_tokens;
  }

  return {
    object: 'list',
    data,
    model: first?.model ?? null,
    usage,
  };
}

/**
 * Orchestrates the whole proxy flow for one incoming embeddings request.
 *
 * Processes batched inputs strictly sequentially, splitting them into chunks of
 * at most `config.maxBatchStrings`. A one-string chunk is sent as a string;
 * larger chunks are sent as arrays. Each call retries 429s.
 *
 * Success statistics are accumulated per request: a fresh accumulator is
 * allocated here, threaded through every upstream call, and flushed (logged)
 * once when the request finishes — success counters never leak across
 * concurrent requests.
 *
 * @param {object} body client request body
 * @param {object} config result of buildConfig()
 * @param {typeof fetch} [fetchImpl]
 * @returns {Promise<object>} OpenAI-shaped embeddings response
 */
export async function proxyEmbeddings(body, config, fetchImpl = globalThis.fetch) {
  const upstreamBody = buildUpstreamBody(body, config);
  const inputs = normalizeInput(body.input);

  // Per-request accumulator: successes from this request's upstream calls are
  // counted here and reported exactly once when the request ends.
  const stats = createSuccessStats();

  // Sequential loop: one upstream call per input string, awaiting each one.
  const responses = [];
  const maxBatchStrings = config.maxBatchStrings ?? DEFAULT_MAX_BATCH_STRINGS;

  const sendBatchWithRecovery = async (chunk) => {
    let response;
    try {
      response = await callUpstreamWithRetry(
        { ...upstreamBody, input: chunk.length === 1 ? chunk[0] : chunk },
        config,
        stats,
        fetchImpl
      );
      validateUpstreamResponse(response, chunk.length);
      return [response];
    } catch (err) {
      if (!(err instanceof UpstreamResponseError)) {
        throw err;
      }
    }

    // A cardinality mismatch gets exactly one immediate retry of the same batch.
    try {
      response = await callUpstreamWithRetry(
        { ...upstreamBody, input: chunk.length === 1 ? chunk[0] : chunk },
        config,
        stats,
        fetchImpl
      );
      validateUpstreamResponse(response, chunk.length);
      return [response];
    } catch (err) {
      if (!(err instanceof UpstreamResponseError) || chunk.length === 1) {
        throw err;
      }
    }

    // If the batch still cannot be trusted, isolate each string sequentially.
    const singletonResponses = [];
    for (const input of chunk) {
      singletonResponses.push(...await sendBatchWithRecovery([input]));
    }
    return singletonResponses;
  };

  try {
    for (let offset = 0; offset < inputs.length; offset += maxBatchStrings) {
      const chunk = inputs.slice(offset, offset + maxBatchStrings);
      responses.push(...await sendBatchWithRecovery(chunk));
    }
  } finally {
    // Report any successes accumulated since the last window expiry, even if
    // the request failed part-way through.
    flushUpstreamSuccess(stats, Date.now(), logInfo, config.mode);
  }

  return mergeResponses(responses);
}

