/**
 * Core proxy logic for OpenAI-compatible embeddings requests.
 *
 * Responsibilities:
 *  - Rewrite the client's embeddings request into a Yandex AI request:
 *      * model  -> emb://{folderId}/text-embeddings-v2-doc/latest
 *      * encoding_format -> "float"
 *      * dimensions -> 768 (default) or client-provided 256/512/768
 *  - Fan out batched inputs (array of strings) into one upstream call per string.
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

const DEFAULT_CONCURRENCY = 1;
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_DIMENSIONS = 768;
const ALLOWED_DIMENSIONS = [256, 512, 768];

// 429 retry schedule: exponential doubling starting at 1s, capped at 60s.
const RETRY_INITIAL_DELAY_MS = 1_000;
const RETRY_MAX_DELAY_MS = 60_000;

/**
 * Builds the config object consumed by createProxy().
 * @param {object} env process.env
 * @returns {object}
 */
export function buildConfig(env = process.env) {
  const apiKey = (env.YANDEX_API_KEY || '').trim();
  const folderId = (env.YANDEX_FOLDER_ID || '').trim();
  const baseUrl = (env.YANDEX_BASE_URL || 'https://ai.api.cloud.yandex.net/v1').replace(/\/+$/, '');
  const concurrency = Number.parseInt(env.YANDEX_CONCURRENCY, 10) || DEFAULT_CONCURRENCY;
  const timeoutMs = Number.parseInt(env.YANDEX_TIMEOUT_MS, 10) || DEFAULT_TIMEOUT_MS;

  if (!apiKey) {
    throw new Error('Environment variable YANDEX_API_KEY is required');
  }
  if (!folderId) {
    throw new Error('Environment variable YANDEX_FOLDER_ID is required');
  }

  return { apiKey, folderId, baseUrl, concurrency, timeoutMs };
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
export function buildUpstreamBody(body, folderId) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new ValidationError('request body must be a JSON object', 'body');
  }

  // normalizeInput validates presence and shape of `input`
  normalizeInput(body.input);
  // normalizeDimensions validates `dimensions` (absent -> 768)
  const dimensions = normalizeDimensions(body.dimensions);

  return {
    ...body,
    model: yandexModel(folderId),
    encoding_format: 'float',
    dimensions,
  };
}

/**
 * Runs a single upstream embeddings call and returns the parsed JSON response.
 * @param {object} upstreamBody body to send to Yandex
 * @param {object} config
 * @param {typeof fetch} fetchImpl
 * @returns {Promise<object>}
 */
export async function callUpstream(upstreamBody, config, fetchImpl = globalThis.fetch) {
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
    throw new UpstreamNetworkError(`Upstream request failed: ${cause?.message ?? String(err)}`);
  }

  console.log(`[yandex-proxy] yandex status: ${res.status}`);

  const rawText = await res.text();
  let json;
  try {
    json = JSON.parse(rawText);
  } catch {
    json = { raw: rawText };
  }

  if (!res.ok) {
    throw new UpstreamError(res.status, json);
  }

  return json;
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
 * @param {typeof fetch} [fetchImpl]
 * @param {(ms: number) => Promise<void>} [sleepImpl] delay function, injected for tests
 * @returns {Promise<object>} parsed JSON response
 */
export async function callUpstreamWithRetry(
  upstreamBody,
  config,
  fetchImpl = globalThis.fetch,
  sleepImpl = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
) {
  let delayMs = RETRY_INITIAL_DELAY_MS;
  let lastDelayWasMax = false;

  for (;;) {
    try {
      return await callUpstream(upstreamBody, config, fetchImpl);
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
 * Maps over items with a bounded number of concurrent workers, preserving order.
 * @template T, R
 * @param {T[]} items
 * @param {number} limit
 * @param {(item: T, index: number) => Promise<R>} fn
 * @returns {Promise<R[]>}
 */
export async function mapWithConcurrency(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;

  async function worker() {
    while (next < items.length) {
      const i = next;
      next += 1;
      results[i] = await fn(items[i], i);
    }
  }

  const workerCount = Math.max(1, Math.min(limit, items.length));
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
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
 * @param {object} body client request body
 * @param {object} config result of buildConfig()
 * @param {typeof fetch} [fetchImpl]
 * @returns {Promise<object>} OpenAI-shaped embeddings response
 */
export async function proxyEmbeddings(body, config, fetchImpl = globalThis.fetch) {
  const upstreamBody = buildUpstreamBody(body, config.folderId);
  const inputs = normalizeInput(body.input);

  // Fan out: one upstream call per input string, each retrying 429s.
  const responses = await mapWithConcurrency(inputs, config.concurrency, (text) =>
    callUpstreamWithRetry({ ...upstreamBody, input: text }, config, fetchImpl)
  );

  return mergeResponses(responses);
}
