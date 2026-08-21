/**
 * OpenAI-compatible embeddings proxy for the Yandex AI endpoint.
 *
 * Accepts OpenAI-shaped POST requests on:
 *   POST /v1/embeddings
 *   POST /embeddings
 *
 * and forwards them to https://ai.api.cloud.yandex.net/v1/embeddings with:
 *   - Authorization: Bearer $YANDEX_API_KEY
 *   - model rewritten to emb://$YANDEX_FOLDER_ID/text-embeddings-v2-doc/latest
 *   - encoding_format forced to "float"
 *   - dimensions forced to the client value (256/512/768) or 768 if absent
 *
 * Batched inputs (arrays of strings) are processed sequentially — one upstream
 * call per string, awaiting each one — and merged back into a single
 * OpenAI-shaped response. Upstream HTTP 429 responses are retried with
 * exponential backoff (1s doubling, capped at 60s); if still rate-limited after
 * the full 60s wait, the request fails 429.
 *
 * Env vars: YANDEX_API_KEY, YANDEX_FOLDER_ID, PORT (default 9988),
 * YANDEX_BASE_URL (optional), YANDEX_TIMEOUT_MS (default 60000).
 */

import express from 'express';
import { pathToFileURL } from 'node:url';
import {
  buildConfig,
  logError,
  logInfo,
  proxyEmbeddings,
  UpstreamError,
  UpstreamNetworkError,
  ValidationError,
} from './embeddings.js';

const PORT = Number.parseInt(process.env.PORT, 10) || 9988;

function createApp(config) {
  const app = express();
  app.disable('x-powered-by');
  app.use(express.json({ limit: '2mb' }));

  app.get('/health', (_req, res) => {
    res.json({ status: 'ok' });
  });

  app.post(['/v1/embeddings', '/embeddings'], async (req, res) => {
    try {
      const input = req.body?.input;
      const inputCount =
        Array.isArray(input) ? input.length : typeof input === 'string' ? 1 : null;
      if (inputCount !== null) {
        logInfo(`[yandex-proxy] request: ${inputCount} input string(s)`);
      }
      const result = await proxyEmbeddings(req.body, config);
      res.json(result);
    } catch (err) {
      if (err instanceof ValidationError) {
        return res.status(400).json({
          error: {
            message: err.message,
            type: 'invalid_request_error',
            param: err.param,
            code: 'invalid_request_error',
          },
        });
      }

      if (err instanceof UpstreamNetworkError) {
        return res.status(502).json({
          error: {
            message: err.message,
            type: 'upstream_network_error',
            param: null,
            code: 'upstream_error',
          },
        });
      }

      if (err instanceof UpstreamError) {
        const body = err.body;
        // Forward the upstream error body verbatim if it already looks like an
        // OpenAI error envelope, otherwise wrap it.
        const payload =
          body && typeof body === 'object' && body.error
            ? body
            : { error: { message: `Upstream request failed with status ${err.status}`, type: 'upstream_error', param: null, code: 'upstream_error', details: body } };
        return res.status(err.status).json(payload);
      }

      logError('[yandex-proxy] unexpected error:', err);
      return res.status(500).json({
        error: {
          message: 'Internal server error',
          type: 'internal_error',
          param: null,
          code: 'internal_error',
        },
      });
    }
  });

  // Single-purpose proxy: anything else is 404.
  app.use((_req, res) => {
    res.status(404).json({
      error: {
        message: 'Not found. This proxy only handles POST /v1/embeddings',
        type: 'not_found',
        param: null,
        code: 'not_found',
      },
    });
  });

  // Convert body-parser / JSON syntax errors into OpenAI-style JSON errors
  // instead of Express's default HTML error page.
  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    if (err && err.type === 'entity.parse.failed') {
      return res.status(400).json({
        error: {
          message: 'Invalid JSON in request body',
          type: 'invalid_request_error',
          param: null,
          code: 'invalid_json',
        },
      });
    }
    if (err && err.type === 'entity.too.large') {
      return res.status(413).json({
        error: {
          message: 'Request body too large',
          type: 'invalid_request_error',
          param: null,
          code: 'body_too_large',
        },
      });
    }
    logError('[yandex-proxy] error:', err);
    return res.status(err?.status || 500).json({
      error: {
        message: err?.message || 'Internal server error',
        type: 'internal_error',
        param: null,
        code: 'internal_error',
      },
    });
  });

  return app;
}

export function start(config = buildConfig()) {
  const app = createApp(config);
  return app.listen(PORT, () => {
    logInfo(`[yandex-proxy] listening on http://localhost:${PORT}`);
    logInfo(`[yandex-proxy] forwarding to ${config.baseUrl}/embeddings`);
    logInfo(
      `[yandex-proxy] model: emb://${config.folderId}/text-embeddings-v2-doc/latest`
    );
  });
}

// Entry point: only start the server when run directly (not when imported).
const isMain = Boolean(process.argv[1]) && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  try {
    start();
  } catch (err) {
    logError(`[yandex-proxy] failed to start: ${err.message}`);
    process.exit(1);
  }
}
