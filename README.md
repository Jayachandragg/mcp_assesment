# AI Field Developer Assessment

Python solutions for all four tasks. Each task lives in its own folder and can be run on its own.

| Task | Folder | What it does |
| ---- | ------ | ------------ |
| 1 | `task1_mcp_server/` | MCP server with two tools over stdio |
| 2 | `task2_mcp_gateway/` | Proxy that checks tokens and blocks `admin_*` tools |
| 3 | `task3_llm_guardrail/` | LLM gateway that redacts PII from a live stream |
| 4 | `task4_model_router/` | Token rate limiter with failover to a backup model |

Each folder has `server.py` as the entry point and `tests.py` for the tests. Schemas go in `models.py`, and anything task-specific sits next to them. Tasks 2 to 4 also include small mock services, so you can run them without a real backend.

## Setup

You need Python 3.11 or newer. I tested on 3.13.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate           # macOS/Linux
python -m pip install -r requirements.txt
```

Run everything below from the project root.

To run all the tests:

```bash
python -m pytest
```

To run one task's tests: `python -m pytest task3_llm_guardrail -v`

---

## Task 1: MCP server

```bash
python -m task1_mcp_server.server
```

It will sit there waiting for JSON-RPC on stdin. To try it by hand, use the MCP Inspector (needs Node.js):

```bash
npx @modelcontextprotocol/inspector python -m task1_mcp_server.server
```

| What happens | What you get back |
| ------------ | ------------------ |
| Unknown tool or bad arguments | JSON-RPC error `-32602`, with the failing fields in `error.data` |
| Good input, but the customer doesn't exist | Normal result with `isError: true` |
| A bug in the server | JSON-RPC error `-32603`, with the details only in stderr |

I used the low-level `Server` rather than FastMCP. FastMCP reports validation failures as `isError` results, but the task asks for JSON-RPC error codes.

The models are strict, so nothing gets coerced: `"50"`, `true`, NaN and Infinity are all rejected as amounts, while `50` is fine. Extra fields are rejected too.

All logs go to stderr. The tests read every line on stdout to confirm it's valid JSON-RPC.

Two things the task left open:
- I took `XXXXX` to mean five digits.
- `reason` is measured as sent, but a reason that's only spaces is rejected as malformed.

## Task 2: MCP gateway

```bash
uvicorn task2_mcp_gateway.mock_downstream:app --port 8001
uvicorn task2_mcp_gateway.server:app --port 8000
```

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer viewer-token" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}'
```

The tokens are mocks: `admin-token` and `viewer-token`. A real gateway would verify a JWT instead.

| What happens | HTTP | JSON-RPC error |
| ------------ | ---- | -------------- |
| Missing or bad token | 401 | `-32000` |
| Non-admin calls an `admin_*` tool | 200 | `-32001 Unauthorized Tool Call` |
| Invalid JSON | 400 | `-32700` |
| Invalid JSON-RPC, or a batch | 400 | `-32600` |
| `tools/call` with no readable tool name | 200 | `-32602` |
| Downstream unreachable | 502 | `-32002` |

A few decisions worth explaining:
- Every request needs a valid token, including `tools/list`.
- If the tool name can't be read, the call is rejected rather than forwarded.
- Batches are rejected, because an `admin_*` call inside one would skip the check.
- The gateway forwards the parsed JSON, not the raw bytes. That way duplicate keys can't show downstream a different tool name from the one that was checked.
- The bearer token isn't passed downstream, since it's meant for the gateway. Other headers are forwarded, apart from connection-level ones.

Replies are streamed straight through, so JSON, SSE and empty `202` responses all work.

## Task 3: Streaming guardrail

```bash
uvicorn task3_llm_guardrail.mock_provider:app --port 9000
uvicorn task3_llm_guardrail.server:app --port 8002
```

```bash
curl -N -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" -d '{"messages":[]}'
```

The gateway uses the OpenAI streaming format. The mock provider sends PII in small pieces so values get split across deltas.

The hard part is PII that lands across two chunks. A fixed overlap buffer doesn't solve it, because a value can still straddle the cut point and neither half will match. `StreamRedactor` holds back only the last unfinished token instead, since that's the only text the next chunk could turn into PII. Everything before it is redacted and sent straight away.

So the first word goes out as soon as it's complete, and held text never goes above 320 characters, however long the response is. The tests split each PII value at every possible position, one character at a time, and at random chunk sizes.

Other choices:
- Each event is rebuilt from the fields that get checked. Tool call arguments, logprobs and anything unparseable are dropped.
- Matching errs on the side of redacting. Any run of 13 or more digits counts as a card and any 9-digit number counts as an SSN, so some order IDs and ZIP+4 codes get redacted too.
- Upstream failures become a plain `502`, or an error event if streaming has already started.

What it won't catch: this is regex matching, so PII that's spelled out or disguised gets through. Tool calls are dropped, not redacted.

## Task 4: Model router

```bash
uvicorn task4_model_router.mock_providers:primary --port 9100
uvicorn task4_model_router.mock_providers:secondary --port 9200
uvicorn task4_model_router.server:app --port 8003
```

```bash
curl -X POST http://localhost:8003/v1/completions \
  -H "X-Tenant-Key: tenant-a" -H "Content-Type: application/json" \
  -d '{"prompt":"hello","max_tokens":50}'
```

Put `busy`, `slow` or `crash` in the prompt to make the mock primary return a 429, take 5 seconds, or return a 500 with a fake stack trace.

**Rate limiting.** Each tenant key gets 50,000 tokens per minute, tracked as a sliding window in on-disk SQLite. The client's own token count is never trusted. The gateway estimates from the prompt plus `max_tokens` and reserves that amount before calling the model. It then corrects the number from `usage.total_tokens`, or releases it if no model answered.

Reserving first matters because otherwise a burst of parallel requests could all pass the check together. Each check runs in a `BEGIN IMMEDIATE` transaction, which also holds up with several gateway workers on one database. Expired rows are cleared on every check, and API keys are stored as hashes.

**Failover.** The primary gets a 3000 ms deadline for the whole call, set with `asyncio.timeout`. httpx's own timeout is per read, so a slow provider could run well past it. A 429 or a timeout moves the request to the secondary, which gets 10 seconds. Other primary failures don't fail over, because the task only names those two.

**Errors.** Every error from the gateway has the same shape, and upstream error text never reaches the client:

```json
{"error": {"type": "rate_limit_exceeded", "message": "Token rate limit exceeded", "request_id": "..."}}
```

| What happens | HTTP | `type` |
| ------------ | ---- | ------ |
| Bad request body | 400 | `invalid_request` |
| Missing `X-Tenant-Key` | 401 | `missing_tenant_key` |
| Wrong URL or method | 404, 405 | `invalid_route` |
| Over the limit (includes `Retry-After`) | 429 | `rate_limit_exceeded` |
| Unexpected gateway failure | 500 | `internal_error` |
| Upstream failure | 502 | `upstream_error` |
| Both providers rate limited | 503 | `upstream_overloaded` |
| Both providers timed out | 504 | `upstream_timeout` |

---

## Configuration

All optional. The defaults point at the local mock services.

| Variable | Default |
| -------- | ------- |
| `DOWNSTREAM_URL` | `http://localhost:8001/mcp` |
| `UPSTREAM_URL` | `http://localhost:9000/v1/chat/completions` |
| `PRIMARY_URL` | `http://localhost:9100/v1/completions` |
| `SECONDARY_URL` | `http://localhost:9200/v1/completions` |
| `RATE_LIMIT_DB` | `task4_model_router/rate_limits.db` |
| `PROVIDER_API_KEY` | Not set. Task 3 sends it as a Bearer token |
| `PRIMARY_API_KEY`, `SECONDARY_API_KEY` | Not set. Task 4 sends them as Bearer tokens |

Tasks 2 to 4 have only been run against the mock services, not real providers. All tokens, keys and customer records here are mock data.
