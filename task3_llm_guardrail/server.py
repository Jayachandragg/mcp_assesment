import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from task3_llm_guardrail.redactor import StreamRedactor


logger = logging.getLogger("llm-guardrail")

# Anything that speaks the OpenAI streaming format.
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:9000/v1/chat/completions")
PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY")

ENVELOPE_FIELDS = ("id", "object", "created", "model")

UPSTREAM_ERROR = {
    "error": {
        "type": "upstream_error",
        "message": "Upstream provider unavailable",
    }
}


def sse(event: dict | str) -> str:
    data = event if isinstance(event, str) else json.dumps(event)

    return f"data: {data}\n\n"


class DeltaGuard:
    """
    Rebuilds each stream event from the fields we actually check. Anything else,
    like tool call arguments or logprobs, gets dropped so PII can't slip through.
    """

    def __init__(self):
        self.redactors: dict[int, StreamRedactor] = {}
        self.envelope: dict = {}

    def process(self, event: dict) -> dict | None:
        self.envelope = {
            field: event[field] for field in ENVELOPE_FIELDS if field in event
        }

        choices = [
            clean
            for choice in event.get("choices") or []
            if isinstance(choice, dict)
            and (clean := self.process_choice(choice)) is not None
        ]

        return {**self.envelope, "choices": choices} if choices else None

    def process_choice(self, choice: dict) -> dict | None:
        index = choice.get("index", 0)
        redactor = self.redactors.setdefault(index, StreamRedactor())

        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        content = delta.get("content")
        text = redactor.feed(content) if isinstance(content, str) else ""

        # Send any held text before the finish event.
        if finish_reason:
            text += redactor.flush()

        clean_delta = {"role": delta["role"]} if "role" in delta else {}

        if text:
            clean_delta["content"] = text

        if not clean_delta and not finish_reason:
            return None

        return {
            "index": index,
            "delta": clean_delta,
            "finish_reason": finish_reason,
        }

    def finish(self) -> dict | None:
        # In case the stream ends without a finish_reason.
        choices = [
            {"index": index, "delta": {"content": text}, "finish_reason": None}
            for index, redactor in self.redactors.items()
            if (text := redactor.flush())
        ]

        return {**self.envelope, "choices": choices} if choices else None


async def guarded_stream(upstream: httpx.Response):
    guard = DeltaGuard()

    try:
        async for line in upstream.aiter_lines():
            if not line.startswith("data:"):
                continue

            data = line[len("data:"):].strip()

            if data == "[DONE]":
                break

            # If we can't parse it, we can't check it, so it doesn't go out.
            try:
                event = json.loads(data)
            except ValueError:
                logger.warning("Dropped an unparseable stream event")
                continue

            if not isinstance(event, dict):
                continue

            clean = guard.process(event)

            if clean is not None:
                yield sse(clean)

        last = guard.finish()

        if last is not None:
            yield sse(last)

    except httpx.HTTPError as error:
        # Keep the upstream error out of the response.
        logger.warning("Upstream stream failed: %r", error)
        yield sse(UPSTREAM_ERROR)

    finally:
        # This also runs if the client disconnects, which stops the upstream call.
        await upstream.aclose()

    yield sse("[DONE]")


def create_app(transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # This is the longest gap between chunks, not a limit on the whole stream.
        timeout = httpx.Timeout(30.0, connect=5.0)

        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            app.state.client = client
            yield

    app = FastAPI(title="LLM Streaming Guardrail", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request",
                    "message": "Request body must be a JSON object",
                }
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict):
        client: httpx.AsyncClient = app.state.client

        upstream_request = client.build_request(
            "POST",
            UPSTREAM_URL,
            json={**payload, "stream": True},
            headers=(
                {"Authorization": f"Bearer {PROVIDER_API_KEY}"}
                if PROVIDER_API_KEY
                else None
            ),
        )

        # Open the stream here, so we can still return a proper error status
        # if upstream fails.
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.RequestError as error:
            logger.warning("Upstream request failed: %r", error)
            return JSONResponse(status_code=502, content=UPSTREAM_ERROR)

        if upstream.status_code != 200:
            logger.warning("Upstream returned status %s", upstream.status_code)
            await upstream.aclose()
            return JSONResponse(status_code=502, content=UPSTREAM_ERROR)

        return StreamingResponse(
            guarded_stream(upstream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
