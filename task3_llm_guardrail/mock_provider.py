"""
Fake LLM provider that streams PII in small pieces, so values get split across
deltas.

Run: uvicorn task3_llm_guardrail.mock_provider:app --port 9000
"""

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse


app = FastAPI(title="Mock LLM Provider")

TEXT = (
    "Hello! Please contact john@example.com about the account. "
    "The SSN on file is 123-45-6789 and the card is 4111 1111 1111 1111. "
    "Order 12345 has shipped."
)

CHUNK_SIZE = 7


def event(delta: dict, finish_reason: str | None = None) -> str:
    data = {
        "id": "mock-1",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }

    return f"data: {json.dumps(data)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions():
    async def stream():
        yield event({"role": "assistant"})

        for start in range(0, len(TEXT), CHUNK_SIZE):
            await asyncio.sleep(0.05)
            yield event({"content": TEXT[start:start + CHUNK_SIZE]})

        yield event({}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
