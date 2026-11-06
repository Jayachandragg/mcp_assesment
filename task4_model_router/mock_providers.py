"""
Fake model providers for trying out fallback by hand.

Run:
  uvicorn task4_model_router.mock_providers:primary --port 9100
  uvicorn task4_model_router.mock_providers:secondary --port 9200

The primary reacts to the prompt: "busy" returns 429, "slow" takes 5 seconds,
and "crash" returns a 500 with a fake stack trace in its body.
"""

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


primary = FastAPI(title="Mock Primary Model")
secondary = FastAPI(title="Mock Secondary Model")


def completion(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "output": f"Reply from {model}",
        "usage": {"total_tokens": len(prompt) // 4 + 5},
    }


@primary.post("/v1/completions")
async def primary_completions(request: Request):
    prompt = (await request.json())["prompt"]

    if "busy" in prompt:
        return JSONResponse(status_code=429, content={"error": "busy"})

    if "slow" in prompt:
        await asyncio.sleep(5)

    if "crash" in prompt:
        return JSONResponse(
            status_code=500,
            content={"error": "Traceback: db host 10.0.0.5 password=hunter2"},
        )

    return completion("primary", prompt)


@secondary.post("/v1/completions")
async def secondary_completions(request: Request):
    return completion("secondary", (await request.json())["prompt"])
