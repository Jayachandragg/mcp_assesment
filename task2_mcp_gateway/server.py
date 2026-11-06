import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from task2_mcp_gateway.models import (
    DOWNSTREAM_UNAVAILABLE,
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    UNAUTHENTICATED,
    UNAUTHORIZED_TOOL_CALL,
    JsonRpcRequest,
    ToolCallParams,
    rpc_error,
)


logger = logging.getLogger("mcp-gateway")

DOWNSTREAM_URL = os.getenv("DOWNSTREAM_URL", "http://localhost:8001/mcp")

# Mock tokens for the assessment. A real gateway would verify a JWT here.
TOKEN_ROLES = {
    "admin-token": "admin",
    "viewer-token": "viewer",
}

# These only apply to one connection, so they don't get forwarded.
CONNECTION_HEADERS = frozenset(
    (
        "accept-encoding",
        "connection",
        "content-encoding",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    )
)

# The token is for the gateway. The downstream server shouldn't see it.
BLOCKED_REQUEST_HEADERS = CONNECTION_HEADERS | {"authorization"}


def get_role(authorization: str | None) -> str | None:
    # Authorization must look like: Bearer <token>
    scheme, _, token = (authorization or "").partition(" ")

    if scheme.lower() != "bearer":
        return None

    return TOKEN_ROLES.get(token.strip())


def error_response(
    status_code: int,
    request_id: int | str | None,
    code: int,
    message: str,
    headers: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=rpc_error(request_id, code, message),
        headers=headers,
    )


async def forward_request(
    client: httpx.AsyncClient,
    request: Request,
    payload: dict,
    request_id: int | str | None,
) -> Response:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name not in BLOCKED_REQUEST_HEADERS
    }

    # Forward the parsed payload, not the raw bytes. If the JSON has duplicate
    # keys, downstream then sees the same tool name we checked.
    downstream_request = client.build_request(
        "POST",
        DOWNSTREAM_URL,
        json=payload,
        headers=headers,
    )

    try:
        downstream = await client.send(downstream_request, stream=True)
    except httpx.RequestError as error:
        # The client doesn't need the internal error.
        logger.warning("Downstream request failed: %r", error)
        return error_response(
            502,
            request_id,
            DOWNSTREAM_UNAVAILABLE,
            "Downstream MCP server unavailable",
        )

    async def body():
        try:
            async for chunk in downstream.aiter_bytes():
                yield chunk
        finally:
            await downstream.aclose()

    # MCP servers can reply with JSON, SSE or an empty 202, so just stream
    # whatever comes back.
    return StreamingResponse(
        body(),
        status_code=downstream.status_code,
        headers={
            name: value
            for name, value in downstream.headers.items()
            if name not in CONNECTION_HEADERS
        },
    )


def create_app(transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Shared client, so connections get reused.
        timeout = httpx.Timeout(30.0, connect=5.0)

        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            app.state.client = client
            yield

    app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)

    @app.post("/mcp")
    async def mcp_gateway(request: Request):
        # Check the token before doing anything else.
        role = get_role(request.headers.get("authorization"))

        if role is None:
            return error_response(
                401,
                None,
                UNAUTHENTICATED,
                "Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            payload = json.loads(await request.body())
        except (ValueError, RecursionError):
            return error_response(400, None, PARSE_ERROR, "Parse error")

        # No batches: an admin_ call inside one would skip the check below.
        if not isinstance(payload, dict):
            return error_response(400, None, INVALID_REQUEST, "Invalid Request")

        try:
            rpc_request = JsonRpcRequest.model_validate(payload)
        except ValidationError:
            return error_response(400, None, INVALID_REQUEST, "Invalid Request")

        # Only tools/call needs checking. Everything else goes straight through.
        if rpc_request.method == "tools/call":
            try:
                params = ToolCallParams.model_validate(rpc_request.params)
            except ValidationError:
                # If we can't read the tool name, we can't authorize it.
                return error_response(
                    200,
                    rpc_request.id,
                    INVALID_PARAMS,
                    "Invalid params",
                )

            # admin_* tools require the admin role.
            if params.name.startswith("admin_") and role != "admin":
                logger.warning("Blocked %s for role %s", params.name, role)
                return error_response(
                    200,
                    rpc_request.id,
                    UNAUTHORIZED_TOOL_CALL,
                    "Unauthorized Tool Call",
                )

        return await forward_request(
            app.state.client,
            request,
            payload,
            rpc_request.id,
        )

    return app


app = create_app()
