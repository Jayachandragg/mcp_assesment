import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from task4_model_router.models import CompletionRequest, GatewayError
from task4_model_router.rate_limiter import DATABASE, RateLimiter, RateLimitExceeded
from task4_model_router.router import PRIMARY_TIMEOUT_SECONDS, ModelRouter


logger = logging.getLogger("model-router")


def error_response(
    status_code: int,
    error_type: str,
    message: str,
    retry_after: int | None = None,
) -> JSONResponse:
    # Every error from this gateway goes through here, so they all look the
    # same. request_id lets us match a client report to the logs.
    request_id = uuid.uuid4().hex
    logger.info("Request %s failed: %s", request_id, error_type)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
                "request_id": request_id,
            }
        },
        headers={"Retry-After": str(retry_after)} if retry_after else None,
    )


def read_total_tokens(result: dict) -> int | None:
    usage = result.get("usage")
    total = usage.get("total_tokens") if isinstance(usage, dict) else None

    if type(total) is int and total >= 0:
        return total

    return None


def create_app(
    transport: httpx.AsyncBaseTransport | None = None,
    database: str | Path = os.getenv("RATE_LIMIT_DB", DATABASE),
    primary_timeout: float = PRIMARY_TIMEOUT_SECONDS,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # No client timeout, because ModelRouter sets its own deadlines.
        async with httpx.AsyncClient(transport=transport, timeout=None) as client:
            app.state.limiter = RateLimiter(database)
            app.state.router = ModelRouter(client, primary_timeout=primary_timeout)
            yield

    app = FastAPI(title="Model Fallback Router", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, error: GatewayError):
        return error_response(
            error.status_code,
            error.error_type,
            error.message,
            error.retry_after,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return error_response(400, "invalid_request", "Invalid request body")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        # 404s and 405s from FastAPI.
        return error_response(error.status_code, "invalid_route", "Route not available")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        logger.exception("Unexpected gateway error")
        return error_response(500, "internal_error", "Internal gateway error")

    @app.post("/v1/completions")
    async def completions(
        request: CompletionRequest,
        x_tenant_key: str | None = Header(default=None),
    ):
        # We can't rate limit without knowing the tenant.
        if not x_tenant_key:
            raise GatewayError(401, "missing_tenant_key", "Missing X-Tenant-Key header")

        try:
            reservation = await app.state.limiter.reserve(
                x_tenant_key,
                request.estimated_tokens,
            )
        except RateLimitExceeded as error:
            raise GatewayError(
                429,
                "rate_limit_exceeded",
                "Token rate limit exceeded",
                retry_after=error.retry_after,
            )

        try:
            result = await app.state.router.complete(request.model_dump())
        except GatewayError:
            # Nothing was generated, so don't charge the tenant.
            await app.state.limiter.release(reservation)
            raise

        total_tokens = read_total_tokens(result)

        if total_tokens is not None:
            await app.state.limiter.reconcile(reservation, total_tokens)

        return result

    return app


app = create_app()
