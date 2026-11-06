import asyncio
import json
import sqlite3
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from task4_model_router.models import CompletionRequest
from task4_model_router.rate_limiter import (
    TOKEN_LIMIT,
    WINDOW_SECONDS,
    RateLimiter,
    RateLimitExceeded,
)
from task4_model_router.router import (
    PRIMARY_TIMEOUT_SECONDS,
    PRIMARY_URL,
    SECONDARY_URL,
)
from task4_model_router.server import create_app


TENANT = {"X-Tenant-Key": "tenant-a"}
BODY = {"prompt": "hello", "max_tokens": 10}
SUCCESS = {"model": "m", "output": "ok", "usage": {"total_tokens": 20}}


class FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def database(tmp_path):
    return tmp_path / "rate_limits.db"


def run_app(handler, database, **options):
    return TestClient(
        create_app(httpx.MockTransport(handler), database, **options),
        raise_server_exceptions=False,
    )


def is_primary(request):
    return str(request.url) == PRIMARY_URL


def test_required_limits_match_the_assessment():
    assert TOKEN_LIMIT == 50_000
    assert WINDOW_SECONDS == 60
    assert PRIMARY_TIMEOUT_SECONDS == 3.0


# Rate limiter


def test_limiter_allows_up_to_the_limit_and_no_further(database):
    limiter = RateLimiter(database, clock=FakeClock())

    async def scenario():
        await limiter.reserve("tenant", TOKEN_LIMIT)

        with pytest.raises(RateLimitExceeded):
            await limiter.reserve("tenant", 1)

    asyncio.run(scenario())


def test_window_slides_and_does_not_reset_all_at_once(database):
    clock = FakeClock()
    limiter = RateLimiter(database, clock=clock)

    async def scenario():
        await limiter.reserve("tenant", 30_000)

        clock.now += 30
        await limiter.reserve("tenant", 20_000)

        # The first entry is still inside the window.
        clock.now += 29.9

        with pytest.raises(RateLimitExceeded):
            await limiter.reserve("tenant", 1)

        # Exactly 60 seconds later only the first entry has left.
        clock.now += 0.1
        await limiter.reserve("tenant", 30_000)

        with pytest.raises(RateLimitExceeded):
            await limiter.reserve("tenant", 1)

    asyncio.run(scenario())


def test_expired_rows_are_evicted_for_all_tenants(database):
    clock = FakeClock()
    limiter = RateLimiter(database, clock=clock)

    async def scenario():
        await limiter.reserve("idle-tenant", 100)
        clock.now += WINDOW_SECONDS
        await limiter.reserve("active-tenant", 100)

    asyncio.run(scenario())

    connection = sqlite3.connect(database)
    count = connection.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
    connection.close()

    assert count == 1


def test_retry_after_reports_when_the_oldest_entry_expires(database):
    clock = FakeClock()
    limiter = RateLimiter(database, clock=clock)

    async def scenario():
        await limiter.reserve("tenant", TOKEN_LIMIT)
        clock.now += 45

        with pytest.raises(RateLimitExceeded) as caught:
            await limiter.reserve("tenant", 1)

        assert caught.value.retry_after == 15

    asyncio.run(scenario())


def test_tenants_have_independent_limits(database):
    limiter = RateLimiter(database, clock=FakeClock())

    async def scenario():
        await limiter.reserve("tenant-a", TOKEN_LIMIT)
        await limiter.reserve("tenant-b", TOKEN_LIMIT)

    asyncio.run(scenario())


def test_concurrent_requests_cannot_overrun_the_limit(database):
    limiter = RateLimiter(database)

    async def attempt():
        try:
            await limiter.reserve("tenant", 5_000)
            return True
        except RateLimitExceeded:
            return False

    async def scenario():
        return await asyncio.gather(*(attempt() for _ in range(25)))

    assert sum(asyncio.run(scenario())) == 10


def test_limit_is_shared_between_gateway_processes(database):
    # Two instances on one file, like two uvicorn workers.
    first = RateLimiter(database)
    second = RateLimiter(database)

    async def attempt(limiter):
        try:
            await limiter.reserve("tenant", 5_000)
            return True
        except RateLimitExceeded:
            return False

    async def scenario():
        return await asyncio.gather(
            *(attempt(first if index % 2 else second) for index in range(25))
        )

    assert sum(asyncio.run(scenario())) == 10


def test_usage_survives_a_restart(database):
    async def scenario():
        await RateLimiter(database).reserve("tenant", TOKEN_LIMIT)

        with pytest.raises(RateLimitExceeded):
            await RateLimiter(database).reserve("tenant", 1)

    asyncio.run(scenario())


def test_api_keys_are_not_stored_in_plain_text(database):
    asyncio.run(RateLimiter(database).reserve("secret-api-key", 1))

    assert b"secret-api-key" not in database.read_bytes()

    wal = database.with_name(database.name + "-wal")

    if wal.exists():
        assert b"secret-api-key" not in wal.read_bytes()


# Token accounting


def test_token_estimate_includes_prompt_and_max_tokens():
    request = CompletionRequest(prompt="x" * 400, max_tokens=50)

    assert request.estimated_tokens == 150


def test_client_cannot_lower_its_token_count(database):
    # 400,000 characters is about 100,000 tokens.
    payload = {"prompt": "x" * 400_000, "tokens": 0, "max_tokens": 1}

    with run_app(lambda request: httpx.Response(200, json=SUCCESS), database) as client:
        response = client.post("/v1/completions", headers=TENANT, json=payload)

    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_exceeded"
    assert int(response.headers["retry-after"]) >= 1


def test_reserved_tokens_are_corrected_to_actual_usage(database):
    with run_app(lambda request: httpx.Response(200, json=SUCCESS), database) as client:
        # Reserves 45,000 but the provider reports 20 used.
        for _ in range(3):
            response = client.post(
                "/v1/completions",
                headers=TENANT,
                json={"prompt": "hello", "max_tokens": 45_000},
            )

            assert response.status_code == 200


def test_tokens_above_the_estimate_are_still_counted(database):
    huge = {"model": "m", "output": "ok", "usage": {"total_tokens": TOKEN_LIMIT}}

    with run_app(lambda request: httpx.Response(200, json=huge), database) as client:
        first = client.post("/v1/completions", headers=TENANT, json=BODY)
        second = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert first.status_code == 200
    assert second.status_code == 429


def test_tokens_are_released_when_no_model_answers(database):
    def handler(request):
        return httpx.Response(500, json={"error": "down"})

    payload = {"prompt": "hello", "max_tokens": 45_000}

    with run_app(handler, database) as client:
        for _ in range(3):
            response = client.post("/v1/completions", headers=TENANT, json=payload)

            assert response.status_code == 502


# Fallback


def test_primary_success_does_not_use_secondary(database):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json=SUCCESS)

    with run_app(handler, database) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 200
    assert response.json() == SUCCESS
    assert calls == [PRIMARY_URL]


def test_primary_429_uses_secondary(database):
    calls = []

    def handler(request):
        calls.append(str(request.url))

        if is_primary(request):
            return httpx.Response(429)

        return httpx.Response(200, json={**SUCCESS, "model": "secondary"})

    with run_app(handler, database) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 200
    assert response.json()["model"] == "secondary"
    assert calls == [PRIMARY_URL, SECONDARY_URL]


def test_primary_timeout_uses_secondary(database):
    async def handler(request):
        if is_primary(request):
            await asyncio.sleep(5)

        return httpx.Response(200, json={**SUCCESS, "model": "secondary"})

    started = time.monotonic()

    with run_app(handler, database, primary_timeout=0.1) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 200
    assert response.json()["model"] == "secondary"
    assert time.monotonic() - started < 2


def test_timeout_covers_the_whole_response_not_each_read(database):
    # Every chunk arrives quickly, but the response as a whole is too slow.
    async def slow_body():
        for _ in range(20):
            await asyncio.sleep(0.05)
            yield b" "

    async def handler(request):
        if is_primary(request):
            return httpx.Response(200, content=slow_body())

        return httpx.Response(200, json={**SUCCESS, "model": "secondary"})

    started = time.monotonic()

    with run_app(handler, database, primary_timeout=0.2) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.json()["model"] == "secondary"
    assert time.monotonic() - started < 0.9


def test_timed_out_primary_call_is_cancelled(database):
    outcome = {}

    async def handler(request):
        if is_primary(request):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                outcome["cancelled"] = True
                raise

        return httpx.Response(200, json=SUCCESS)

    with run_app(handler, database, primary_timeout=0.1) as client:
        client.post("/v1/completions", headers=TENANT, json=BODY)

    assert outcome["cancelled"] is True


def test_only_429_and_timeout_trigger_fallback(database):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(500)

    with run_app(handler, database) as client:
        client.post("/v1/completions", headers=TENANT, json=BODY)

    assert calls == [PRIMARY_URL]


# Error sanitization


LEAK = "Traceback: db host 10.0.0.5 password=hunter2"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"error": LEAK}),
        httpx.Response(400, json={"error": LEAK}),
        httpx.Response(502, text=f"<html>{LEAK}</html>"),
        httpx.Response(200, text=LEAK),
        httpx.Response(200, json=[LEAK]),
    ],
)
def test_upstream_failures_never_reach_the_client(database, response):
    with run_app(lambda request: response, database) as client:
        result = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert result.status_code == 502
    assert "hunter2" not in result.text
    assert result.json()["error"]["type"] == "upstream_error"


def test_transport_failure_is_sanitized(database):
    def handler(request):
        raise httpx.ConnectError(LEAK)

    with run_app(handler, database) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 502
    assert "hunter2" not in response.text


def test_secondary_failure_is_sanitized(database):
    def handler(request):
        if is_primary(request):
            return httpx.Response(429)

        return httpx.Response(500, text=LEAK)

    with run_app(handler, database) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 502
    assert "hunter2" not in response.text


def test_both_providers_rate_limited(database):
    with run_app(lambda request: httpx.Response(429), database) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "upstream_overloaded"


def test_both_providers_timing_out(database):
    async def handler(request):
        await asyncio.sleep(30)

    with run_app(handler, database, primary_timeout=0.05) as client:
        client.app.state.router.secondary_timeout = 0.05
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "upstream_timeout"


def test_unexpected_bug_returns_standard_error(database):
    def handler(request):
        raise RuntimeError(LEAK)

    with run_app(handler, database) as client:
        response = client.post("/v1/completions", headers=TENANT, json=BODY)

    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert response.json()["error"]["type"] == "internal_error"


def test_all_errors_share_one_shape(database):
    with run_app(lambda request: httpx.Response(500), database) as client:
        responses = [
            client.post("/v1/completions", json=BODY),
            client.post("/v1/completions", headers=TENANT, json={"prompt": ""}),
            client.post("/v1/completions", headers=TENANT, json={"prompt": 5}),
            client.post("/v1/completions", headers=TENANT, json=BODY),
            client.post(
                "/v1/completions",
                headers={**TENANT, "Content-Type": "application/json"},
                content=b"{bad",
            ),
        ]

    for response in responses:
        error = response.json()["error"]

        assert set(error) == {"type", "message", "request_id"}


def test_extra_fields_are_passed_to_the_provider(database):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=SUCCESS)

    with run_app(handler, database) as client:
        client.post(
            "/v1/completions",
            headers=TENANT,
            json={**BODY, "temperature": 0.5},
        )

    assert seen == {**BODY, "temperature": 0.5}


def test_wrong_route_and_method_use_the_standard_shape(database):
    with run_app(lambda request: httpx.Response(200, json=SUCCESS), database) as client:
        responses = [
            client.post("/nope", json=BODY),
            client.get("/v1/completions"),
        ]

    assert [response.status_code for response in responses] == [404, 405]

    for response in responses:
        assert set(response.json()["error"]) == {"type", "message", "request_id"}


def test_each_provider_gets_its_own_api_key(database):
    seen = {}

    def handler(request):
        seen[str(request.url)] = request.headers["authorization"]

        if is_primary(request):
            return httpx.Response(429)

        return httpx.Response(200, json=SUCCESS)

    with run_app(handler, database) as client:
        client.app.state.router.primary_api_key = "primary-key"
        client.app.state.router.secondary_api_key = "secondary-key"
        client.post("/v1/completions", headers=TENANT, json=BODY)

    assert seen == {
        PRIMARY_URL: "Bearer primary-key",
        SECONDARY_URL: "Bearer secondary-key",
    }
