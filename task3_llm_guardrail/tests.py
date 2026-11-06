import asyncio
import json
import random

import httpx
import pytest
from fastapi.testclient import TestClient

from task3_llm_guardrail import server
from task3_llm_guardrail.redactor import (
    MAX_HOLD,
    REDACTED,
    StreamRedactor,
    redact_text,
)
from task3_llm_guardrail.server import create_app, guarded_stream


PII_VALUES = [
    "bob@example.com",
    "first.last+tag@mail.example.co.uk",
    "BOB@EXAMPLE.COM",
    "bob@localhost",
    "123-45-6789",
    "123 45 6789",
    "123.45.6789",
    "123456789",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "4111111111111111",
    "4111  1111  1111  1111",
    "1234 5678 9012 3456",
    "1234567890123456",
    "1234-5678-9012-345",
    "3782 822463 10005",
    "2024 4111 1111 1111 1111",
]

TEMPLATE = "Hello there, the value is {} and that was the value. Thanks!"


def stream_through(chunks):
    redactor = StreamRedactor()
    output = "".join(redactor.feed(chunk) for chunk in chunks)

    return output + redactor.flush()


@pytest.mark.parametrize("value", PII_VALUES)
def test_redact_text(value):
    result = redact_text(TEMPLATE.format(value))

    assert value not in result
    assert result == TEMPLATE.format(REDACTED)


@pytest.mark.parametrize(
    "text",
    [
        "Order 12345 has shipped.",
        "Call +1 415 555 0100 today.",
        "Version 1.2.3 of a-b_c is out.",
    ],
)
def test_normal_text_is_not_redacted(text):
    assert redact_text(text) == text
    assert stream_through(text) == text


@pytest.mark.parametrize("value", PII_VALUES)
def test_pii_is_redacted_at_every_two_chunk_split(value):
    text = TEMPLATE.format(value)

    for split in range(len(text) + 1):
        output = stream_through([text[:split], text[split:]])

        assert output == TEMPLATE.format(REDACTED), f"split at {split}"


@pytest.mark.parametrize("value", PII_VALUES)
def test_pii_is_redacted_one_character_at_a_time(value):
    text = TEMPLATE.format(value)

    assert stream_through(text) == TEMPLATE.format(REDACTED)


def test_pii_is_redacted_at_random_chunk_sizes():
    generator = random.Random(1234)
    text = " ".join(TEMPLATE.format(value) for value in PII_VALUES)
    expected = " ".join(TEMPLATE.format(REDACTED) for _ in PII_VALUES)

    for _ in range(200):
        chunks = []
        position = 0

        while position < len(text):
            size = generator.randint(1, 40)
            chunks.append(text[position:position + size])
            position += size

        assert stream_through(chunks) == expected


def test_email_is_not_redacted_early_and_leaked_later():
    redactor = StreamRedactor()

    # "bob@example.co" is already a full match, but it is not finished.
    output = redactor.feed("mail bob@example.co")
    output += redactor.feed("m.au now")
    output += redactor.flush()

    assert output == f"mail {REDACTED} now"


def test_finished_words_are_released_immediately():
    redactor = StreamRedactor()

    assert redactor.feed("Hello ") == "Hello "
    assert redactor.pending == ""


def test_only_the_unfinished_word_is_held():
    redactor = StreamRedactor()

    assert redactor.feed("Hello wor") == "Hello "
    assert redactor.pending == "wor"
    assert redactor.feed("ld ") == "world "


def test_held_text_stays_bounded_without_spaces():
    redactor = StreamRedactor()
    output = ""

    # 2 MB with no whitespace: the worst case for holding text back.
    for _ in range(20_000):
        output += redactor.feed("a" * 100)
        assert len(redactor.pending) <= MAX_HOLD

    assert output + redactor.flush() == "a" * 2_000_000


def sse_body(*contents, finish=True):
    lines = [
        {"id": "1", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
    ]

    lines += [
        {"id": "1", "model": "m", "choices": [{"index": 0, "delta": {"content": text}}]}
        for text in contents
    ]

    if finish:
        lines.append(
            {"id": "1", "model": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )

    return "".join(f"data: {json.dumps(line)}\n\n" for line in lines) + "data: [DONE]\n\n"


def read_stream(response):
    events = [
        line[len("data: "):]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert events[-1] == "[DONE]"

    return [json.loads(event) for event in events[:-1]]


def stream_text(events):
    return "".join(
        choice["delta"].get("content", "")
        for event in events
        for choice in event.get("choices", [])
    )


def post(handler, payload=None):
    with TestClient(create_app(httpx.MockTransport(handler))) as client:
        return client.post(
            "/v1/chat/completions",
            json=payload or {"messages": []},
        )


def sse_response(body):
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=body,
    )


def test_gateway_redacts_pii_split_across_deltas():
    body = sse_body("Mail bo", "b@exam", "ple.com, SSN 123-", "45-6789", " thanks")

    response = post(lambda request: sse_response(body))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = read_stream(response)

    assert stream_text(events) == f"Mail {REDACTED}, SSN {REDACTED} thanks"
    assert "bob" not in response.text
    assert "6789" not in response.text


def test_gateway_keeps_the_event_envelope_and_finish_reason():
    events = read_stream(post(lambda request: sse_response(sse_body("Hello "))))

    assert events[0]["id"] == "1"
    assert events[0]["model"] == "m"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_held_text_is_sent_before_finish_reason():
    events = read_stream(post(lambda request: sse_response(sse_body("my card 4111111111111111"))))

    assert stream_text(events) == f"my card {REDACTED}"
    assert events[-1]["choices"][0]["delta"] == {"content": REDACTED}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_held_text_is_flushed_when_finish_reason_is_missing():
    events = read_stream(post(lambda request: sse_response(sse_body("bye bob@example.com", finish=False))))

    assert stream_text(events) == f"bye {REDACTED}"


def test_stream_is_forced_on_upstream():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return sse_response(sse_body("ok"))

    post(handler, {"messages": [], "stream": False})

    assert seen["stream"] is True


def test_provider_api_key_is_sent_when_configured(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return sse_response(sse_body("ok"))

    monkeypatch.setattr(server, "PROVIDER_API_KEY", "provider-key")
    post(handler)

    assert seen["authorization"] == "Bearer provider-key"


def test_uninspected_fields_are_dropped():
    event = {
        "id": "1",
        "secret_field": "bob@example.com",
        "choices": [
            {
                "index": 0,
                "logprobs": {"tokens": ["bob@example.com"]},
                "delta": {
                    "content": "ok ",
                    "tool_calls": [{"function": {"arguments": "bob@example.com"}}],
                },
            }
        ],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"

    response = post(lambda request: sse_response(body))

    assert stream_text(read_stream(response)) == "ok "
    assert "bob@example.com" not in response.text


def test_unparseable_event_is_dropped():
    body = "data: plain text bob@example.com\n\n" + sse_body("ok")

    response = post(lambda request: sse_response(body))

    assert "bob@example.com" not in response.text
    assert stream_text(read_stream(response)) == "ok"


def test_upstream_error_status_is_sanitized():
    response = post(lambda request: httpx.Response(500, text="Traceback: SECRET"))

    assert response.status_code == 502
    assert "SECRET" not in response.text
    assert response.json()["error"]["type"] == "upstream_error"


def test_upstream_connection_failure_is_sanitized():
    def handler(request):
        raise httpx.ConnectError("SECRET internal host")

    response = post(handler)

    assert response.status_code == 502
    assert "SECRET" not in response.text


def test_failure_in_the_middle_of_a_stream_is_sanitized():
    async def body():
        yield b'data: {"choices":[{"index":0,"delta":{"content":"Hello "}}]}\n\n'
        raise httpx.ReadError("SECRET internal host")

    response = post(lambda request: httpx.Response(200, content=body()))

    assert response.status_code == 200
    assert "SECRET" not in response.text

    events = read_stream(response)

    assert stream_text(events) == "Hello "
    assert events[-1]["error"]["type"] == "upstream_error"


def test_invalid_body_returns_standard_error():
    with TestClient(create_app(httpx.MockTransport(lambda request: None))) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            content=b"{bad",
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"


def test_first_token_is_sent_while_upstream_is_still_generating():
    async def scenario():
        release = asyncio.Event()

        async def body():
            yield b'data: {"choices":[{"index":0,"delta":{"content":"Hello "}}]}\n\n'
            await release.wait()
            yield b"data: [DONE]\n\n"

        stream = guarded_stream(httpx.Response(200, content=body()))

        # Passes only if nothing waits for the rest of the upstream response.
        first = await asyncio.wait_for(anext(stream), timeout=1)

        assert "Hello " in first

        release.set()
        assert [event async for event in stream] == ["data: [DONE]\n\n"]

    asyncio.run(scenario())
