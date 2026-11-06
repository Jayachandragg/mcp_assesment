import httpx
import pytest
from fastapi.testclient import TestClient

from task2_mcp_gateway import mock_downstream
from task2_mcp_gateway.server import create_app


ADMIN = {"Authorization": "Bearer admin-token"}
VIEWER = {"Authorization": "Bearer viewer-token"}


def tool_call(name, request_id=1):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name},
    }


def client_for(handler):
    return TestClient(create_app(httpx.MockTransport(handler)))


@pytest.fixture
def client():
    # The gateway talking to the real mock downstream app.
    mock_downstream.received.clear()
    transport = httpx.ASGITransport(app=mock_downstream.app)

    with TestClient(create_app(transport)) as test_client:
        yield test_client


def test_missing_authentication_is_rejected(client):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == -32000
    assert mock_downstream.received == []


@pytest.mark.parametrize(
    "value",
    ["Bearer invalid-token", "Basic admin-token", "admin-token", "Bearer"],
)
def test_bad_authorization_header_is_rejected(client, value):
    response = client.post(
        "/mcp",
        headers={"Authorization": value},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )

    assert response.status_code == 401
    assert mock_downstream.received == []


def test_tools_list_is_forwarded_transparently(client):
    response = client.post(
        "/mcp",
        headers=VIEWER,
        json={"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["id"] == 7
    assert [tool["name"] for tool in body["result"]["tools"]] == [
        "get_customer_record",
        "admin_reset_key",
    ]


def test_viewer_cannot_call_admin_tool(client):
    response = client.post("/mcp", headers=VIEWER, json=tool_call("admin_reset_key", 42))

    assert response.status_code == 200
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 42,
        "error": {"code": -32001, "message": "Unauthorized Tool Call"},
    }
    assert mock_downstream.received == []


def test_admin_can_call_admin_tool(client):
    response = client.post("/mcp", headers=ADMIN, json=tool_call("admin_reset_key"))

    assert response.status_code == 200
    assert "admin_reset_key executed" in response.text
    assert len(mock_downstream.received) == 1


def test_viewer_can_call_normal_tool(client):
    response = client.post("/mcp", headers=VIEWER, json=tool_call("get_customer_record"))

    assert response.status_code == 200
    assert "get_customer_record executed" in response.text


@pytest.mark.parametrize("name", ["get_admin_report", "administrator", "ADMIN_reset_key"])
def test_only_the_exact_admin_prefix_is_restricted(client, name):
    response = client.post("/mcp", headers=VIEWER, json=tool_call(name))

    assert "error" not in response.json()
    assert len(mock_downstream.received) == 1


def test_notification_cannot_reach_admin_tool(client):
    payload = tool_call("admin_reset_key")
    del payload["id"]

    response = client.post("/mcp", headers=VIEWER, json=payload)

    assert response.json()["error"]["code"] == -32001
    assert mock_downstream.received == []


def test_duplicate_json_keys_cannot_hide_admin_tool(client):
    # Which "name" wins depends on the parser. Only the parsed value is forwarded.
    raw = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"get_customer_record","name":"admin_reset_key"}}'
    )

    response = client.post(
        "/mcp",
        headers={**VIEWER, "Content-Type": "application/json"},
        content=raw,
    )

    assert response.json()["error"]["code"] == -32001
    assert mock_downstream.received == []


def test_invalid_json_returns_parse_error(client):
    response = client.post("/mcp", headers=VIEWER, content=b"{not json")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


@pytest.mark.parametrize(
    "payload",
    [
        [tool_call("admin_reset_key")],
        "text",
        {"id": 1, "method": "tools/list"},
        {"jsonrpc": "1.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "2.0", "id": 1, "method": 5},
        {"jsonrpc": "2.0", "id": True, "method": "tools/list"},
    ],
)
def test_invalid_request_is_not_forwarded(client, payload):
    response = client.post("/mcp", headers=ADMIN, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600
    assert mock_downstream.received == []


@pytest.mark.parametrize(
    "params",
    [None, {}, {"name": 5}, {"name": ""}, ["admin_reset_key"]],
)
def test_invalid_tool_call_params_are_not_forwarded(client, params):
    response = client.post(
        "/mcp",
        headers=VIEWER,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params},
    )

    assert response.json()["error"]["code"] == -32602
    assert response.json()["id"] == 1
    assert mock_downstream.received == []


def test_headers_are_forwarded_but_token_is_not():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(
            200,
            headers={"x-downstream": "value"},
            json={"jsonrpc": "2.0", "id": 1, "result": {}},
        )

    with client_for(handler) as client:
        response = client.post(
            "/mcp",
            headers={
                **VIEWER,
                "Accept": "application/json, text/event-stream",
                "Mcp-Session-Id": "session-1",
                "X-Trace-Id": "trace-1",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )

    assert seen["accept"] == "application/json, text/event-stream"
    assert seen["mcp-session-id"] == "session-1"
    assert seen["x-trace-id"] == "trace-1"
    assert "authorization" not in seen
    assert response.headers["x-downstream"] == "value"


def test_sse_reply_passes_through():
    event = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{}}\n\n'

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "mcp-session-id": "session-2"},
            text=event,
        )

    with client_for(handler) as client:
        response = client.post(
            "/mcp",
            headers=VIEWER,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["mcp-session-id"] == "session-2"
    assert response.text == event


def test_notification_gets_empty_202(client):
    response = client.post(
        "/mcp",
        headers=VIEWER,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )

    assert response.status_code == 202
    assert response.content == b""


def test_downstream_failure_is_sanitized():
    def handler(request):
        raise httpx.ConnectError("SECRET internal host 10.0.0.5")

    with client_for(handler) as client:
        response = client.post("/mcp", headers=VIEWER, json=tool_call("get_customer_record", 9))

    assert response.status_code == 502
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 9,
        "error": {"code": -32002, "message": "Downstream MCP server unavailable"},
    }
