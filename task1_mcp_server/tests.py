"""
These start the real server and talk raw JSON-RPC to it over stdio, so they
test the protocol and not just the Pydantic models.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent

GET = "get_customer_record"
REFUND = "trigger_refund"
REASON = "Item arrived damaged"


class Client:
    def __init__(self, stderr_file):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "task1_mcp_server.server"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
        )
        self.next_id = 1

    def send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method, params=None, request_id=None):
        if request_id is None:
            request_id = self.next_id
            self.next_id += 1

        message = {"jsonrpc": "2.0", "id": request_id, "method": method}

        if params is not None:
            message["params"] = params

        self.send(message)
        reply = json.loads(self.process.stdout.readline())

        assert reply["jsonrpc"] == "2.0"
        assert reply["id"] == request_id

        return reply

    def call(self, name, arguments, request_id=None):
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            request_id,
        )

    def initialize(self):
        reply = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        return reply

    def close(self):
        # Returns whatever is left on stdout.
        self.process.stdin.close()
        remaining = self.process.stdout.read()
        self.process.wait(timeout=10)

        return remaining


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    stderr_path = tmp_path_factory.mktemp("task1") / "stderr.log"

    with open(stderr_path, "w", encoding="utf-8") as stderr_file:
        instance = Client(stderr_file)
        instance.initialize()
        yield instance
        instance.close()


def classify(reply):
    if "error" in reply:
        return "invalid" if reply["error"]["code"] == -32602 else "other_error"

    return "tool_error" if reply["result"].get("isError") else "ok"


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        (GET, {"customer_id": "CUST-12345"}, "ok"),
        (GET, {"customer_id": "CUST-99999"}, "tool_error"),
        (GET, {"customer_id": "hackerCUST-12345"}, "invalid"),
        (GET, {"customer_id": "CUST-12345x"}, "invalid"),
        (GET, {"customer_id": "CUST-12345\n"}, "invalid"),
        (GET, {"customer_id": "CUST-1234"}, "invalid"),
        (GET, {"customer_id": "CUST-123456"}, "invalid"),
        (GET, {"customer_id": "cust-12345"}, "invalid"),
        (GET, {"customer_id": "CUST-ABCDE"}, "invalid"),
        (GET, {"customer_id": "CUST-١٢٣٤٥"}, "invalid"),
        (GET, {"customer_id": 12345}, "invalid"),
        (GET, {}, "invalid"),
        (GET, {"customer_id": "CUST-12345", "admin": True}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 49.99, "reason": REASON}, "ok"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 50, "reason": REASON}, "ok"),
        (REFUND, {"customer_id": "CUST-99999", "amount": 10.0, "reason": REASON}, "tool_error"),
        (REFUND, {"customer_id": " CUST-12345 ", "amount": 5, "reason": REASON}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 0, "reason": REASON}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": -25.5, "reason": REASON}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": "50", "reason": REASON}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": True, "reason": REASON}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": None, "reason": REASON}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": float("inf"), "reason": REASON}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 49.99, "reason": "bad"}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 49.99, "reason": " " * 10}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 49.99, "reason": "123456789"}, "invalid"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 49.99, "reason": "1234567890"}, "ok"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 49.99, "reason": "   short   "}, "ok"),
        (REFUND, {"customer_id": "CUST-12345", "amount": 49.99}, "invalid"),
        ("delete_everything", {}, "invalid"),
    ],
)
def test_tool_calls(client, tool, arguments, expected):
    assert classify(client.call(tool, arguments)) == expected


def test_tools_list_advertises_the_validation_rules(client):
    tools = {
        tool["name"]: tool["inputSchema"]
        for tool in client.request("tools/list")["result"]["tools"]
    }

    assert set(tools) == {GET, REFUND}

    refund = tools[REFUND]

    assert refund["properties"]["reason"]["minLength"] == 10
    assert refund["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert refund["additionalProperties"] is False


def test_all_invalid_fields_are_reported_together(client):
    reply = client.call(REFUND, {"customer_id": "CUST-1", "amount": -5, "reason": "no"})

    assert reply["error"]["code"] == -32602
    assert {item["field"] for item in reply["error"]["data"]} == {
        "customer_id",
        "amount",
        "reason",
    }


def test_validation_error_does_not_expose_pydantic_internals(client):
    reply = client.call(GET, {"customer_id": "BAD"})

    assert "pydantic.dev" not in json.dumps(reply)


def test_string_request_id_is_echoed_back(client):
    reply = client.call(GET, {"customer_id": "CUST-12345"}, request_id="abc-1")

    assert reply["id"] == "abc-1"


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr(tmp_path):
    stderr_path = tmp_path / "stderr.log"

    with open(stderr_path, "w", encoding="utf-8") as stderr_file:
        client = Client(stderr_file)

        # Every reply is parsed as JSON inside request().
        client.initialize()
        client.call(GET, {"customer_id": "CUST-12345"})
        client.call(GET, {"customer_id": "BAD"})
        client.call(REFUND, {"customer_id": "CUST-12345", "amount": 1, "reason": REASON})

        # Nothing else may be left over on stdout.
        assert client.close().strip() == ""

    logs = stderr_path.read_text(encoding="utf-8")

    assert "Starting customer MCP server" in logs
    assert "Customer record served" in logs
