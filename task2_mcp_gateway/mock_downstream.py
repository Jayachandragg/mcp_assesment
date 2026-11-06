"""
Fake MCP server to sit behind the gateway.

Run: uvicorn task2_mcp_gateway.mock_downstream:app --port 8001
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


app = FastAPI(title="Mock Downstream MCP Server")

TOOLS = [
    {
        "name": "get_customer_record",
        "description": "Look up a customer record.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "admin_reset_key",
        "description": "Reset an API key.",
        "inputSchema": {"type": "object"},
    },
]

# Lets the tests check that a blocked call never got here.
received: list[dict] = []


@app.post("/mcp")
async def mcp(request: Request):
    payload = await request.json()
    received.append(payload)

    # Notifications have no id and get no JSON-RPC reply.
    if "id" not in payload:
        return Response(status_code=202)

    if payload["method"] == "tools/list":
        result = {"tools": TOOLS}

    elif payload["method"] == "tools/call":
        name = payload["params"]["name"]
        result = {"content": [{"type": "text", "text": f"{name} executed"}]}

    else:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "error": {"code": -32601, "message": "Method not found"},
            }
        )

    return JSONResponse({"jsonrpc": "2.0", "id": payload["id"], "result": result})
