"""
Customer MCP server over stdio.

How errors are reported:
  Unknown tool or bad arguments -> JSON-RPC error -32602
  Good input, customer missing  -> normal result with isError: true
  A bug in the server           -> JSON-RPC error -32603

This uses the low-level Server because FastMCP reports validation failures as
isError results, and the task asks for JSON-RPC error codes.
"""

import json
import logging
import sys

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
from pydantic import BaseModel, ValidationError

from task1_mcp_server.models import GetCustomerRecordArgs, TriggerRefundArgs


# stdout belongs to JSON-RPC, so logs have to go to stderr. No print() anywhere.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("customer-service")


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "get_customer_record": (
        GetCustomerRecordArgs,
        "Look up a customer record by customer ID (format CUST-12345).",
    ),
    "trigger_refund": (
        TriggerRefundArgs,
        "Issue a refund to a customer. amount must be a positive number; "
        "reason must be at least 10 characters.",
    ),
}

# Mock customer data.
CUSTOMERS = {
    "CUST-12345": {
        "customer_id": "CUST-12345",
        "name": "John Doe",
        "email": "john@example.com",
        "status": "active",
    },
    "CUST-67890": {
        "customer_id": "CUST-67890",
        "name": "Jane Smith",
        "email": "jane@example.com",
        "status": "active",
    },
}


def text_result(payload: dict | str, is_error: bool = False) -> CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload)

    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=is_error,
    )


async def list_tools(ctx, params) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name=name,
                description=description,
                inputSchema=model.model_json_schema(),
            )
            for name, (model, description) in TOOLS.items()
        ]
    )


async def call_tool(ctx, params) -> CallToolResult:
    if params.name not in TOOLS:
        logger.warning("Unknown tool requested: %s", params.name)
        raise MCPError(code=INVALID_PARAMS, message=f"Unknown tool: {params.name}")

    model, _ = TOOLS[params.name]

    try:
        args = model.model_validate(params.arguments or {})
    except ValidationError as error:
        details = [
            {
                "field": ".".join(map(str, item["loc"])) or "(root)",
                "msg": item["msg"],
            }
            for item in error.errors()
        ]
        logger.warning("Rejected %s: %s", params.name, details)
        raise MCPError(code=INVALID_PARAMS, message="Invalid params", data=details)

    try:
        return run_tool(params.name, args)
    except Exception:
        # Log the details, but don't send them to the client.
        logger.exception("Tool %s crashed", params.name)
        raise MCPError(code=INTERNAL_ERROR, message="Internal error")


def run_tool(name: str, args: BaseModel) -> CallToolResult:
    if args.customer_id not in CUSTOMERS:
        logger.info("Customer not found: %s", args.customer_id)
        return text_result(f"Customer {args.customer_id} not found", is_error=True)

    if name == "get_customer_record":
        logger.info("Customer record served: %s", args.customer_id)
        return text_result(CUSTOMERS[args.customer_id])

    # This is a mock refund operation.
    logger.info("Refund initiated: %s %.2f", args.customer_id, args.amount)
    return text_result(
        {
            "status": "refund_initiated",
            "customer_id": args.customer_id,
            "amount": args.amount,
            "reason": args.reason,
        }
    )


server = Server(
    "customer-service",
    version="1.0.0",
    on_list_tools=list_tools,
    on_call_tool=call_tool,
)


async def main() -> None:
    logger.info("Starting customer MCP server on stdio")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    anyio.run(main)
