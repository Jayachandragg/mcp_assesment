from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
UNAUTHENTICATED = -32000
UNAUTHORIZED_TOOL_CALL = -32001
DOWNSTREAM_UNAVAILABLE = -32002


class JsonRpcRequest(BaseModel):
    # Other fields are left alone so forwarding stays transparent.
    model_config = ConfigDict(strict=True, extra="allow")

    jsonrpc: Literal["2.0"]
    method: str
    id: int | str | None = None
    params: dict[str, Any] | list[Any] | None = None


class ToolCallParams(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")

    name: str = Field(min_length=1)


def rpc_error(request_id: int | str | None, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }
