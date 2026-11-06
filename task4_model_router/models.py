import math

from pydantic import BaseModel, ConfigDict, Field


# Rough rule of thumb for English text.
CHARS_PER_TOKEN = 4


class CompletionRequest(BaseModel):
    # Extra fields like temperature just get passed on to the provider.
    model_config = ConfigDict(strict=True, extra="allow")

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=256, gt=0)

    @property
    def estimated_tokens(self) -> int:
        # Worked out here, because a client could lie about its own count.
        return math.ceil(len(self.prompt) / CHARS_PER_TOKEN) + self.max_tokens


class GatewayError(Exception):
    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        retry_after: int | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.retry_after = retry_after
