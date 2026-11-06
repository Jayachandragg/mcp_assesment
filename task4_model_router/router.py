import asyncio
import logging
import os

import httpx

from task4_model_router.models import GatewayError


logger = logging.getLogger("model-router")

PRIMARY_URL = os.getenv("PRIMARY_URL", "http://localhost:9100/v1/completions")
SECONDARY_URL = os.getenv("SECONDARY_URL", "http://localhost:9200/v1/completions")
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY")
SECONDARY_API_KEY = os.getenv("SECONDARY_API_KEY")

PRIMARY_TIMEOUT_SECONDS = 3.0
SECONDARY_TIMEOUT_SECONDS = 10.0


class ModelRouter:
    def __init__(
        self,
        client: httpx.AsyncClient,
        primary_url: str = PRIMARY_URL,
        secondary_url: str = SECONDARY_URL,
        primary_timeout: float = PRIMARY_TIMEOUT_SECONDS,
        secondary_timeout: float = SECONDARY_TIMEOUT_SECONDS,
        primary_api_key: str | None = PRIMARY_API_KEY,
        secondary_api_key: str | None = SECONDARY_API_KEY,
    ):
        self.client = client
        self.primary_url = primary_url
        self.secondary_url = secondary_url
        self.primary_timeout = primary_timeout
        self.secondary_timeout = secondary_timeout
        self.primary_api_key = primary_api_key
        self.secondary_api_key = secondary_api_key

    async def post(
        self,
        url: str,
        payload: dict,
        timeout: float,
        api_key: str | None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

        # Deadline for the whole call. httpx's timeout is per read, so a provider
        # sending a byte at a time could run well past 3 seconds. Timing out
        # here also cancels the request.
        async with asyncio.timeout(timeout):
            return await self.client.post(url, json=payload, headers=headers)

    async def complete(self, payload: dict) -> dict:
        try:
            response = await self.post(
                self.primary_url,
                payload,
                self.primary_timeout,
                self.primary_api_key,
            )
        except TimeoutError:
            logger.warning("Primary model timed out, using secondary")
            return await self.complete_on_secondary(payload)
        except httpx.HTTPError as error:
            logger.warning("Primary model failed: %r", error)
            raise GatewayError(502, "upstream_error", "Model provider unavailable")

        if response.status_code == 429:
            logger.warning("Primary model returned 429, using secondary")
            return await self.complete_on_secondary(payload)

        return self.read_result(response)

    async def complete_on_secondary(self, payload: dict) -> dict:
        try:
            response = await self.post(
                self.secondary_url,
                payload,
                self.secondary_timeout,
                self.secondary_api_key,
            )
        except TimeoutError:
            logger.warning("Secondary model timed out")
            raise GatewayError(504, "upstream_timeout", "Model providers timed out")
        except httpx.HTTPError as error:
            logger.warning("Secondary model failed: %r", error)
            raise GatewayError(502, "upstream_error", "Model provider unavailable")

        if response.status_code == 429:
            logger.warning("Secondary model returned 429")
            raise GatewayError(503, "upstream_overloaded", "Model providers are overloaded")

        return self.read_result(response)

    def read_result(self, response: httpx.Response) -> dict:
        # Error bodies can contain stack traces or internal hosts, so they never
        # go back to the client.
        if response.status_code != 200:
            logger.warning("Model provider returned status %s", response.status_code)
            raise GatewayError(502, "upstream_error", "Model provider unavailable")

        try:
            result = response.json()
        except ValueError:
            result = None

        if not isinstance(result, dict):
            logger.warning("Model provider returned an unreadable body")
            raise GatewayError(502, "upstream_error", "Model provider unavailable")

        return result
