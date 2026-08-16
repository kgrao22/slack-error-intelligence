"""One HTTP layer for every provider.

Slack, Voyage and Anthropic are all JSON over HTTPS, and all three fail in the
same three ways: transient 5xx, connection resets, and 429 with a Retry-After
header. Rather than importing three SDKs with three retry philosophies, they
share this client so the backoff, jitter, timeout and rate-limit policy is
written once and tested once.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ApiError(RuntimeError):
    def __init__(self, provider: str, status: int, body: str) -> None:
        super().__init__(f"{provider} returned HTTP {status}: {body[:400]}")
        self.provider = provider
        self.status = status
        self.body = body


class HttpClient:
    def __init__(
        self,
        provider: str,
        *,
        timeout: float = 45.0,
        max_attempts: int = 5,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.provider = provider
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self._client = httpx.Client(timeout=timeout, follow_redirects=False)

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _sleep_for(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    # Slack tells us exactly how long to wait; honour it rather
                    # than guessing, plus a small margin.
                    return min(float(retry_after) + 0.5, self.max_backoff)
                except ValueError:
                    pass
        backoff = min(self.base_backoff * (2**attempt), self.max_backoff)
        return backoff * (0.5 + random.random() * 0.5)  # full-ish jitter

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            response: httpx.Response | None = None
            try:
                response = self._client.request(
                    method, url, headers=headers, params=params, json=json_body
                )
                if response.status_code in RETRY_STATUS:
                    if attempt == self.max_attempts - 1:
                        raise ApiError(self.provider, response.status_code, response.text)
                    delay = self._sleep_for(attempt, response)
                    log.warning(
                        "%s HTTP %s, retrying in %.1fs (attempt %d/%d)",
                        self.provider,
                        response.status_code,
                        delay,
                        attempt + 1,
                        self.max_attempts,
                    )
                    time.sleep(delay)
                    continue
                if response.status_code >= 400:
                    raise ApiError(self.provider, response.status_code, response.text)
                return response.json()
            except (httpx.TransportError, httpx.HTTPError) as exc:
                last_error = exc
                if attempt == self.max_attempts - 1:
                    break
                delay = self._sleep_for(attempt, response)
                log.warning("%s transport error %s, retrying in %.1fs", self.provider, exc, delay)
                time.sleep(delay)

        raise ApiError(self.provider, 0, f"exhausted retries: {last_error}")
