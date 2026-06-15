"""Tiny HTTP client with retry/backoff and rate-limit awareness.

Shared by all API clients so retry behaviour is consistent and testable.
"""

from __future__ import annotations

import time
from typing import Any

import requests


class HttpError(RuntimeError):
    pass


class HttpClient:
    def __init__(
        self,
        base_url: str = "",
        *,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.session = session or requests.Session()

    def get_json(self, path: str = "", params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}" if path.startswith("/") or not self.base_url else f"{self.base_url}/{path}"
        if not self.base_url:
            url = path
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise HttpError(f"retryable status {resp.status_code} for {url}")
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, HttpError) as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.backoff_base ** attempt)
        raise HttpError(f"GET failed after {self.max_retries} attempts: {url}: {last_err}")
