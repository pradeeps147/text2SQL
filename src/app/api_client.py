"""HTTP client used by the Tenarai Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class ApiError(RuntimeError):
    """A backend connection or response error safe to show in the UI."""


@dataclass(frozen=True)
class TenaraiApiClient:
    base_url: str
    admin_key: str = ""

    def _url(self, path: str) -> str:
        base_url = self.base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        return f"{base_url}{path}"

    @property
    def _admin_headers(self) -> dict[str, str]:
        return {"X-Tenarai-Admin-Key": self.admin_key} if self.admin_key else {}

    @staticmethod
    def _parse(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiError(f"Backend returned an invalid response ({response.status_code}).") from exc
        if response.is_error:
            detail = payload.get("detail", payload)
            if isinstance(detail, dict):
                detail = detail.get("message", detail)
            raise ApiError(str(detail))
        return payload

    def health(self) -> dict:
        try:
            response = httpx.get(self._url("/api/v1/health"), timeout=3.0)
        except httpx.RequestError as exc:
            raise ApiError(f"Backend is unavailable at {self.base_url}: {exc}") from exc
        return self._parse(response)

    def dataset(self) -> dict:
        try:
            response = httpx.get(self._url("/api/v1/dataset"), timeout=10.0)
        except httpx.RequestError as exc:
            raise ApiError(f"Could not read dataset status: {exc}") from exc
        return self._parse(response)

    def upload(self, uploads: list[tuple[str, bytes]]) -> dict:
        files = [("files", (name, content, "text/csv")) for name, content in uploads]
        try:
            response = httpx.post(
                self._url("/api/v1/datasets/upload"),
                files=files,
                headers=self._admin_headers,
                timeout=120.0,
            )
        except httpx.RequestError as exc:
            raise ApiError(f"Upload failed before the backend responded: {exc}") from exc
        return self._parse(response)

    def chat(self, question: str, model: str) -> dict:
        try:
            response = httpx.post(
                self._url("/api/v1/chat"),
                json={"question": question, "model": model},
                timeout=180.0,
            )
        except httpx.RequestError as exc:
            raise ApiError(f"Chat request failed before the backend responded: {exc}") from exc
        return self._parse(response)

    def rag(self) -> dict:
        try:
            response = httpx.get(self._url("/api/v1/rag"), timeout=10.0)
        except httpx.RequestError as exc:
            raise ApiError(f"Could not load RAG details: {exc}") from exc
        return self._parse(response)

    def logs(self, limit: int = 200, level: str | None = None) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if level:
            params["level"] = level
        try:
            response = httpx.get(
                self._url("/api/v1/logs"),
                params=params,
                headers=self._admin_headers,
                timeout=10.0,
            )
        except httpx.RequestError as exc:
            raise ApiError(f"Could not load backend logs: {exc}") from exc
        return self._parse(response)
