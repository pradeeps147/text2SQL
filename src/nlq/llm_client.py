"""Small provider abstraction for local Ollama and hosted OpenAI-compatible LLMs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx
import ollama

DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
DEFAULT_MODEL = os.environ.get(
    "LLM_MODEL",
    os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if DEFAULT_PROVIDER == "openai"
    else os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
)
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


class LLMConnectionError(RuntimeError):
    """Raised when the configured model provider is unavailable or invalid."""


class LLMClient(Protocol):
    model: str

    def check_connection(self) -> None: ...

    def generate(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict] = None,
        temperature: float = 0.0,
    ) -> str: ...


class OllamaConnectionError(LLMConnectionError):
    """Raised when the local Ollama daemon is unreachable, or the configured
    model has not been pulled yet."""


@dataclass
class OllamaClient:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST

    def __post_init__(self) -> None:
        self._client = ollama.Client(host=self.host)

    def check_connection(self) -> None:
        """Raise OllamaConnectionError with an actionable message if Ollama
        isn't running or the model isn't available. Call this once at
        startup so failures surface immediately instead of mid-query."""
        try:
            response = self._client.list()
        except Exception as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.host}. Is the Ollama app/daemon "
                f"running? (Install: https://ollama.com, then run `ollama serve`). "
                f"Original error: {exc}"
            ) from exc

        available = {m.get("model") or m.get("name") for m in response.get("models", [])}
        if not any(self.model == m or (m and m.split(":")[0] == self.model.split(":")[0]) for m in available):
            raise OllamaConnectionError(
                f"Model '{self.model}' is not available locally. Pull it first with: "
                f"`ollama pull {self.model}`. Available models: {sorted(filter(None, available))}"
            )

    def generate(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict] = None,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat request and return the raw text content. If
        `json_schema` is provided, asks the model to constrain its output to
        that structure (Ollama structured-output mode)."""
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            kwargs["format"] = json_schema

        response = self._client.chat(**kwargs)
        return response["message"]["content"]


class OpenAIConnectionError(LLMConnectionError):
    """Raised when the configured hosted API cannot be reached or authenticated."""


@dataclass
class OpenAICompatibleClient:
    """Minimal Chat Completions client with JSON Schema structured outputs."""

    model: str = DEFAULT_MODEL
    api_key: str = ""
    base_url: str = DEFAULT_OPENAI_BASE_URL

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = self.base_url.rstrip("/")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def check_connection(self) -> None:
        if not self.api_key:
            raise OpenAIConnectionError(
                "OPENAI_API_KEY is not configured for the hosted LLM provider."
            )
        try:
            response = httpx.get(f"{self.base_url}/models", headers=self._headers, timeout=15.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OpenAIConnectionError(
                f"Hosted LLM authentication or model access check failed ({exc.response.status_code})."
            ) from exc
        except httpx.RequestError as exc:
            raise OpenAIConnectionError(f"Could not reach the hosted LLM API: {exc}") from exc

    def generate(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict] = None,
        temperature: float = 0.0,
    ) -> str:
        if not self.api_key:
            raise OpenAIConnectionError("OPENAI_API_KEY is not configured.")

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "developer", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "tenarai_structured_response",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise OpenAIConnectionError(
                f"Hosted LLM request failed ({exc.response.status_code})."
            ) from exc
        except (httpx.RequestError, KeyError, IndexError, ValueError, TypeError) as exc:
            raise OpenAIConnectionError(f"Hosted LLM returned an invalid response: {exc}") from exc
        if not isinstance(content, str):
            raise OpenAIConnectionError("Hosted LLM response did not contain text content.")
        return content


def build_llm_client(model: str | None = None) -> LLMClient:
    """Build the provider selected by ``LLM_PROVIDER`` (ollama or openai)."""
    selected_model = model or DEFAULT_MODEL
    if DEFAULT_PROVIDER == "ollama":
        return OllamaClient(model=selected_model)
    if DEFAULT_PROVIDER == "openai":
        return OpenAICompatibleClient(model=selected_model)
    raise LLMConnectionError(
        f"Unsupported LLM_PROVIDER '{DEFAULT_PROVIDER}'. Use 'ollama' or 'openai'."
    )
