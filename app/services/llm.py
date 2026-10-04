"""LLM provider abstraction plus the local Ollama implementation.

Only local, free providers belong here. The default model is
``qwen2.5-coder:7b`` served by Ollama on ``POST /api/generate``.
"""

from __future__ import annotations

import abc
from types import TracebackType
from typing import Any, Final

import httpx

from app.core.exceptions import LLMAPIError, LLMConnectionError, LLMError, LLMTimeoutError
from app.core.logging import get_logger

logger = get_logger("services.llm")

DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: Final[str] = "qwen2.5-coder:7b"
GENERATE_ENDPOINT: Final[str] = "/api/generate"


class LLMProvider(abc.ABC):
    """Interface every generation backend must implement."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Identifier of the model that will answer prompts."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``.

        Raises:
            LLMError: On timeout, connection failure, or an unusable response.
        """


class OllamaProvider(LLMProvider):
    """Local Ollama generation via ``POST /api/generate`` (non-streaming)."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout_seconds: float = 120.0,
        *,
        client: httpx.Client | None = None,
        endpoint: str = GENERATE_ENDPOINT,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = float(timeout_seconds)
        self._endpoint = endpoint
        self._client = client
        self._owns_client = client is None

    @property
    def model_name(self) -> str:
        """The configured Ollama model name."""
        return self._model

    @property
    def url(self) -> str:
        """Full generation endpoint URL."""
        return f"{self._base_url}{self._endpoint}"

    def _get_client(self) -> tuple[httpx.Client, bool]:
        if self._client is not None:
            return self._client, False
        return httpx.Client(timeout=self._timeout), True

    def generate(self, prompt: str) -> str:
        """Generate a completion for ``prompt``.

        Raises:
            LLMError: If the prompt is empty, the server is unreachable, the
                request times out, or the payload is not usable.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("prompt must be a non-empty string")

        payload: dict[str, Any] = {"model": self._model, "prompt": prompt, "stream": False}
        client, owned = self._get_client()
        try:
            response = client.post(self.url, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama did not respond within {self._timeout:.0f}s at {self.url}"
            ) from exc
        except httpx.TransportError as exc:
            raise LLMConnectionError(f"cannot reach Ollama at {self.url}: {exc}") from exc
        except httpx.HTTPError as exc:  # pragma: no cover - defensive
            raise LLMError(f"HTTP error while calling Ollama: {exc}") from exc
        finally:
            if owned:
                client.close()

        if response.status_code >= 400:
            detail = response.text.strip()[:300]
            raise LLMAPIError(
                f"Ollama returned HTTP {response.status_code} for model {self._model!r}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMAPIError(f"Ollama returned a non-JSON response: {exc}") from exc
        return self._extract_completion(data)

    def _extract_completion(self, data: Any) -> str:
        if not isinstance(data, dict):
            raise LLMAPIError(f"unexpected Ollama payload type: {type(data).__name__}")
        if data.get("error"):
            raise LLMAPIError(f"Ollama reported an error: {str(data['error'])[:300]}")
        completion = data.get("response")
        if not isinstance(completion, str) or not completion.strip():
            raise LLMAPIError("Ollama response did not contain a non-empty 'response' field")
        return completion.strip()

    def close(self) -> None:
        """Close the client if this provider created it."""
        if self._owns_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> OllamaProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class StaticLLMProvider(LLMProvider):
    """A provider that always returns the same text.

    Used by tests and by the deterministic no-context path, never as a silent
    production fallback: instantiating it is always an explicit decision.
    """

    def __init__(self, text: str, *, model_name: str = "static") -> None:
        self._text = text
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """The configured model name."""
        return self._model_name

    def generate(self, prompt: str) -> str:
        """Return the canned response."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("prompt must be a non-empty string")
        return self._text
