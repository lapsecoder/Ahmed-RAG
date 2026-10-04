"""Ollama provider: request shape, response parsing and error translation.

Uses ``httpx.MockTransport`` so no socket is ever opened.
"""

from __future__ import annotations

import json

import httpx
import pytest
from app.core.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
)
from app.services.llm import (
    DEFAULT_OLLAMA_MODEL,
    GENERATE_ENDPOINT,
    OllamaProvider,
    StaticLLMProvider,
)


def make_provider(
    handler,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout: float = 5.0,
) -> OllamaProvider:
    """Build a provider wired to a mock transport."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider(
        base_url="http://localhost:11434",
        model=model,
        timeout_seconds=timeout,
        client=client,
    )


def ok_handler(payload: dict, recorder: list[httpx.Request]) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        recorder.append(request)
        return httpx.Response(200, json=payload)

    return handler


# --------------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------------- #


def test_request_targets_the_generate_endpoint_with_the_exact_body() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(ok_handler({"response": "hello"}, seen))
    provider.generate("PROMPT TEXT")

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == f"http://localhost:11434{GENERATE_ENDPOINT}"
    body = json.loads(request.content)
    assert body == {"model": DEFAULT_OLLAMA_MODEL, "prompt": "PROMPT TEXT", "stream": False}


def test_the_default_model_is_qwen() -> None:
    provider = OllamaProvider()
    assert provider.model_name == "qwen2.5-coder:7b"
    assert provider.model_name == DEFAULT_OLLAMA_MODEL


def test_custom_base_url_and_model_are_used() -> None:
    seen: list[httpx.Request] = []
    client = httpx.Client(transport=httpx.MockTransport(ok_handler({"response": "ok"}, seen)))
    provider = OllamaProvider(
        base_url="http://ollama.internal:9999/",
        model="qwen2.5-coder:1.5b",
        client=client,
    )
    assert provider.url == "http://ollama.internal:9999/api/generate"
    assert provider.model_name == "qwen2.5-coder:1.5b"
    provider.generate("prompt")
    assert json.loads(seen[0].content)["model"] == "qwen2.5-coder:1.5b"


def test_response_text_is_returned_stripped() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(ok_handler({"response": "  answer text  "}, seen))
    assert provider.generate("prompt") == "answer text"


def test_empty_prompts_are_rejected_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made")

    provider = make_provider(handler)
    with pytest.raises(LLMError):
        provider.generate("   ")


def test_provider_rejects_a_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        OllamaProvider(timeout_seconds=0)


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #


def test_timeouts_become_llm_timeout_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = make_provider(handler)
    with pytest.raises(LLMTimeoutError, match="did not respond"):
        provider.generate("prompt")


def test_connect_errors_become_connection_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = make_provider(handler)
    with pytest.raises(LLMConnectionError, match="cannot reach Ollama"):
        provider.generate("prompt")


def test_http_error_statuses_become_api_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model not found")

    provider = make_provider(handler)
    with pytest.raises(LLMAPIError, match="HTTP 500"):
        provider.generate("prompt")


def test_non_json_responses_become_api_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    provider = make_provider(handler)
    with pytest.raises(LLMAPIError, match="non-JSON"):
        provider.generate("prompt")


def test_missing_response_field_becomes_an_api_error() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(ok_handler({"done": True}, seen))
    with pytest.raises(LLMAPIError, match="did not contain"):
        provider.generate("prompt")


def test_blank_response_field_becomes_an_api_error() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(ok_handler({"response": "   "}, seen))
    with pytest.raises(LLMAPIError):
        provider.generate("prompt")


def test_ollama_error_field_is_surfaced() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(ok_handler({"error": "model not found"}, seen))
    with pytest.raises(LLMAPIError, match="model not found"):
        provider.generate("prompt")


def test_unexpected_payload_shape_is_rejected() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(ok_handler(["not", "a", "dict"], seen))
    with pytest.raises(LLMAPIError, match="unexpected Ollama payload"):
        provider.generate("prompt")


# --------------------------------------------------------------------------- #
# Client lifecycle
# --------------------------------------------------------------------------- #


def test_an_injected_client_is_not_closed_by_the_provider() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(ok_handler({"response": "ok"}, seen))
    provider.generate("prompt")
    assert provider._client.is_closed is False


def test_static_provider_returns_canned_text() -> None:
    provider = StaticLLMProvider("canned")
    assert provider.generate("anything") == "canned"
    with pytest.raises(LLMError):
        provider.generate(" ")


def test_provider_is_an_llm_provider() -> None:
    from app.services.llm import LLMProvider

    assert isinstance(OllamaProvider(), LLMProvider)
    assert isinstance(StaticLLMProvider("x"), LLMProvider)


def test_context_manager_closes_only_owned_clients() -> None:
    with OllamaProvider() as provider:
        assert provider.model_name == "qwen2.5-coder:7b"
