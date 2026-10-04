"""FastAPI routes and application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
)
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.core.exceptions import (
    AhmedRAGError,
    ConfigurationError,
    DependencyMissingError,
    LLMError,
)
from app.core.logging import configure_logging, get_logger
from app.models.chat import ChatResult
from app.services.responses import LLM_UNAVAILABLE_RESPONSE

logger = get_logger("api.routes")

API_PREFIX = "/api"

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Invalid request payload"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
    503: {"model": ErrorResponse, "description": "A local dependency is unavailable"},
}


def get_container(request: Request) -> Container:
    """FastAPI dependency returning the startup-built container."""
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only if lifespan did not run
        raise ConfigurationError("application container is not initialised")
    return container


def _to_response(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        response=result.response,
        classification=result.classification,
        outcome=result.outcome,
        sources=result.sources,
        retrieval_scores=result.retrieval_scores,
        llm_used=result.llm_used,
        injection_rule_ids=result.injection_rule_ids,
    )


def create_app(
    container: Container | None = None,
    *,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        container: Pre-built container. When provided, startup will not load any
            model -- this is how the test suite injects fakes. The app is ready
            to serve immediately.
        settings: Configuration used when ``container`` is not supplied.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging((settings or get_settings()).log_level)
        if container is not None:
            logger.info("using the injected application container")
        else:
            app.state.container = build_container(settings)
            logger.info("application container ready")
        try:
            yield
        finally:
            if hasattr(app.state, "container"):
                delattr(app.state, "container")

    app = FastAPI(
        title="Ahmed-RAG",
        version="0.1.0",
        description=(
            "Local-first RAG backend for Ahmed's portfolio. Retrieved documents are "
            "treated as untrusted data; the backend performs prompt-injection "
            "classification before anything is retrieved or generated."
        ),
        lifespan=lifespan,
    )
    if container is not None:
        # An injected container is fully built already, so the app is usable
        # without waiting for startup. Tests rely on this.
        app.state.container = container
    _register_routes(app)
    _register_exception_handlers(app)
    return app


def _register_routes(app: FastAPI) -> None:

    @app.post(
        f"{API_PREFIX}/chat",
        response_model=ChatResponse,
        responses=_ERROR_RESPONSES,
        summary="Ask the portfolio assistant a question",
    )
    def chat(
        payload: ChatRequest,
        container: Container = Depends(get_container),
    ) -> ChatResponse:
        """Classify, retrieve, compose and vet the answer.

        Injection attempts, out-of-scope questions and empty retrievals are
        answered deterministically. No language model is invoked at any point.
        """
        limit = container.settings.max_message_chars
        if len(payload.message) > limit:
            raise ConfigurationError(f"message exceeds the {limit} character limit")
        result = container.chat_service.chat(payload.message)
        return _to_response(result)

    @app.get(
        f"{API_PREFIX}/health",
        response_model=HealthResponse,
        summary="Report service health",
    )
    def health(container: Container = Depends(get_container)) -> HealthResponse:
        """Return configuration and index status. No model is contacted."""
        return HealthResponse(
            status="ok",
            app=container.settings.app_name,
            answerer="deterministic-extractive",
            llm_model=None,
            embedding_model=container.embedder.model_name,
            index_size=container.store.size,
            index_dimension=container.store.dimension,
            similarity_threshold=container.retriever.similarity_threshold,
            knowledge_base_documents=container.knowledge_base_documents,
        )


def _register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(LLMError)
    def _handle_llm_error(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("local LLM unavailable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail=f"{LLM_UNAVAILABLE_RESPONSE} ({exc})",
                code="llm_unavailable",
            ).model_dump(mode="json"),
        )

    @app.exception_handler(DependencyMissingError)
    def _handle_missing_dependency(_: Request, exc: DependencyMissingError) -> JSONResponse:
        logger.error("missing dependency: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail=str(exc),
                code="dependency_missing",
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ConfigurationError)
    def _handle_configuration_error(_: Request, exc: ConfigurationError) -> JSONResponse:
        logger.error("configuration error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=ErrorResponse(detail=str(exc), code="invalid_request").model_dump(mode="json"),
        )

    @app.exception_handler(AhmedRAGError)
    def _handle_app_error(_: Request, exc: AhmedRAGError) -> JSONResponse:
        logger.exception("application error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An internal error occurred while processing the request.",
                code="internal_error",
            ).model_dump(mode="json"),
        )


app = create_app()


def run() -> None:  # pragma: no cover - convenience entry point
    """Start the development server (``ahmed-rag`` console script)."""
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
