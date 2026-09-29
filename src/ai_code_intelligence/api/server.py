from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from graphql import (
    ExecutionResult,
    NoSchemaIntrospectionCustomRule,
    execute,
    parse,
    specified_rules,
    validate,
)

from ai_code_intelligence.api.auth import (
    SESSION_COOKIE_NAME,
    SESSION_LIFETIME_SECONDS,
    BrowserSession,
    BrowserSessionManager,
    LoginRateLimiter,
    csrf_matches,
)
from ai_code_intelligence.api.schema import create_schema
from ai_code_intelligence.api.service import PlatformService, UserInputError
from ai_code_intelligence.config import ApplicationConfig
from ai_code_intelligence.domain.knowledge import KnowledgeDocumentView

_LOGGER = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 1_048_576
_MAX_LOGIN_REQUEST_BYTES = 8_192
_STATIC_DIRECTORY = Path(__file__).with_name("static")
_REPOSITORY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "base-uri 'none'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        "connect-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
    )
)


def create_app(service: PlatformService, config: ApplicationConfig) -> FastAPI:
    """Create the authenticated web application and bounded GraphQL HTTP service."""

    schema = create_schema(service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        await asyncio.to_thread(service.startup)
        worker_stop = Event()
        worker_thread: Thread | None = None
        if config.repository_registry.driver == "local":
            worker_thread = Thread(
                target=service.run_worker,
                args=(worker_stop,),
                name="code-intelligence-indexing-worker",
                daemon=True,
            )
            worker_thread.start()
        try:
            yield
        finally:
            worker_stop.set()
            if worker_thread is not None:
                worker_thread.join(timeout=5)
            await asyncio.to_thread(service.shutdown)

    app = FastAPI(
        title="AI Code Intelligence",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    if not config.graphql.api_key_env:
        raise RuntimeError("GraphQL API key environment variable name is not configured")
    configured_key = os.getenv(config.graphql.api_key_env)
    if not configured_key:
        raise RuntimeError(f"GraphQL API key environment variable {config.graphql.api_key_env} is not set")
    sessions = BrowserSessionManager(configured_key)
    login_limiter = LoginRateLimiter()

    @app.middleware("http")
    async def security_boundary(request: Request, call_next: Any) -> Response:
        path = request.url.path
        browser_session: BrowserSession | None = None
        if _protected_path(path, config.graphql.path):
            supplied_api_key = request.headers.get("x-api-key")
            if supplied_api_key is not None:
                if not sessions.valid_api_key(supplied_api_key):
                    response = _authentication_error(path, status_code=401)
                    return _secure_response(request, response)
            else:
                browser_session = sessions.verify(request.cookies.get(SESSION_COOKIE_NAME))
                if browser_session is None:
                    response = _authentication_error(path, status_code=401)
                    return _secure_response(request, response)
                if (
                    path == config.graphql.path
                    and request.method == "POST"
                    and not csrf_matches(
                        browser_session,
                        request.headers.get("x-csrf-token", ""),
                    )
                ):
                    response = _authentication_error(path, status_code=403)
                    return _secure_response(request, response)
        request.state.browser_session = browser_session
        response = await call_next(request)
        return _secure_response(request, response)

    @app.get("/", include_in_schema=False)
    async def web_application() -> Response:
        return _static_response("index.html", "text/html; charset=utf-8")

    @app.get("/static/app.js", include_in_schema=False)
    async def web_javascript() -> Response:
        return _static_response("app.js", "text/javascript; charset=utf-8")

    @app.get("/static/styles.css", include_in_schema=False)
    async def web_styles() -> Response:
        return _static_response("styles.css", "text/css; charset=utf-8")

    @app.post("/auth/login")
    async def login(request: Request) -> JSONResponse:
        client_id = request.client.host if request.client is not None else "unknown"
        if not login_limiter.allow(client_id):
            return JSONResponse(
                {"error": "too many login attempts"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        supplied_api_key = await _login_api_key(request)
        if supplied_api_key is None or not sessions.valid_api_key(supplied_api_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        token, session = sessions.issue()
        response = JSONResponse(_session_payload(session))
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            max_age=SESSION_LIFETIME_SECONDS,
            expires=session.expires_at,
            path="/",
            secure=request.url.scheme == "https",
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/auth/session")
    async def restore_session(request: Request) -> JSONResponse:
        session = sessions.verify(request.cookies.get(SESSION_COOKIE_NAME))
        if session is None:
            return JSONResponse({"authenticated": False}, status_code=401)
        return JSONResponse(_session_payload(session))

    @app.post("/auth/logout")
    async def logout(request: Request) -> JSONResponse:
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(
            SESSION_COOKIE_NAME,
            path="/",
            secure=request.url.scheme == "https",
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return service.health()

    @app.get("/api/knowledge/central")
    async def central_knowledge() -> JSONResponse:
        return await _knowledge_response(service.central_knowledge_document)

    @app.get("/api/knowledge/repositories/{repository_id}")
    async def repository_knowledge(repository_id: str) -> JSONResponse:
        if not _valid_repository_id(repository_id):
            return JSONResponse({"error": "knowledge document not found"}, status_code=404)
        return await _knowledge_response(
            service.repository_knowledge_document,
            repository_id,
        )

    @app.get("/api/knowledge/human/central")
    async def human_central_knowledge() -> JSONResponse:
        return await _knowledge_response(
            service.central_knowledge_document,
            human_readable=True,
        )

    @app.get("/api/knowledge/human/repositories/{repository_id}")
    async def human_repository_knowledge(repository_id: str) -> JSONResponse:
        if not _valid_repository_id(repository_id):
            return JSONResponse({"error": "knowledge document not found"}, status_code=404)
        return await _knowledge_response(
            service.repository_knowledge_document,
            repository_id,
            human_readable=True,
        )

    @app.post(config.graphql.path)
    async def graphql_endpoint(request: Request) -> JSONResponse:
        content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        if content_type != "application/json":
            return JSONResponse(
                {"errors": [{"message": "content-type must be application/json"}]},
                status_code=415,
            )
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse({"errors": [{"message": "invalid content-length"}]}, status_code=400)
            if declared_length < 0:
                return JSONResponse({"errors": [{"message": "invalid content-length"}]}, status_code=400)
            if declared_length > _MAX_REQUEST_BYTES:
                return JSONResponse({"errors": [{"message": "request too large"}]}, status_code=413)
        try:
            request_body = await request.body()
            if len(request_body) > _MAX_REQUEST_BYTES:
                return JSONResponse({"errors": [{"message": "request too large"}]}, status_code=413)
            body = json.loads(request_body)
            query = body["query"]
            variables = body.get("variables")
            operation_name = body.get("operationName")
            if not isinstance(query, str):
                raise TypeError("query must be a string")
            document = parse(query)
            rules = (
                specified_rules
                if config.graphql.allow_introspection
                else (*specified_rules, NoSchemaIntrospectionCustomRule)
            )
            errors = validate(schema, document, rules)
            if errors:
                return JSONResponse(
                    {"errors": [{"message": error.message} for error in errors]},
                    status_code=400,
                )
            result = execute(
                schema,
                document,
                variable_values=variables,
                operation_name=operation_name,
            )
            if inspect.isawaitable(result):
                result = await result
            assert isinstance(result, ExecutionResult)
            payload: dict[str, Any] = {"data": result.data}
            if result.errors:
                response_errors: list[dict[str, str]] = []
                for error in result.errors:
                    if error.original_error is None:
                        response_errors.append({"message": error.message})
                        continue
                    if isinstance(error.original_error, UserInputError):
                        response_errors.append({"message": str(error.original_error)})
                        continue
                    _LOGGER.error(
                        "GraphQL resolver failed",
                        extra={
                            "fields": {
                                "error_type": type(error.original_error).__name__,
                                "path": error.path,
                            }
                        },
                    )
                    response_errors.append({"message": "internal server error"})
                payload["errors"] = response_errors
            return JSONResponse(payload, status_code=200 if not result.errors else 400)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return JSONResponse({"errors": [{"message": str(error)}]}, status_code=400)

    return app


async def _login_api_key(request: Request) -> str | None:
    content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if content_type != "application/json":
        return None
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            return None
        if declared_length < 0 or declared_length > _MAX_LOGIN_REQUEST_BYTES:
            return None
    body = await request.body()
    if len(body) > _MAX_LOGIN_REQUEST_BYTES:
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"apiKey"}:
        return None
    api_key = payload.get("apiKey")
    if not isinstance(api_key, str) or not api_key or len(api_key.encode("utf-8")) > 4_096:
        return None
    return api_key


def _protected_path(path: str, graphql_path: str) -> bool:
    return path == graphql_path or path.startswith("/api/knowledge/")


def _authentication_error(path: str, *, status_code: int) -> JSONResponse:
    message = "unauthorized" if status_code == 401 else "forbidden"
    if path.endswith("/graphql"):
        return JSONResponse({"errors": [{"message": message}]}, status_code=status_code)
    return JSONResponse({"error": message}, status_code=status_code)


def _secure_response(request: Request, response: Response) -> Response:
    response.headers["Content-Security-Policy"] = _CONTENT_SECURITY_POLICY
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    path = request.url.path
    if (
        path == "/health"
        or path.startswith("/auth/")
        or path.startswith("/api/")
        or path.endswith("/graphql")
    ):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


def _static_response(filename: str, media_type: str) -> Response:
    path = _STATIC_DIRECTORY / filename
    if not path.is_file():
        return JSONResponse({"error": "web application is unavailable"}, status_code=503)
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


def _session_payload(session: BrowserSession) -> dict[str, object]:
    return {
        "authenticated": True,
        "csrfToken": session.csrf_token,
        "expiresAt": session.expires_at.isoformat(),
    }


async def _knowledge_response(
    loader: Callable[..., KnowledgeDocumentView],
    *arguments: str,
    **keyword_arguments: bool,
) -> JSONResponse:
    try:
        document = await asyncio.to_thread(loader, *arguments, **keyword_arguments)
        if not isinstance(document, KnowledgeDocumentView):
            raise TypeError("unexpected knowledge document result")
        return JSONResponse(_knowledge_payload(document))
    except (FileNotFoundError, KeyError, UserInputError):
        return JSONResponse({"error": "knowledge document not found"}, status_code=404)
    except Exception as error:
        _LOGGER.error(
            "Knowledge document request failed",
            extra={"fields": {"error_type": type(error).__name__}},
        )
        return JSONResponse({"error": "knowledge document is unavailable"}, status_code=503)


def _knowledge_payload(document: KnowledgeDocumentView) -> dict[str, object]:
    generated_at: datetime = document.generated_at
    return {
        "kind": document.kind,
        "title": document.title,
        "repositoryId": document.repository_id,
        "sourceUri": document.source_uri,
        "markdown": document.markdown,
        "contentSha256": document.content_sha256,
        "sizeBytes": document.size_bytes,
        "scanRunId": document.scan_run_id,
        "generatedAt": generated_at.isoformat(),
    }


def _valid_repository_id(value: str) -> bool:
    return _REPOSITORY_ID.fullmatch(value) is not None
