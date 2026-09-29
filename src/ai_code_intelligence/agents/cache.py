from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, BinaryIO, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from ai_code_intelligence.agents.bedrock import (
    StructuredModelClient,
    StructuredRequest,
    StructuredResponse,
)
from ai_code_intelligence.domain.models import AgentInvocation

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_KEY_VERSION = 1  # Preserve every existing content-addressed GLM checkpoint.
_ENTRY_VERSION = 2
_LOCK_POLL_SECONDS = 0.05
_LOCK_TIMEOUT_SECONDS = 7_200
_PATH_LOCKS: dict[Path, RLock] = {}
_PATH_LOCKS_GUARD = Lock()


@dataclass(frozen=True, slots=True)
class AgentCheckpointStatistics:
    """In-process observability counters for durable specialist checkpoints."""

    hits: int
    misses: int
    writes: int
    invalid_entries: int


@dataclass(frozen=True, slots=True)
class _CacheRead:
    response: StructuredResponse[Any] | None = None
    invalid_error_type: str | None = None


class LocalStructuredResponseCache:
    """Durably checkpoints each successfully validated specialist response.

    Checkpoints are content-addressed by the complete model request and survive
    worker or container restarts when the configured directory is on persistent
    storage. A per-key process lock plus an OS advisory lock prevents duplicate
    calls and torn writes when workers converge on the same batch.
    """

    def __init__(self, delegate: StructuredModelClient, directory: str | Path, model_id: str) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        self._delegate = delegate
        self._directory = Path(directory).expanduser().resolve()
        self._model_id = model_id
        self._logger = logging.getLogger(__name__)
        self._statistics_lock = Lock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._invalid_entries = 0

    @property
    def statistics(self) -> AgentCheckpointStatistics:
        """Return a consistent snapshot of cache resume telemetry."""

        with self._statistics_lock:
            return AgentCheckpointStatistics(
                hits=self._hits,
                misses=self._misses,
                writes=self._writes,
                invalid_entries=self._invalid_entries,
            )

    def invoke(self, request: StructuredRequest[ResponseT]) -> StructuredResponse[ResponseT]:
        digest = self._digest(request)
        cache_path = self._path(request, digest)
        cached = self._read(cache_path, digest, request)
        if cached.response is not None:
            self._record_hit()
            self._log_resume(request, digest, waited_for_writer=False)
            return _typed_response(cached.response)

        # Recheck under both locks. This turns concurrent identical invocations
        # into one model call, including callers in separate worker processes.
        with _path_lock(cache_path), _checkpoint_file_lock(self._lock_path(request, digest)):
            cached = self._read(cache_path, digest, request)
            if cached.response is not None:
                self._record_hit()
                self._log_resume(request, digest, waited_for_writer=True)
                return _typed_response(cached.response)
            if cached.invalid_error_type is not None:
                self._record_invalid()
                self._logger.warning(
                    "Discarding invalid durable agent checkpoint",
                    extra={
                        "fields": {
                            **self._safe_fields(request, digest),
                            "error_type": cached.invalid_error_type,
                        }
                    },
                )
                cache_path.unlink(missing_ok=True)

            self._record_miss()
            self._logger.info(
                "Durable agent checkpoint miss; invoking model",
                extra={"fields": self._safe_fields(request, digest)},
            )
            # Exceptions and invalid delegate values deliberately escape before
            # _write, so failed responses can never become retry checkpoints.
            response = self._validate_delegate_response(request, self._delegate.invoke(request))
            self._write(cache_path, digest, response)
            self._record_write()
            self._logger.info(
                "Stored durable agent checkpoint",
                extra={"fields": self._safe_fields(request, digest)},
            )
            return response

    def _digest(self, request: StructuredRequest[Any]) -> str:
        material = json.dumps(
            {
                "version": _KEY_VERSION,
                "model_id": self._model_id,
                "role": request.role.value,
                "schema_name": request.schema_name,
                "schema_description": request.schema_description,
                "schema": request.response_model.model_json_schema(),
                "system_prompt": request.system_prompt,
                "user_prompt": request.user_prompt,
                "repository_id": request.repository_id,
                "batch_id": request.batch_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def _path(self, request: StructuredRequest[Any], digest: str) -> Path:
        return self._directory / request.role.value / f"{digest}.json"

    def _lock_path(self, request: StructuredRequest[Any], digest: str) -> Path:
        return self._directory / ".locks" / request.role.value / f"{digest}.lock"

    def _read(
        self,
        path: Path,
        digest: str,
        request: StructuredRequest[ResponseT],
    ) -> _CacheRead:
        if not path.is_file():
            return _CacheRead()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("checkpoint root is not an object")
            entry_version = payload.get("entry_version")
            if entry_version is not None:
                if entry_version != _ENTRY_VERSION:
                    raise ValueError("unsupported checkpoint entry version")
                if payload.get("request_digest") != digest:
                    raise ValueError("checkpoint request digest mismatch")
            elif "request_digest" in payload:
                raise ValueError("checkpoint entry version is missing")
            response = self._validated_response(
                request,
                payload["value"],
                payload["invocation"],
            )
            invocation = response.invocation.model_copy(
                update={
                    "invocation_id": f"cache:{uuid4()}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": 0,
                    "cache_hit": True,
                }
            )
            return _CacheRead(response=StructuredResponse(value=response.value, invocation=invocation))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return _CacheRead(invalid_error_type=type(error).__name__)

    def _validate_delegate_response(
        self,
        request: StructuredRequest[ResponseT],
        response: StructuredResponse[ResponseT],
    ) -> StructuredResponse[ResponseT]:
        try:
            value = response.value.model_dump(mode="python")
            invocation = response.invocation.model_dump(mode="python")
            return _typed_response(self._validated_response(request, value, invocation))
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"{request.role.value} delegate returned an invalid checkpoint response"
            ) from error

    def _validated_response(
        self,
        request: StructuredRequest[ResponseT],
        raw_value: object,
        raw_invocation: object,
    ) -> StructuredResponse[ResponseT]:
        value = request.response_model.model_validate(raw_value)
        invocation = AgentInvocation.model_validate(raw_invocation)
        if invocation.role is not request.role:
            raise ValueError("checkpoint invocation role does not match request")
        if invocation.model_id != self._model_id:
            raise ValueError("checkpoint invocation model does not match cache namespace")
        if invocation.repository_id != request.repository_id:
            raise ValueError("checkpoint repository does not match request")
        if invocation.batch_id != request.batch_id:
            raise ValueError("checkpoint batch does not match request")
        if invocation.cache_hit:
            raise ValueError("a resumed invocation cannot be stored as a new checkpoint")
        if (
            not invocation.invocation_id.strip()
            or invocation.input_tokens < 0
            or invocation.output_tokens < 0
            or invocation.total_tokens < 0
            or invocation.latency_ms < 0
        ):
            raise ValueError("checkpoint invocation telemetry is invalid")
        return StructuredResponse(value=value, invocation=invocation)

    def _write(
        self,
        path: Path,
        digest: str,
        response: StructuredResponse[Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
        payload = {
            "entry_version": _ENTRY_VERSION,
            "request_digest": digest,
            "value": response.value.model_dump(mode="json"),
            "invocation": response.invocation.model_dump(mode="json"),
        }
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except OSError as error:
            raise RuntimeError("unable to persist durable agent checkpoint") from error
        finally:
            temporary.unlink(missing_ok=True)

    def _log_resume(
        self,
        request: StructuredRequest[Any],
        digest: str,
        *,
        waited_for_writer: bool,
    ) -> None:
        self._logger.info(
            "Resuming from durable agent checkpoint",
            extra={
                "fields": {
                    **self._safe_fields(request, digest),
                    "waited_for_writer": waited_for_writer,
                }
            },
        )

    def _safe_fields(self, request: StructuredRequest[Any], digest: str) -> dict[str, object]:
        # Prompts, schemas, filesystem paths, URLs, and model output never enter logs.
        return {
            "checkpoint_id": digest[:16],
            "agent_role": request.role.value,
            "repository_id": request.repository_id,
            "batch_id": request.batch_id,
        }

    def _record_hit(self) -> None:
        with self._statistics_lock:
            self._hits += 1

    def _record_miss(self) -> None:
        with self._statistics_lock:
            self._misses += 1

    def _record_write(self) -> None:
        with self._statistics_lock:
            self._writes += 1

    def _record_invalid(self) -> None:
        with self._statistics_lock:
            self._invalid_entries += 1


def _path_lock(path: Path) -> RLock:
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(path, RLock())


@contextmanager
def _checkpoint_file_lock(path: Path) -> Iterator[None]:
    """Hold a cross-process advisory lock for one content-addressed request."""

    handle: BinaryIO | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        _prepare_lock_byte(handle)
        _acquire_file_lock(handle)
    except OSError as error:
        if handle is not None:
            handle.close()
        raise RuntimeError("unable to lock durable agent checkpoint") from error
    try:
        # Keep body exceptions intact: an OSError from the delegate is a model
        # failure, not a checkpoint-lock failure.
        yield
    finally:
        try:
            _release_file_lock(handle)
        finally:
            handle.close()


def _prepare_lock_byte(handle: BinaryIO) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)


def _acquire_file_lock(handle: BinaryIO) -> None:
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl: Any = importlib.import_module("fcntl")

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError("durable agent checkpoint lock timed out") from None
            time.sleep(_LOCK_POLL_SECONDS)


def _release_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl: Any = importlib.import_module("fcntl")

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _typed_response(response: StructuredResponse[Any]) -> StructuredResponse[ResponseT]:
    return StructuredResponse(value=response.value, invocation=response.invocation)
