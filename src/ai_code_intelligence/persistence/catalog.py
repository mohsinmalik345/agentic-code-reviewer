from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, RLock
from time import sleep
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_code_intelligence.domain.models import RepositoryDefinition
from ai_code_intelligence.domain.repositories import (
    IndexingJob,
    IndexingJobEvent,
    IndexingJobStatus,
    IndexingStage,
    JobEventLevel,
    RepositoryProvider,
    RepositoryRecord,
    RepositoryStatus,
)

_SCHEMA_VERSION = 1
_PATH_LOCKS: dict[Path, RLock] = {}
_PATH_LOCKS_GUARD = Lock()
_STAGE_ORDER = {
    IndexingStage.QUEUED: 0,
    IndexingStage.MATERIALIZING: 1,
    IndexingStage.SCANNING: 2,
    IndexingStage.PUBLISHING: 3,
    IndexingStage.COMPLETE: 4,
}


class RepositoryCatalog(Protocol):
    """Persistence port for durable repository registrations."""

    def seed_local_repositories(
        self,
        definitions: Iterable[RepositoryDefinition],
    ) -> tuple[RepositoryRecord, ...]: ...

    def upsert_repository(self, record: RepositoryRecord) -> RepositoryRecord: ...

    def upsert_repositories(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]: ...

    def get_repository(self, repository_id: str) -> RepositoryRecord | None: ...

    def list_repositories(self, *, include_disabled: bool = True) -> tuple[RepositoryRecord, ...]: ...

    def publish_index_metadata(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]: ...

    def migrate_local_sources_to_gitlab(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]: ...


class IndexingJobStore(Protocol):
    """Persistence port for a durable FIFO indexing-job queue."""

    def enqueue_job(self, job: IndexingJob) -> IndexingJob: ...

    def get_job(self, job_id: str) -> IndexingJob | None: ...

    def list_jobs(self, *, status: IndexingJobStatus | None = None) -> tuple[IndexingJob, ...]: ...

    def claim_next_job(self) -> IndexingJob | None: ...

    def request_job_cancellation(self, job_id: str) -> IndexingJob: ...

    def update_job(
        self,
        job_id: str,
        *,
        status: IndexingJobStatus,
        stage: IndexingStage,
        scan_run_id: str | None = None,
        error_code: str | None = None,
    ) -> IndexingJob: ...


class IndexingJobEventStore(Protocol):
    """Persistence port for sanitized, durable indexing-job progress events."""

    def append_job_event(self, event: IndexingJobEvent) -> IndexingJobEvent: ...

    def list_job_events(
        self,
        job_id: str,
        *,
        limit: int = 200,
    ) -> tuple[IndexingJobEvent, ...]: ...


class RepositoryStateStore(RepositoryCatalog, IndexingJobStore, IndexingJobEventStore, Protocol):
    """Combined catalog and job queue used by the application composition root."""

    def register_repositories_and_ensure_queued_job(
        self,
        records: Iterable[RepositoryRecord],
        *,
        requested_repository_id: str | None = None,
    ) -> tuple[tuple[RepositoryRecord, ...], IndexingJob, bool]: ...


class _CatalogState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    repositories: dict[str, RepositoryRecord] = Field(default_factory=dict)
    jobs: dict[str, IndexingJob] = Field(default_factory=dict)
    job_events: dict[str, tuple[IndexingJobEvent, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def keys_match_record_ids(self) -> _CatalogState:
        if any(key != record.id for key, record in self.repositories.items()):
            raise ValueError("repository catalog keys must match record IDs")
        if any(key != job.id for key, job in self.jobs.items()):
            raise ValueError("indexing job catalog keys must match job IDs")
        for job_id, events in self.job_events.items():
            if job_id not in self.jobs or any(event.job_id != job_id for event in events):
                raise ValueError("indexing job events must reference their catalog job")
        return self


class FileRepositoryCatalog(RepositoryCatalog, IndexingJobStore):
    """Atomic, thread-safe JSON repository catalog and FIFO indexing-job store."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._lock = _path_lock(self._path)
        with self._lock:
            if self._path.exists():
                self._read()
            else:
                self._write(_CatalogState())

    @property
    def path(self) -> Path:
        return self._path

    def seed_local_repositories(
        self,
        definitions: Iterable[RepositoryDefinition],
    ) -> tuple[RepositoryRecord, ...]:
        """Insert configured local repositories without resetting existing index metadata."""

        with self._lock:
            state = self._read()
            repositories = dict(state.repositories)
            changed = False
            seeded: list[RepositoryRecord] = []
            for definition in definitions:
                path = str(Path(definition.path).expanduser().resolve())
                existing = repositories.get(definition.id)
                if existing is None:
                    record = RepositoryRecord(
                        id=definition.id,
                        provider=RepositoryProvider.LOCAL,
                        name=definition.name,
                        description=definition.description,
                        local_path=path,
                        status=RepositoryStatus.READY,
                    )
                    repositories[record.id] = record
                    seeded.append(record)
                    changed = True
                    continue
                if existing.provider is not RepositoryProvider.LOCAL:
                    raise ValueError(
                        f"configured local repository {definition.id} conflicts with "
                        f"existing {existing.provider.value} registration"
                    )
                updates = {
                    "name": definition.name.strip(),
                    "description": definition.description.strip() if definition.description else None,
                    "local_path": path,
                }
                candidate = existing.model_copy(update=updates)
                candidate = RepositoryRecord.model_validate(candidate.model_dump(mode="python"))
                if candidate != existing:
                    candidate = candidate.model_copy(update={"updated_at": _now()})
                    repositories[candidate.id] = candidate
                    existing = candidate
                    changed = True
                seeded.append(existing)
            if changed:
                self._write(state.model_copy(update={"repositories": repositories}))
            return tuple(seeded)

    def upsert_repository(self, record: RepositoryRecord) -> RepositoryRecord:
        """Insert or update a record while preserving its immutable source identity."""

        return self.upsert_repositories((record,))[0]

    def upsert_repositories(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]:
        """Atomically insert or update a nonempty set of repository registrations."""

        requested = tuple(
            RepositoryRecord.model_validate(record.model_dump(mode="python")) for record in records
        )
        if not requested:
            raise ValueError("repository registration requires at least one record")
        if len({record.id for record in requested}) != len(requested):
            raise ValueError("repository registration requires unique repository IDs")
        canonical_urls = tuple(
            record.canonical_url for record in requested if record.canonical_url is not None
        )
        if len(set(canonical_urls)) != len(canonical_urls):
            raise ValueError("repository registration requires unique GitLab URLs")
        with self._lock:
            state = self._read()
            repositories = dict(state.repositories)
            stored: list[RepositoryRecord] = []
            changed = False
            for record in requested:
                existing = repositories.get(record.id)
                duplicate = next(
                    (
                        item
                        for item in repositories.values()
                        if item.id != record.id
                        and record.canonical_url is not None
                        and item.canonical_url == record.canonical_url
                    ),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(
                        f"GitLab repository URL is already registered as {duplicate.id}"
                    )
                if existing is None:
                    repositories[record.id] = record
                    stored.append(record)
                    changed = True
                    continue
                if existing.provider is not record.provider:
                    raise ValueError(f"repository {record.id} provider cannot be changed")
                if existing.canonical_url != record.canonical_url:
                    raise ValueError(f"repository {record.id} canonical_url cannot be changed")

                candidate = record.model_copy(
                    update={
                        "created_at": existing.created_at,
                        "updated_at": existing.updated_at,
                    }
                )
                if _repository_payload(candidate) == _repository_payload(existing):
                    stored.append(existing)
                    continue
                candidate = candidate.model_copy(update={"updated_at": _now()})
                candidate = RepositoryRecord.model_validate(candidate.model_dump(mode="python"))
                repositories[candidate.id] = candidate
                stored.append(candidate)
                changed = True
            if changed:
                self._write(state.model_copy(update={"repositories": repositories}))
            return tuple(stored)

    def register_repositories_and_ensure_queued_job(
        self,
        records: Iterable[RepositoryRecord],
        *,
        requested_repository_id: str | None = None,
    ) -> tuple[tuple[RepositoryRecord, ...], IndexingJob, bool]:
        """Register operator fields and ensure one queued job in one durable file write."""

        requested = _validated_repository_batch(records)
        if requested_repository_id is not None and requested_repository_id not in {
            record.id for record in requested
        }:
            raise ValueError("requested repository must belong to the registration batch")
        with self._lock:
            state = self._read()
            repositories = dict(state.repositories)
            stored: list[RepositoryRecord] = []
            changed = False
            for record in requested:
                existing = repositories.get(record.id)
                duplicate = next(
                    (
                        item
                        for item in repositories.values()
                        if item.id != record.id
                        and record.canonical_url is not None
                        and item.canonical_url == record.canonical_url
                    ),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(
                        f"GitLab repository URL is already registered as {duplicate.id}"
                    )
                if existing is None:
                    repositories[record.id] = record
                    stored.append(record)
                    changed = True
                    continue
                if existing.provider is not record.provider:
                    raise ValueError(f"repository {record.id} provider cannot be changed")
                if existing.canonical_url != record.canonical_url:
                    raise ValueError(f"repository {record.id} canonical_url cannot be changed")
                update = {
                    "name": record.name,
                    "description": record.description,
                    "canonical_ref": record.canonical_ref,
                }
                candidate = existing.model_copy(update=update)
                if _repository_payload(candidate) == _repository_payload(existing):
                    stored.append(existing)
                    continue
                candidate = candidate.model_copy(update={"updated_at": _now()})
                candidate = RepositoryRecord.model_validate(candidate.model_dump(mode="python"))
                repositories[candidate.id] = candidate
                stored.append(candidate)
                changed = True

            queued = sorted(
                (job for job in state.jobs.values() if job.status is IndexingJobStatus.QUEUED),
                key=lambda job: (job.created_at, job.id),
            )
            job_created = not queued
            job = queued[0] if queued else IndexingJob.create(requested_repository_id)
            jobs = state.jobs if queued else {**state.jobs, job.id: job}
            if changed or job_created:
                self._write(
                    state.model_copy(
                        update={"repositories": repositories, "jobs": jobs}
                    )
                )
            return tuple(stored), job, job_created

    def get_repository(self, repository_id: str) -> RepositoryRecord | None:
        with self._lock:
            return self._read().repositories.get(repository_id)

    def list_repositories(self, *, include_disabled: bool = True) -> tuple[RepositoryRecord, ...]:
        with self._lock:
            values = self._read().repositories.values()
            return tuple(
                sorted(
                    (
                        record
                        for record in values
                        if include_disabled or record.status is not RepositoryStatus.DISABLED
                    ),
                    key=lambda record: record.id,
                )
            )

    def publish_index_metadata(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]:
        """Atomically publish index metadata for an already registered portfolio."""

        requested = tuple(records)
        if not requested or len({record.id for record in requested}) != len(requested):
            raise ValueError("index metadata requires nonempty, unique repository records")
        with self._lock:
            state = self._read()
            repositories = dict(state.repositories)
            published: list[RepositoryRecord] = []
            changed = False
            for requested_record in requested:
                record = RepositoryRecord.model_validate(requested_record.model_dump(mode="python"))
                existing = repositories.get(record.id)
                if existing is None:
                    raise ValueError(f"unknown repository id: {record.id}")
                if existing.provider is not record.provider or existing.canonical_url != record.canonical_url:
                    raise ValueError("repository source identity cannot be changed")
                candidate = existing.model_copy(update=_index_metadata_payload(record))
                candidate = RepositoryRecord.model_validate(candidate.model_dump(mode="python"))
                if _index_metadata_payload(candidate) == _index_metadata_payload(existing):
                    published.append(existing)
                    continue
                candidate = candidate.model_copy(update={"updated_at": _now()})
                candidate = RepositoryRecord.model_validate(candidate.model_dump(mode="python"))
                repositories[candidate.id] = candidate
                published.append(candidate)
                changed = True
            if changed:
                self._write(state.model_copy(update={"repositories": repositories}))
            return tuple(published)

    def migrate_local_sources_to_gitlab(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]:
        """Atomically convert local registrations into validated GitLab sources."""

        requested = tuple(
            RepositoryRecord.model_validate(record.model_dump(mode="python")) for record in records
        )
        if not requested or len({record.id for record in requested}) != len(requested):
            raise ValueError("source migration requires nonempty, unique repository records")
        if len({record.canonical_url for record in requested}) != len(requested):
            raise ValueError("source migration requires unique GitLab URLs")
        with self._lock:
            state = self._read()
            if any(
                job.status in {IndexingJobStatus.QUEUED, IndexingJobStatus.RUNNING}
                for job in state.jobs.values()
            ):
                raise ValueError("repository sources cannot be migrated while an indexing job is active")
            repositories = dict(state.repositories)
            migrated_ids = {record.id for record in requested}
            occupied_urls = {
                record.canonical_url
                for record in repositories.values()
                if record.id not in migrated_ids and record.canonical_url is not None
            }
            if occupied_urls.intersection(record.canonical_url for record in requested):
                raise ValueError("GitLab repository URL is already registered")

            migrated: list[RepositoryRecord] = []
            now = _now()
            for record in requested:
                existing = repositories.get(record.id)
                if existing is None:
                    raise ValueError(f"unknown repository id: {record.id}")
                if existing.provider is not RepositoryProvider.LOCAL:
                    raise ValueError(f"repository {record.id} is not a local registration")
                if record.provider is not RepositoryProvider.GITLAB or record.canonical_url is None:
                    raise ValueError("source migration target must be a GitLab registration")
                candidate = existing.model_copy(
                    update={
                        "provider": RepositoryProvider.GITLAB,
                        "local_path": record.local_path,
                        "canonical_url": record.canonical_url,
                        "canonical_ref": record.canonical_ref,
                        "exact_commit": record.exact_commit,
                        "status": RepositoryStatus.READY,
                        "updated_at": now,
                    }
                )
                candidate = RepositoryRecord.model_validate(candidate.model_dump(mode="python"))
                repositories[candidate.id] = candidate
                migrated.append(candidate)
            self._write(state.model_copy(update={"repositories": repositories}))
            return tuple(migrated)

    def enqueue_job(self, job: IndexingJob) -> IndexingJob:
        """Append a caller-created queued job, rejecting duplicate IDs."""

        job = IndexingJob.model_validate(job.model_dump(mode="python"))
        if job.status is not IndexingJobStatus.QUEUED:
            raise ValueError("only queued jobs can be enqueued")
        with self._lock:
            state = self._read()
            if job.id in state.jobs:
                raise ValueError(f"indexing job already exists: {job.id}")
            if (
                job.requested_repository_id is not None
                and job.requested_repository_id not in state.repositories
            ):
                raise ValueError(f"unknown repository id: {job.requested_repository_id}")
            jobs = {**state.jobs, job.id: job}
            self._write(state.model_copy(update={"jobs": jobs}))
            return job

    def get_job(self, job_id: str) -> IndexingJob | None:
        with self._lock:
            return self._read().jobs.get(job_id)

    def list_jobs(self, *, status: IndexingJobStatus | None = None) -> tuple[IndexingJob, ...]:
        with self._lock:
            values = self._read().jobs.values()
            return tuple(
                sorted(
                    (job for job in values if status is None or job.status is status),
                    key=lambda job: (job.created_at, job.id),
                )
            )

    def claim_next_job(self) -> IndexingJob | None:
        """Atomically transition the oldest queued job into materialization."""

        with self._lock:
            state = self._read()
            queued = sorted(
                (job for job in state.jobs.values() if job.status is IndexingJobStatus.QUEUED),
                key=lambda job: (job.created_at, job.id),
            )
            if not queued:
                return None
            job = queued[0]
            now = _now()
            claimed = job.model_copy(
                update={
                    "status": IndexingJobStatus.RUNNING,
                    "stage": IndexingStage.MATERIALIZING,
                    "started_at": now,
                    "updated_at": now,
                }
            )
            claimed = IndexingJob.model_validate(claimed.model_dump(mode="python"))
            jobs = {**state.jobs, claimed.id: claimed}
            self._write(state.model_copy(update={"jobs": jobs}))
            return claimed

    def update_job(
        self,
        job_id: str,
        *,
        status: IndexingJobStatus,
        stage: IndexingStage,
        scan_run_id: str | None = None,
        error_code: str | None = None,
    ) -> IndexingJob:
        """Advance a claimed job through a monotonic, terminal-safe state machine."""

        with self._lock:
            state = self._read()
            existing = state.jobs.get(job_id)
            if existing is None:
                raise KeyError(job_id)
            _validate_job_transition(existing, status, stage)
            now = _now()
            completed_at = (
                now
                if status
                in {
                    IndexingJobStatus.SUCCEEDED,
                    IndexingJobStatus.FAILED,
                    IndexingJobStatus.CANCELLED,
                }
                else None
            )
            updated = existing.model_copy(
                update={
                    "status": status,
                    "stage": stage,
                    "scan_run_id": scan_run_id or existing.scan_run_id,
                    "error_code": error_code,
                    "updated_at": now,
                    "completed_at": completed_at,
                }
            )
            updated = IndexingJob.model_validate(updated.model_dump(mode="python"))
            jobs = {**state.jobs, updated.id: updated}
            self._write(state.model_copy(update={"jobs": jobs}))
            return updated

    def request_job_cancellation(self, job_id: str) -> IndexingJob:
        """Atomically cancel queued work or flag active work for cooperative cancellation."""

        with self._lock:
            state = self._read()
            existing = state.jobs.get(job_id)
            if existing is None:
                raise KeyError(job_id)
            if existing.status is IndexingJobStatus.CANCELLED:
                return existing
            if existing.status in {IndexingJobStatus.SUCCEEDED, IndexingJobStatus.FAILED}:
                raise ValueError("completed indexing jobs cannot be cancelled")
            if existing.stage is IndexingStage.PUBLISHING:
                raise ValueError("an indexing job cannot be cancelled after publication starts")
            if existing.cancellation_requested_at is not None:
                return existing
            now = _now()
            changes: dict[str, object] = {
                "cancellation_requested_at": now,
                "updated_at": now,
            }
            if existing.status is IndexingJobStatus.QUEUED:
                changes.update(
                    {
                        "status": IndexingJobStatus.CANCELLED,
                        "completed_at": now,
                    }
                )
            updated = existing.model_copy(update=changes)
            updated = IndexingJob.model_validate(updated.model_dump(mode="python"))
            jobs = {**state.jobs, updated.id: updated}
            self._write(state.model_copy(update={"jobs": jobs}))
            return updated

    def append_job_event(self, event: IndexingJobEvent) -> IndexingJobEvent:
        """Append one validated event without modifying the job state."""

        event = IndexingJobEvent.model_validate(event.model_dump(mode="python"))
        with self._lock:
            state = self._read()
            if event.job_id not in state.jobs:
                raise KeyError(event.job_id)
            existing = state.job_events.get(event.job_id, ())
            if any(item.id == event.id for item in existing):
                raise ValueError(f"indexing job event already exists: {event.id}")
            job_events = {
                **state.job_events,
                event.job_id: (*existing, event),
            }
            self._write(state.model_copy(update={"job_events": job_events}))
            return event

    def list_job_events(
        self,
        job_id: str,
        *,
        limit: int = 200,
    ) -> tuple[IndexingJobEvent, ...]:
        """Return the newest bounded event window in chronological order."""

        if limit < 1 or limit > 1_000:
            raise ValueError("job event limit must be between 1 and 1000")
        with self._lock:
            events = self._read().job_events.get(job_id, ())
            return tuple(events[-limit:])

    def _read(self) -> _CatalogState:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("repository catalog root must be a JSON object")
            if raw.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported repository catalog schema version: {raw.get('schema_version')!r}"
                )
            return _CatalogState.model_validate(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"repository catalog is not valid JSON: {self._path}") from error
        except OSError as error:
            raise RuntimeError(f"unable to read repository catalog: {self._path}") from error

    def _write(self, state: _CatalogState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid4()}.tmp")
        payload = json.dumps(
            state.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_retry(temporary, self._path)
        except OSError as error:
            raise RuntimeError(f"unable to write repository catalog: {self._path}") from error
        finally:
            temporary.unlink(missing_ok=True)

    def recover_interrupted_jobs(self) -> None:
        """Fail jobs left running by a prior exclusive local worker process."""

        with self._lock:
            state = self._read()
            self._recover_interrupted_jobs(state)

    def _recover_interrupted_jobs(self, state: _CatalogState) -> None:
        running = [job for job in state.jobs.values() if job.status is IndexingJobStatus.RUNNING]
        if not running:
            return
        now = _now()
        jobs = dict(state.jobs)
        job_events = dict(state.job_events)
        for job in running:
            cancellation_requested = job.cancellation_requested_at is not None
            recovered = job.model_copy(
                update={
                    "status": (
                        IndexingJobStatus.CANCELLED
                        if cancellation_requested
                        else IndexingJobStatus.FAILED
                    ),
                    "error_code": None if cancellation_requested else "WORKER_RESTARTED",
                    "updated_at": now,
                    "completed_at": now,
                }
            )
            jobs[job.id] = IndexingJob.model_validate(recovered.model_dump(mode="python"))
            event = IndexingJobEvent(
                id=f"event:{uuid4()}",
                job_id=job.id,
                level=JobEventLevel.WARNING if cancellation_requested else JobEventLevel.ERROR,
                stage=job.stage,
                code="JOB_CANCELLED" if cancellation_requested else "WORKER_RESTARTED",
                message=(
                    "The worker stopped after cancellation was requested."
                    if cancellation_requested
                    else "The worker stopped before this indexing job completed."
                ),
                error_type=None if cancellation_requested else "WorkerRestart",
                created_at=now,
            )
            job_events[job.id] = (*job_events.get(job.id, ()), event)
        self._write(state.model_copy(update={"jobs": jobs, "job_events": job_events}))


def _path_lock(path: Path) -> RLock:
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(path, RLock())


def _replace_with_retry(source: Path, target: Path) -> None:
    """Handle short-lived Windows/OneDrive destination locks without losing atomicity."""

    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            sleep(0.02 * (2**attempt))


def _now() -> datetime:
    return datetime.now(UTC)


def _repository_payload(record: RepositoryRecord) -> dict[str, object]:
    return record.model_dump(mode="python", exclude={"created_at", "updated_at"})


def _validated_repository_batch(
    records: Iterable[RepositoryRecord],
) -> tuple[RepositoryRecord, ...]:
    requested = tuple(
        RepositoryRecord.model_validate(record.model_dump(mode="python")) for record in records
    )
    if not requested:
        raise ValueError("repository registration requires at least one record")
    if len({record.id for record in requested}) != len(requested):
        raise ValueError("repository registration requires unique repository IDs")
    canonical_urls = tuple(
        record.canonical_url for record in requested if record.canonical_url is not None
    )
    if len(set(canonical_urls)) != len(canonical_urls):
        raise ValueError("repository registration requires unique GitLab URLs")
    return requested


def _index_metadata_payload(record: RepositoryRecord) -> dict[str, object]:
    return {
        "status": record.status,
        "knowledge_base_uri": record.knowledge_base_uri,
        "graph_node_count": record.graph_node_count,
        "graph_relationship_count": record.graph_relationship_count,
        "last_scan_run_id": record.last_scan_run_id,
        "indexed_at": record.indexed_at,
    }


def _validate_job_transition(
    existing: IndexingJob,
    status: IndexingJobStatus,
    stage: IndexingStage,
) -> None:
    if existing.status in {
        IndexingJobStatus.SUCCEEDED,
        IndexingJobStatus.FAILED,
        IndexingJobStatus.CANCELLED,
    }:
        raise ValueError("terminal indexing jobs cannot be updated")
    if existing.status is IndexingJobStatus.QUEUED:
        raise ValueError("an indexing job must be claimed before it can be updated")
    if status not in {
        IndexingJobStatus.RUNNING,
        IndexingJobStatus.SUCCEEDED,
        IndexingJobStatus.FAILED,
        IndexingJobStatus.CANCELLED,
    }:
        raise ValueError("a running indexing job cannot return to queued")
    if _STAGE_ORDER[stage] < _STAGE_ORDER[existing.stage]:
        raise ValueError("indexing stage cannot move backwards")
    if status is IndexingJobStatus.RUNNING and stage not in {
        IndexingStage.MATERIALIZING,
        IndexingStage.SCANNING,
        IndexingStage.PUBLISHING,
    }:
        raise ValueError("running jobs require an active indexing stage")
    if status is IndexingJobStatus.SUCCEEDED and stage is not IndexingStage.COMPLETE:
        raise ValueError("succeeded jobs must use the complete stage")
    if status is IndexingJobStatus.FAILED and stage is IndexingStage.COMPLETE:
        raise ValueError("failed jobs must retain the stage that failed")
    if status is IndexingJobStatus.CANCELLED and stage not in {
        IndexingStage.MATERIALIZING,
        IndexingStage.SCANNING,
    }:
        raise ValueError("cancelled active jobs must stop before publication")
    if status is IndexingJobStatus.CANCELLED and existing.cancellation_requested_at is None:
        raise ValueError("cancellation must be requested before a running job is cancelled")
    if (
        existing.cancellation_requested_at is not None
        and status not in {IndexingJobStatus.RUNNING, IndexingJobStatus.CANCELLED}
    ):
        raise ValueError("job cancellation has been requested")
    if stage is IndexingStage.PUBLISHING and existing.cancellation_requested_at is not None:
        raise ValueError("job cancellation has been requested")
