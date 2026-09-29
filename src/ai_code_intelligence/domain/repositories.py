from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from ai_code_intelligence.domain.models import FrozenModel, RepositoryDefinition

_REPOSITORY_ID = r"^[a-z0-9][a-z0-9-]{0,62}$"
_JOB_ID = r"^[A-Za-z0-9][A-Za-z0-9:._-]*$"
_ERROR_CODE = r"^[A-Z][A-Z0-9_]{0,127}$"
_EVENT_CODE = r"^[A-Z][A-Z0-9_]{0,127}$"
_ERROR_TYPE = r"^[A-Za-z_][A-Za-z0-9_.]{0,255}$"
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SCP_GIT_URL = re.compile(r"^git@(?P<host>[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?):(?P<path>[^\s]+)$")


def _now() -> datetime:
    return datetime.now(UTC)


class RepositoryProvider(StrEnum):
    """Supported origins for repositories registered with the platform."""

    LOCAL = "local"
    GITLAB = "gitlab"


class RepositoryStatus(StrEnum):
    """Lifecycle state for a registered repository."""

    REGISTERED = "registered"
    READY = "ready"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    DISABLED = "disabled"


class IndexingJobStatus(StrEnum):
    """Durable execution state for a whole-platform indexing job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IndexingStage(StrEnum):
    """Observable stage within an indexing job."""

    QUEUED = "queued"
    MATERIALIZING = "materializing"
    SCANNING = "scanning"
    PUBLISHING = "publishing"
    COMPLETE = "complete"


class JobEventLevel(StrEnum):
    """Severity attached to a durable indexing-job event."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RepositoryRecord(FrozenModel):
    """Durable repository registration and latest successful indexing metadata.

    Authentication material is intentionally absent. ``canonical_url`` accepts only
    credential-free HTTPS or SSH Git URLs for GitLab records.
    """

    id: str = Field(pattern=_REPOSITORY_ID)
    provider: RepositoryProvider
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4_000)
    local_path: str = Field(min_length=1)
    canonical_url: str | None = None
    canonical_ref: str | None = None
    exact_commit: str | None = None
    status: RepositoryStatus = RepositoryStatus.REGISTERED
    knowledge_base_uri: str | None = None
    graph_node_count: int = Field(default=0, ge=0)
    graph_relationship_count: int = Field(default=0, ge=0)
    last_scan_run_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    indexed_at: datetime | None = None

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        """Normalize the display name while rejecting whitespace-only values."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("name cannot be empty")
        if any(ord(character) < 32 for character in normalized):
            raise ValueError("name cannot contain control characters")
        return normalized

    @field_validator("description", "knowledge_base_uri", "last_scan_run_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional display and artifact metadata."""

        if value is None:
            return None
        normalized = value.strip()
        if any(ord(character) < 32 for character in normalized):
            raise ValueError("optional text cannot contain control characters")
        return normalized or None

    @field_validator("local_path")
    @classmethod
    def absolute_local_path(cls, value: str) -> str:
        """Require a concrete internal path before a repository enters the catalog."""

        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("local_path must be absolute")
        return str(path.resolve())

    @field_validator("canonical_ref")
    @classmethod
    def safe_canonical_ref(cls, value: str | None) -> str | None:
        """Reject option-like or ambiguous Git revisions."""

        if value is None:
            return None
        ref = value.strip()
        if not _REF.fullmatch(ref) or ".." in ref or "//" in ref or ref.endswith(("/", ".", ".lock")):
            raise ValueError("canonical_ref is not a safe Git revision")
        return ref

    @field_validator("exact_commit")
    @classmethod
    def canonical_commit(cls, value: str | None) -> str | None:
        """Store only full Git SHA-1 or SHA-256 object identifiers."""

        if value is None:
            return None
        commit = value.strip().lower()
        if not _COMMIT.fullmatch(commit):
            raise ValueError("exact_commit must be a full Git SHA-1 or SHA-256")
        return commit

    @field_validator("created_at", "updated_at", "indexed_at")
    @classmethod
    def timezone_aware(cls, value: datetime | None) -> datetime | None:
        """Normalize persisted timestamps to UTC."""

        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def valid_source_and_lifecycle(self) -> RepositoryRecord:
        if self.provider is RepositoryProvider.GITLAB:
            if self.canonical_url is None:
                raise ValueError("GitLab repositories require canonical_url")
            _validate_credential_free_git_url(self.canonical_url)
        elif self.canonical_url is not None:
            raise ValueError("local repositories cannot have canonical_url")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.indexed_at is not None and self.indexed_at < self.created_at:
            raise ValueError("indexed_at cannot precede created_at")
        if self.status is RepositoryStatus.INDEXED and self.indexed_at is None:
            raise ValueError("indexed repositories require indexed_at")
        return self

    def to_repository_definition(self) -> RepositoryDefinition:
        """Convert a materialized registration to the scanner's local input model."""

        return RepositoryDefinition(
            id=self.id,
            name=self.name,
            path=self.local_path,
            description=self.description,
        )


class IndexingJob(FrozenModel):
    """Durable request to rebuild the graph and knowledge for registered repositories."""

    id: str = Field(pattern=_JOB_ID)
    requested_repository_id: str | None = Field(default=None, pattern=_REPOSITORY_ID)
    status: IndexingJobStatus = IndexingJobStatus.QUEUED
    stage: IndexingStage = IndexingStage.QUEUED
    scan_run_id: str | None = None
    error_code: str | None = Field(default=None, pattern=_ERROR_CODE)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancellation_requested_at: datetime | None = None

    @classmethod
    def create(cls, requested_repository_id: str | None = None) -> IndexingJob:
        """Create a queued job without accepting external identifiers or secret-bearing errors."""

        return cls(
            id=f"job:{uuid4()}",
            requested_repository_id=requested_repository_id,
        )

    @property
    def can_cancel(self) -> bool:
        """Return whether a cancellation request can still prevent publication."""

        return self.cancellation_requested_at is None and (
            self.status is IndexingJobStatus.QUEUED
            or (
                self.status is IndexingJobStatus.RUNNING
                and self.stage is not IndexingStage.PUBLISHING
            )
        )

    @field_validator("scan_run_id")
    @classmethod
    def strip_scan_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator(
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "cancellation_requested_at",
    )
    @classmethod
    def timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def valid_state(self) -> IndexingJob:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at cannot precede created_at")
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("completed_at cannot precede created_at")
        if (
            self.cancellation_requested_at is not None
            and self.cancellation_requested_at < self.created_at
        ):
            raise ValueError("cancellation_requested_at cannot precede created_at")

        if self.status is IndexingJobStatus.QUEUED:
            if self.stage is not IndexingStage.QUEUED or self.started_at or self.completed_at:
                raise ValueError("queued jobs must remain at the queued stage")
        elif self.status is IndexingJobStatus.RUNNING:
            if self.stage not in {
                IndexingStage.MATERIALIZING,
                IndexingStage.SCANNING,
                IndexingStage.PUBLISHING,
            }:
                raise ValueError("running jobs require an active indexing stage")
            if self.started_at is None or self.completed_at is not None:
                raise ValueError("running jobs require started_at and no completed_at")
        elif self.status is IndexingJobStatus.SUCCEEDED:
            if self.stage is not IndexingStage.COMPLETE:
                raise ValueError("succeeded jobs must be complete")
            if self.started_at is None or self.completed_at is None:
                raise ValueError("succeeded jobs require start and completion timestamps")
        elif self.status is IndexingJobStatus.FAILED:
            if self.started_at is None or self.completed_at is None:
                raise ValueError("failed jobs require start and completion timestamps")
        elif self.status is IndexingJobStatus.CANCELLED:
            if self.completed_at is None or self.cancellation_requested_at is None:
                raise ValueError("cancelled jobs require cancellation and completion timestamps")
            if self.stage not in {
                IndexingStage.QUEUED,
                IndexingStage.MATERIALIZING,
                IndexingStage.SCANNING,
            }:
                raise ValueError("cancelled jobs must stop before publication")
            if self.stage is not IndexingStage.QUEUED and self.started_at is None:
                raise ValueError("cancelled active jobs require a start timestamp")

        if self.status is IndexingJobStatus.FAILED:
            if self.error_code is None:
                raise ValueError("failed jobs require a safe error_code")
        elif self.error_code is not None:
            raise ValueError("only failed jobs may have error_code")
        if (
            self.cancellation_requested_at is not None
            and self.status
            not in {IndexingJobStatus.RUNNING, IndexingJobStatus.CANCELLED}
        ):
            raise ValueError("only running or cancelled jobs may request cancellation")
        return self


class IndexingJobEvent(FrozenModel):
    """Sanitized, durable progress or failure detail for one indexing job."""

    id: str = Field(pattern=_JOB_ID)
    job_id: str = Field(pattern=_JOB_ID)
    level: JobEventLevel
    stage: IndexingStage
    code: str = Field(pattern=_EVENT_CODE)
    message: str = Field(min_length=1, max_length=2_000)
    error_type: str | None = Field(default=None, pattern=_ERROR_TYPE)
    created_at: datetime = Field(default_factory=_now)

    @classmethod
    def create(
        cls,
        job_id: str,
        *,
        level: JobEventLevel,
        stage: IndexingStage,
        code: str,
        message: str,
        error_type: str | None = None,
    ) -> IndexingJobEvent:
        """Create an event with a caller-controlled safe message and opaque identifier."""

        return cls(
            id=f"event:{uuid4()}",
            job_id=job_id,
            level=level,
            stage=stage,
            code=code,
            message=message,
            error_type=error_type,
        )

    @field_validator("message")
    @classmethod
    def normalized_message(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("event message cannot be empty")
        return normalized

    @field_validator("created_at")
    @classmethod
    def event_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamp must be timezone-aware")
        return value.astimezone(UTC)


def _validate_credential_free_git_url(value: str) -> None:
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("canonical_url contains whitespace or control characters")
    scp_match = _SCP_GIT_URL.fullmatch(value)
    if scp_match:
        if ".." in scp_match.group("path").split("/"):
            raise ValueError("canonical_url contains an unsafe path")
        return
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
        raise ValueError("canonical_url must use credential-free HTTPS or SSH")
    if parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("canonical_url cannot contain credentials, a query, or a fragment")
    if parsed.scheme == "https" and parsed.username is not None:
        raise ValueError("HTTPS canonical_url cannot contain user information")
    if parsed.scheme == "ssh" and parsed.username not in {None, "git"}:
        raise ValueError("SSH canonical_url may use only the git user")
    if not parsed.path or any(part == ".." for part in parsed.path.split("/")):
        raise ValueError("canonical_url contains an unsafe path")
