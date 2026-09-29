from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from ai_code_intelligence.domain.models import FrozenModel, GraphStatistics

_REPOSITORY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_COMMIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class KnowledgeDocument(FrozenModel):
    """Generated Markdown knowledge with graph citation allowlists."""

    id: str
    kind: Literal["repository", "central", "repository-human", "central-human"]
    title: str
    markdown: str
    repository_id: str | None = None
    evidence_node_ids: tuple[str, ...] = ()
    evidence_relationship_ids: tuple[str, ...] = ()
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def repository_scope_matches_kind(self) -> KnowledgeDocument:
        repository_kind = self.kind in {"repository", "repository-human"}
        if repository_kind and self.repository_id is None:
            raise ValueError("repository knowledge document requires a repository id")
        if not repository_kind and self.repository_id is not None:
            raise ValueError("central knowledge document cannot have a repository id")
        return self


class StoredDocument(FrozenModel):
    """Content-addressed metadata for a persisted knowledge object."""

    document_id: str
    uri: str
    content_sha256: str
    size_bytes: int = Field(ge=0)


class KnowledgeDocumentRecord(FrozenModel):
    """Immutable audit pointer for a knowledge document from a completed scan."""

    document_id: str = Field(min_length=1)
    scan_run_id: str = Field(min_length=1)
    repository_id: str | None = None
    kind: Literal["repository", "central", "repository-human", "central-human"]
    uri: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("uri")
    @classmethod
    def valid_uri(cls, value: str) -> str:
        return _validate_knowledge_uri(value)

    @model_validator(mode="after")
    def repository_scope_matches_kind(self) -> KnowledgeDocumentRecord:
        repository_kind = self.kind in {"repository", "repository-human"}
        if repository_kind and self.repository_id is None:
            raise ValueError("repository knowledge record requires a repository id")
        if not repository_kind and self.repository_id is not None:
            raise ValueError("central knowledge record cannot have a repository id")
        return self


class KnowledgeManifestRepository(FrozenModel):
    """Repository document pointer from a completed portfolio publication."""

    id: str
    commit_sha: str | None = Field(validation_alias="commitSha")
    knowledge_uri: str = Field(validation_alias="knowledgeUri")
    human_knowledge_uri: str | None = Field(
        default=None,
        validation_alias="humanKnowledgeUri",
    )

    @field_validator("id")
    @classmethod
    def valid_repository_id(cls, value: str) -> str:
        if not _REPOSITORY_ID.fullmatch(value):
            raise ValueError("manifest repository id is invalid")
        return value

    @field_validator("commit_sha")
    @classmethod
    def valid_commit_sha(cls, value: str | None) -> str | None:
        if value is not None and not _COMMIT_SHA.fullmatch(value):
            raise ValueError("manifest commit SHA is invalid")
        return value

    @field_validator("knowledge_uri")
    @classmethod
    def valid_knowledge_uri(cls, value: str) -> str:
        return _validate_knowledge_uri(value)

    @field_validator("human_knowledge_uri")
    @classmethod
    def valid_human_knowledge_uri(cls, value: str | None) -> str | None:
        return _validate_knowledge_uri(value) if value is not None else None


class KnowledgeManifestGraph(FrozenModel):
    """Graph snapshot metadata associated with a published knowledge portfolio."""

    neo4j_snapshot_id: str | None = Field(validation_alias="neo4jSnapshotId")
    statistics: GraphStatistics


class KnowledgeManifest(FrozenModel):
    """Validated pointer set for the latest atomically published portfolio."""

    schema_version: Literal[1, 2] = Field(validation_alias="schemaVersion")
    scan_run_id: str = Field(min_length=1, validation_alias="scanRunId")
    generated_at: datetime = Field(validation_alias="generatedAt")
    repositories: tuple[KnowledgeManifestRepository, ...]
    central_knowledge_uri: str = Field(validation_alias="centralKnowledgeUri")
    human_central_knowledge_uri: str | None = Field(
        default=None,
        validation_alias="humanCentralKnowledgeUri",
    )
    graph: KnowledgeManifestGraph

    @field_validator("generated_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("manifest generatedAt must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("central_knowledge_uri")
    @classmethod
    def valid_central_knowledge_uri(cls, value: str) -> str:
        return _validate_knowledge_uri(value)

    @field_validator("human_central_knowledge_uri")
    @classmethod
    def valid_human_central_knowledge_uri(cls, value: str | None) -> str | None:
        return _validate_knowledge_uri(value) if value is not None else None

    @model_validator(mode="after")
    def unique_repository_entries(self) -> KnowledgeManifest:
        repository_ids = [repository.id for repository in self.repositories]
        if len(repository_ids) != len(set(repository_ids)):
            raise ValueError("manifest contains duplicate repository ids")
        if self.schema_version == 2 and (
            self.human_central_knowledge_uri is None
            or any(repository.human_knowledge_uri is None for repository in self.repositories)
        ):
            raise ValueError("schema version 2 requires every human-readable knowledge pointer")
        return self


class KnowledgeDocumentView(FrozenModel):
    """Authenticated, read-only Markdown response exposed by the application layer."""

    kind: Literal["repository", "central", "repository-human", "central-human"]
    title: str
    repository_id: str | None = None
    source_uri: str
    markdown: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    scan_run_id: str
    generated_at: datetime


def _validate_knowledge_uri(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("knowledge URI cannot be empty or contain surrounding whitespace")
    if any(ord(character) < 32 for character in value):
        raise ValueError("knowledge URI contains control characters")
    if Path(value).is_absolute():
        return value
    if any(character.isspace() for character in value):
        raise ValueError("S3 knowledge URI contains whitespace")
    parsed = urlsplit(value)
    if parsed.scheme:
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError("knowledge URI must be an S3 object URI or absolute local path")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("knowledge URI cannot contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("knowledge URI cannot contain a query or fragment")
        if any(part in {".", ".."} for part in parsed.path.split("/")):
            raise ValueError("knowledge URI contains an unsafe path")
        return value
    raise ValueError("knowledge URI must be an S3 object URI or absolute local path")
