from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ai_code_intelligence.domain.models import FrozenModel, GraphNode

_COMMIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class FileChangeStatus(StrEnum):
    """Deterministic Git file-change classifications."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class ChangeSetBasis(StrEnum):
    """Evidence available when a repository change set was computed."""

    EXACT_COMMITS = "exact_commits"
    FIRST_INDEX = "first_index"
    BASE_UNAVAILABLE = "base_unavailable"


class DiffHunk(FrozenModel):
    """One parsed unified-diff hunk with old and new line ranges."""

    old_start: int = Field(ge=0)
    old_lines: int = Field(ge=0)
    new_start: int = Field(ge=0)
    new_lines: int = Field(ge=0)
    header: str
    lines: tuple[str, ...]


class ChangedFile(FrozenModel):
    """File-level Git change and its parsed hunks."""

    path: str
    old_path: str | None = None
    status: FileChangeStatus
    additions: int = 0
    deletions: int = 0
    binary: bool = False
    hunks: tuple[DiffHunk, ...] = ()

    @field_validator("path", "old_path", mode="before")
    @classmethod
    def safe_repository_path(cls, value: object) -> object:
        """Normalize separators and reject paths that could escape a checkout."""

        if value is None or not isinstance(value, str):
            return value
        normalized = value.replace("\\", "/")
        if (
            not normalized
            or normalized != normalized.strip()
            or normalized.startswith("/")
            or any(ord(character) < 32 for character in normalized)
        ):
            raise ValueError("changed file path must be a safe repository-relative path")
        path = PurePosixPath(normalized)
        if normalized in {".", ".."} or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("changed file path must not contain traversal segments")
        return path.as_posix()

    @model_validator(mode="after")
    def paths_match_status(self) -> ChangedFile:
        """Keep source and destination path semantics unambiguous for graph patching."""

        if self.status is FileChangeStatus.ADDED and self.old_path is not None:
            raise ValueError("added files cannot have an old path")
        if self.status is FileChangeStatus.RENAMED:
            if self.old_path is None or self.old_path == self.path:
                raise ValueError("renamed files require a distinct old path")
        elif self.old_path not in {None, self.path}:
            raise ValueError("only renamed files can have a distinct old path")
        return self

    @property
    def current_path(self) -> str | None:
        """Path present at the head commit, or ``None`` for a deletion."""

        return None if self.status is FileChangeStatus.DELETED else self.path

    @property
    def previous_path(self) -> str | None:
        """Path present at the base commit, or ``None`` for an addition."""

        if self.status is FileChangeStatus.ADDED:
            return None
        if self.status is FileChangeStatus.RENAMED:
            return self.old_path
        return self.path

    @property
    def affected_paths(self) -> tuple[str, ...]:
        """Unique base/head paths that downstream graph patches must consider."""

        previous = self.previous_path
        current = self.current_path
        if previous is None:
            return (current,) if current is not None else ()
        if current is None or current == previous:
            return (previous,)
        return (previous, current)


class RepositoryChangeSet(FrozenModel):
    """Deterministic file changes between one indexed commit and a new exact commit.

    ``BASE_UNAVAILABLE`` deliberately carries no inferred file changes because Git
    cannot prove which paths were deleted. Callers must honor ``requires_full_reindex``
    and may use ``head_paths`` only as the inventory of the new snapshot.
    """

    repository_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    base_commit: str | None = None
    head_commit: str
    basis: ChangeSetBasis
    files: tuple[ChangedFile, ...] = ()
    head_paths: tuple[str, ...] = ()
    unified_diff: str | None = None

    @field_validator("base_commit", "head_commit")
    @classmethod
    def exact_commit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not _COMMIT_SHA.fullmatch(normalized):
            raise ValueError("change-set commits must be full Git SHA-1 or SHA-256 identifiers")
        return normalized

    @field_validator("head_paths")
    @classmethod
    def sorted_unique_head_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_safe_repository_path(path) for path in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("head_paths must be sorted and unique")
        return normalized

    @model_validator(mode="after")
    def evidence_matches_basis(self) -> RepositoryChangeSet:
        if self.basis is ChangeSetBasis.FIRST_INDEX:
            if self.base_commit is not None:
                raise ValueError("first-index change sets cannot have a base commit")
            if self.unified_diff is not None:
                raise ValueError("first-index change sets cannot contain a commit diff")
            if any(file.status is not FileChangeStatus.ADDED for file in self.files):
                raise ValueError("first-index files must all be additions")
            if tuple(file.path for file in self.files) != self.head_paths:
                raise ValueError("first-index files must match the complete head inventory")
        elif self.base_commit is None:
            raise ValueError("non-first-index change sets require a base commit")

        if self.basis is ChangeSetBasis.BASE_UNAVAILABLE:
            if self.files or self.unified_diff is not None:
                raise ValueError("an unavailable base cannot have inferred file changes")
        elif self.basis is ChangeSetBasis.EXACT_COMMITS and self.unified_diff is None:
            raise ValueError("exact-commit change sets require their deterministic unified diff")

        keys = tuple((file.status, file.previous_path, file.current_path) for file in self.files)
        if len(keys) != len(set(keys)):
            raise ValueError("change-set files must be unique")
        return self

    @property
    def requires_full_reindex(self) -> bool:
        """Whether selective patching would risk leaving stale graph facts."""

        return self.basis is not ChangeSetBasis.EXACT_COMMITS

    @property
    def is_unchanged(self) -> bool:
        """Whether two proven exact commits contain no file changes."""

        return self.basis is ChangeSetBasis.EXACT_COMMITS and not self.files

    @property
    def affected_paths(self) -> tuple[str, ...]:
        """Sorted base/head paths touched by an exact comparison."""

        return tuple(sorted({path for file in self.files for path in file.affected_paths}))

    @property
    def deleted_paths(self) -> tuple[str, ...]:
        """Paths proven absent at the head commit, including rename sources."""

        values = {
            path
            for file in self.files
            for path in (file.previous_path,)
            if path is not None and path != file.current_path
        }
        values.update(file.path for file in self.files if file.status is FileChangeStatus.DELETED)
        return tuple(sorted(values))


def _safe_repository_path(value: str) -> str:
    validated = ChangedFile(path=value, status=FileChangeStatus.ADDED)
    return validated.path


class DependencyChange(FrozenModel):
    """Exact package version delta between complete manifests."""

    package_name: str
    change: Literal["added", "removed", "updated"]
    old_version: str | None = None
    new_version: str | None = None


class CodeChangeAnalysis(FrozenModel):
    """Structured mapping from Git diff to validated graph entities."""

    repository_id: str
    base_revision: str | None = None
    head_revision: str | None = None
    staged: bool = False
    diff: str
    changed_files: tuple[ChangedFile, ...]
    changed_functions: tuple[GraphNode, ...] = ()
    changed_classes: tuple[GraphNode, ...] = ()
    changed_endpoints: tuple[GraphNode, ...] = ()
    changed_apis: tuple[GraphNode, ...] = ()
    changed_business_rules: tuple[GraphNode, ...] = ()
    changed_dependencies: tuple[DependencyChange, ...] = ()
