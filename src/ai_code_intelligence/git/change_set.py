from __future__ import annotations

import re

from ai_code_intelligence.domain.change import (
    ChangedFile,
    ChangeSetBasis,
    FileChangeStatus,
    RepositoryChangeSet,
)
from ai_code_intelligence.git.diff_parser import UnifiedDiffParser
from ai_code_intelligence.git.local import GitProvider

_COMMIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class GitChangeSetResolver:
    """Computes a deterministic repository change set from exact Git commits.

    A missing base is never approximated from the current tree. That case returns
    ``BASE_UNAVAILABLE`` so the indexing pipeline can safely rebuild the repository
    instead of retaining relationships for paths that may have been deleted.
    """

    def __init__(self, provider: GitProvider, parser: UnifiedDiffParser | None = None) -> None:
        self._provider = provider
        self._parser = parser or UnifiedDiffParser()

    def resolve(
        self,
        repository_id: str,
        repository_path: str,
        *,
        base_commit: str | None,
        head_commit: str,
    ) -> RepositoryChangeSet:
        """Compare ``base_commit`` with ``head_commit`` without changing the worktree."""

        base = _exact_commit(base_commit, "base_commit") if base_commit is not None else None
        head = _exact_commit(head_commit, "head_commit")
        if not self._provider.has_revision(repository_path, head):
            raise RuntimeError("newly resolved head commit is unavailable in the managed checkout")

        if base is None:
            paths = self._provider.tracked_files(repository_path, head)
            return RepositoryChangeSet(
                repository_id=repository_id,
                base_commit=None,
                head_commit=head,
                basis=ChangeSetBasis.FIRST_INDEX,
                files=tuple(
                    ChangedFile(path=path, status=FileChangeStatus.ADDED) for path in paths
                ),
                head_paths=paths,
            )

        if base == head:
            return RepositoryChangeSet(
                repository_id=repository_id,
                base_commit=base,
                head_commit=head,
                basis=ChangeSetBasis.EXACT_COMMITS,
                unified_diff="",
            )

        if not self._provider.has_revision(repository_path, base):
            return RepositoryChangeSet(
                repository_id=repository_id,
                base_commit=base,
                head_commit=head,
                basis=ChangeSetBasis.BASE_UNAVAILABLE,
                head_paths=self._provider.tracked_files(repository_path, head),
            )

        diff = self._provider.diff(
            repository_path,
            base_revision=base,
            head_revision=head,
            staged=False,
        )
        return RepositoryChangeSet(
            repository_id=repository_id,
            base_commit=base,
            head_commit=head,
            basis=ChangeSetBasis.EXACT_COMMITS,
            files=self._parser.parse(diff),
            unified_diff=diff,
        )


def _exact_commit(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not _COMMIT_SHA.fullmatch(normalized):
        raise ValueError(f"{label} must be a full Git SHA-1 or SHA-256 identifier")
    return normalized
