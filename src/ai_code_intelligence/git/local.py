from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class GitProvider(Protocol):
    """Read-only Git operations used by change and deployment analysis."""

    def diff(
        self,
        repository_path: str,
        *,
        base_revision: str | None,
        head_revision: str | None,
        staged: bool,
    ) -> str: ...

    def read_file(
        self,
        repository_path: str,
        path: str,
        *,
        revision: str | None = None,
        staged: bool = False,
    ) -> str | None: ...

    def has_revision(self, repository_path: str, revision: str) -> bool: ...

    def tracked_files(self, repository_path: str, revision: str) -> tuple[str, ...]: ...


class LocalGitProvider:
    """Reads unified diffs from a local repository without changing worktree state."""

    def diff(
        self,
        repository_path: str,
        *,
        base_revision: str | None,
        head_revision: str | None,
        staged: bool,
    ) -> str:
        repository = Path(repository_path).resolve(strict=True)
        if base_revision and base_revision.startswith("-"):
            raise ValueError("base revision may not begin with '-'")
        if head_revision and head_revision.startswith("-"):
            raise ValueError("head revision may not begin with '-'")
        if staged and (base_revision or head_revision):
            raise ValueError("staged cannot be combined with explicit revisions")

        command = [
            "git",
            "-c",
            "core.quotePath=false",
            "-c",
            "diff.renameLimit=0",
            "-C",
            str(repository),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--find-renames=50%",
            "--ignore-submodules=all",
            "--diff-algorithm=myers",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "--unified=3",
        ]
        if staged:
            command.append("--cached")
        elif base_revision and head_revision:
            command.extend([base_revision, head_revision])
        elif base_revision:
            command.append(base_revision)
        elif head_revision:
            raise ValueError("head_revision requires base_revision")
        command.append("--")
        result = _run(command)
        if result.returncode != 0:
            message = result.stderr.strip() or "unknown git error"
            raise RuntimeError(f"git diff failed: {message}")
        return result.stdout

    def read_file(
        self,
        repository_path: str,
        path: str,
        *,
        revision: str | None = None,
        staged: bool = False,
    ) -> str | None:
        repository = Path(repository_path).resolve(strict=True)
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Git file path must be repository-relative")
        normalized = relative.as_posix()
        if staged and revision:
            raise ValueError("staged file reads cannot specify a revision")
        if revision and revision.startswith("-"):
            raise ValueError("revision may not begin with '-'")
        if revision or staged:
            object_name = f":{normalized}" if staged else f"{revision}:{normalized}"
            result = _run(["git", "-C", str(repository), "show", object_name])
            return result.stdout if result.returncode == 0 else None
        target = (repository / relative).resolve()
        if repository not in target.parents or not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"Git file is not UTF-8 text: {normalized}") from error

    def has_revision(self, repository_path: str, revision: str) -> bool:
        """Return whether an exact revision exists as a commit in the local object store."""

        repository = Path(repository_path).resolve(strict=True)
        _validate_revision(revision)
        result = _run(
            [
                "git",
                "-C",
                str(repository),
                "rev-parse",
                "--verify",
                "--quiet",
                "--end-of-options",
                f"{revision}^{{commit}}",
            ]
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        message = result.stderr.strip() or "unknown git error"
        raise RuntimeError(f"git revision lookup failed: {message}")

    def tracked_files(self, repository_path: str, revision: str) -> tuple[str, ...]:
        """List the complete tracked-file inventory at one exact revision."""

        repository = Path(repository_path).resolve(strict=True)
        _validate_revision(revision)
        result = _run(
            [
                "git",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "--full-tree",
                "--name-only",
                "-z",
                revision,
                "--",
            ]
        )
        if result.returncode != 0:
            message = result.stderr.strip() or "unknown git error"
            raise RuntimeError(f"git tracked-file listing failed: {message}")
        return tuple(sorted(path for path in result.stdout.split("\0") if path))


def _validate_revision(revision: str) -> None:
    if not revision or revision.startswith("-") or any(character.isspace() for character in revision):
        raise ValueError("revision must be a safe non-empty Git revision")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("git command exceeded 60-second safety timeout") from error
