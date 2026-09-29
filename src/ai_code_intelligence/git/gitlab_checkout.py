from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

from ai_code_intelligence.domain.change import RepositoryChangeSet
from ai_code_intelligence.git.change_set import GitChangeSetResolver
from ai_code_intelligence.git.local import GitProvider, LocalGitProvider

_REPOSITORY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_HOST = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_COMMIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MANAGED_MARKER = "code-intelligence-managed.json"
_TOKEN_ENV = "AI_CODE_INTELLIGENCE_GIT_TOKEN"
_USERNAME_ENV = "AI_CODE_INTELLIGENCE_GIT_USERNAME"


@dataclass(frozen=True, slots=True)
class GitCheckout:
    """A managed checkout pinned to one exact Git commit."""

    path: Path
    commit_sha: str


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Minimal process result returned by an injectable Git command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class GitCommandRunner(Protocol):
    """Executes one already-validated Git argument vector."""

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: int,
    ) -> GitCommandResult: ...


class SubprocessGitCommandRunner:
    """Runs Git without a shell and returns bounded UTF-8 process output."""

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: int,
    ) -> GitCommandResult:
        try:
            completed = subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=dict(env),
                timeout=timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("git executable was not found") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("git command exceeded its safety timeout") from error
        return GitCommandResult(completed.returncode, completed.stdout, completed.stderr)


class ManagedGitLabCheckout:
    """Clones or refreshes an allowlisted GitLab HTTPS repository under a managed root.

    Authentication is supplied through a temporary askpass helper and inherited
    environment variables. Credentials never appear in the Git argument vector,
    repository remote URL, managed marker, or raised command error.
    """

    def __init__(
        self,
        managed_root: str | Path,
        allowed_hosts: tuple[str, ...],
        *,
        token: str | None = None,
        username: str = "oauth2",
        runner: GitCommandRunner | None = None,
        timeout_seconds: int = 300,
        fetch_timeout_seconds: int | None = None,
    ) -> None:
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if fetch_timeout_seconds is not None and fetch_timeout_seconds < 1:
            raise ValueError("fetch_timeout_seconds must be positive")
        if token is not None and (not token or _has_control_character(token)):
            raise ValueError("token must be non-empty and contain no control characters")
        if not username or _has_control_character(username):
            raise ValueError("username must be non-empty and contain no control characters")

        hosts = tuple(_validate_allowed_host(host) for host in allowed_hosts)
        if not hosts:
            raise ValueError("at least one GitLab host must be allowlisted")
        if len(hosts) != len(set(hosts)):
            raise ValueError("allowed GitLab hosts must be unique")

        root = Path(managed_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self._root = root.resolve(strict=True)
        if not self._root.is_dir():
            raise ValueError("managed checkout root is not a directory")
        self._hooks = self._root / ".hooks-disabled"
        self._hooks.mkdir(exist_ok=True)
        self._allowed_hosts = frozenset(hosts)
        self._token = token
        self._username = username
        self._runner = runner or SubprocessGitCommandRunner()
        self._timeout_seconds = timeout_seconds
        self._fetch_timeout_seconds = fetch_timeout_seconds or timeout_seconds

    def checkout(self, repository_id: str, url: str, ref: str | None = None) -> GitCheckout:
        """Return a clean managed checkout and the exact commit selected by ``ref``."""

        if not _REPOSITORY_ID.fullmatch(repository_id):
            raise ValueError("repository id must be a lowercase slug without path traversal")
        canonical_url = validate_gitlab_url(url, tuple(self._allowed_hosts))
        _validate_ref(ref)
        target = self._target(repository_id)

        with self._git_environment() as environment:
            if target.exists():
                commit_sha = self._refresh_existing(target, canonical_url, ref, environment)
                return GitCheckout(target, commit_sha)
            return self._clone_atomically(repository_id, target, canonical_url, ref, environment)

    def changes_since(
        self,
        repository_id: str,
        checkout: GitCheckout,
        last_indexed_commit: str | None,
        *,
        provider: GitProvider | None = None,
    ) -> RepositoryChangeSet:
        """Compare a managed checkout with its last successfully indexed exact commit.

        The optional provider exists for unit tests and alternative local Git
        implementations. Production callers use the read-only ``LocalGitProvider``.
        """

        if not _REPOSITORY_ID.fullmatch(repository_id):
            raise ValueError("repository id must be a lowercase slug without path traversal")
        target = self._target(repository_id)
        self._ensure_checkout_directory(target)
        if checkout.path.resolve(strict=True) != target.resolve(strict=True):
            raise ValueError("Git checkout does not belong to the requested managed repository")
        marker = self._read_marker(target)
        if marker.get("commit_sha") != checkout.commit_sha:
            raise RuntimeError("managed checkout marker does not match the resolved commit")
        return GitChangeSetResolver(provider or LocalGitProvider()).resolve(
            repository_id,
            str(target),
            base_commit=last_indexed_commit,
            head_commit=checkout.commit_sha,
        )

    def _clone_atomically(
        self,
        repository_id: str,
        target: Path,
        canonical_url: str,
        ref: str | None,
        environment: Mapping[str, str],
    ) -> GitCheckout:
        temporary_parent = Path(tempfile.mkdtemp(prefix=f".{repository_id}-", dir=self._root)).resolve(
            strict=True
        )
        self._ensure_inside_root(temporary_parent)
        temporary_checkout = temporary_parent / "checkout"
        try:
            self._invoke(
                "clone",
                (
                    *self._base_command(),
                    "clone",
                    "--no-checkout",
                    "--no-recurse-submodules",
                    "--origin",
                    "origin",
                    "--",
                    canonical_url,
                    str(temporary_checkout),
                ),
                environment,
                timeout_seconds=self._timeout_seconds,
            )
            self._ensure_checkout_directory(temporary_checkout)
            commit_sha = self._select_and_checkout(temporary_checkout, ref, environment, fetch=False)
            self._write_marker(temporary_checkout, canonical_url, commit_sha)
            try:
                temporary_checkout.replace(target)
            except OSError:
                if not target.exists():
                    raise
                commit_sha = self._refresh_existing(target, canonical_url, ref, environment)
            return GitCheckout(target.resolve(strict=True), commit_sha)
        finally:
            self._safe_remove_temporary(temporary_parent)

    def _refresh_existing(
        self,
        target: Path,
        canonical_url: str,
        ref: str | None,
        environment: Mapping[str, str],
    ) -> str:
        self._ensure_checkout_directory(target)
        marker = self._read_marker(target)
        if marker.get("canonical_url") != canonical_url:
            raise ValueError("managed checkout URL does not match the requested GitLab repository")
        remote = self._invoke(
            "inspect remote",
            (*self._base_command(), "-C", str(target), "remote", "get-url", "origin"),
            environment,
        ).stdout.strip()
        try:
            normalized_remote = validate_gitlab_url(remote, tuple(self._allowed_hosts))
        except ValueError as error:
            raise RuntimeError("managed checkout has an invalid origin remote") from error
        if normalized_remote != canonical_url:
            raise RuntimeError("managed checkout origin does not match its registered URL")
        commit_sha = self._select_and_checkout(target, ref, environment, fetch=True)
        self._write_marker(target, canonical_url, commit_sha)
        return commit_sha

    def _select_and_checkout(
        self,
        checkout: Path,
        ref: str | None,
        environment: Mapping[str, str],
        *,
        fetch: bool,
    ) -> str:
        if fetch:
            self._invoke(
                "fetch",
                (
                    *self._base_command(),
                    "-C",
                    str(checkout),
                    "fetch",
                    "--prune",
                    "--tags",
                    "--force",
                    "--no-recurse-submodules",
                    "origin",
                ),
                environment,
                timeout_seconds=self._fetch_timeout_seconds,
            )
        if ref is None:
            self._invoke(
                "resolve default branch",
                (
                    *self._base_command(),
                    "-C",
                    str(checkout),
                    "remote",
                    "set-head",
                    "origin",
                    "--auto",
                ),
                environment,
            )
        commit_sha = self._resolve_commit(checkout, ref, environment)
        self._invoke(
            "checkout",
            (
                *self._base_command(),
                "-C",
                str(checkout),
                "checkout",
                "--detach",
                "--force",
                commit_sha,
            ),
            environment,
        )
        self._invoke(
            "clean checkout",
            (*self._base_command(), "-C", str(checkout), "clean", "-ffd", "-x"),
            environment,
        )
        verified = self._rev_parse(checkout, "HEAD", environment)
        if verified != commit_sha:
            raise RuntimeError("Git checkout HEAD did not match the resolved commit")
        return commit_sha

    def _resolve_commit(
        self,
        checkout: Path,
        ref: str | None,
        environment: Mapping[str, str],
    ) -> str:
        for candidate in _revision_candidates(ref):
            value = self._rev_parse(checkout, candidate, environment, required=False)
            if value is not None:
                return value
        raise ValueError("requested Git ref does not resolve to a fetched commit")

    def _rev_parse(
        self,
        checkout: Path,
        candidate: str,
        environment: Mapping[str, str],
        *,
        required: bool = True,
    ) -> str | None:
        result = self._invoke(
            "resolve revision",
            (
                *self._base_command(),
                "-C",
                str(checkout),
                "rev-parse",
                "--verify",
                "--quiet",
                "--end-of-options",
                f"{candidate}^{{commit}}",
            ),
            environment,
            required=required,
        )
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        if not _COMMIT_SHA.fullmatch(value):
            raise RuntimeError("Git did not return an exact commit SHA")
        return value

    def _base_command(self) -> tuple[str, ...]:
        return (
            "git",
            "-c",
            f"core.hooksPath={self._hooks}",
            "-c",
            "submodule.recurse=false",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "protocol.ext.allow=never",
            "-c",
            "credential.helper=",
        )

    def _invoke(
        self,
        stage: str,
        argv: Sequence[str],
        environment: Mapping[str, str],
        *,
        required: bool = True,
        timeout_seconds: int | None = None,
    ) -> GitCommandResult:
        try:
            result = self._runner.run(
                argv,
                env=environment,
                timeout_seconds=timeout_seconds or self._timeout_seconds,
            )
        except Exception as error:
            message = self._redact(str(error))
            raise RuntimeError(f"Git {stage} failed: {message or type(error).__name__}") from error
        if required and result.returncode != 0:
            detail = self._redact(result.stderr.strip() or result.stdout.strip())
            if len(detail) > 2_000:
                detail = f"{detail[:2_000]}..."
            raise RuntimeError(
                f"Git {stage} failed with exit code {result.returncode}" + (f": {detail}" if detail else "")
            )
        return result

    @contextmanager
    def _git_environment(self) -> Iterator[dict[str, str]]:
        environment = dict(os.environ)
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            }
        )
        helper_directory: Path | None = None
        try:
            if self._token is not None:
                helper_directory, helper = self._create_askpass_helper()
                environment.update(
                    {
                        "GIT_ASKPASS": str(helper),
                        "GIT_ASKPASS_REQUIRE": "force",
                        _TOKEN_ENV: self._token,
                        _USERNAME_ENV: self._username,
                    }
                )
            yield environment
        finally:
            if helper_directory is not None:
                self._safe_remove_temporary(helper_directory)

    def _create_askpass_helper(self) -> tuple[Path, Path]:
        directory = Path(tempfile.mkdtemp(prefix=".git-auth-", dir=self._root)).resolve(strict=True)
        self._ensure_inside_root(directory)
        script = directory / "askpass.py"
        script.write_text(
            "import os, sys\n"
            "prompt = ' '.join(sys.argv[1:]).lower()\n"
            f"name = '{_USERNAME_ENV}' if 'username' in prompt else '{_TOKEN_ENV}'\n"
            "sys.stdout.write(os.environ.get(name, ''))\n",
            encoding="utf-8",
            newline="\n",
        )
        if os.name == "nt":
            helper = directory / "askpass.cmd"
            helper.write_text(
                f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
                encoding="utf-8",
                newline="",
            )
        else:
            helper = directory / "askpass"
            helper.write_text(
                f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n',
                encoding="utf-8",
                newline="\n",
            )
            helper.chmod(0o700)
        return directory, helper

    def _target(self, repository_id: str) -> Path:
        target = (self._root / repository_id).resolve(strict=False)
        self._ensure_inside_root(target)
        return target

    def _ensure_checkout_directory(self, target: Path) -> None:
        if target.is_symlink() or target.is_junction():
            raise RuntimeError("managed checkout path must not be a link or junction")
        resolved = target.resolve(strict=True)
        self._ensure_inside_root(resolved)
        if not resolved.is_dir():
            raise RuntimeError("managed checkout path is not a safe directory")
        git_directory = resolved / ".git"
        if not git_directory.is_dir() or git_directory.is_symlink() or git_directory.is_junction():
            raise RuntimeError("managed checkout is not a standalone Git worktree")

    def _ensure_inside_root(self, candidate: Path) -> None:
        if candidate == self._root or self._root not in candidate.parents:
            raise RuntimeError("managed checkout path escaped its configured root")

    def _marker_path(self, checkout: Path) -> Path:
        return checkout / ".git" / _MANAGED_MARKER

    def _write_marker(self, checkout: Path, canonical_url: str, commit_sha: str) -> None:
        marker = self._marker_path(checkout)
        temporary = marker.with_suffix(f".tmp-{uuid4().hex}")
        temporary.write_text(
            json.dumps({"canonical_url": canonical_url, "commit_sha": commit_sha}),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(marker)

    def _read_marker(self, checkout: Path) -> dict[str, str]:
        marker = self._marker_path(checkout)
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing checkout is not managed by this application") from error
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise RuntimeError("managed checkout marker is invalid")
        return {key: item for key, item in value.items() if isinstance(item, str)}

    def _safe_remove_temporary(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        self._ensure_inside_root(resolved)
        if resolved.name.startswith("."):
            shutil.rmtree(resolved, ignore_errors=True)

    def _redact(self, value: str) -> str:
        if self._token is None:
            return value
        redacted = value.replace(self._token, "[REDACTED]")
        encoded = quote(self._token, safe="")
        if encoded != self._token:
            redacted = redacted.replace(encoded, "[REDACTED]")
        return redacted


def validate_gitlab_url(url: str, allowed_hosts: tuple[str, ...]) -> str:
    """Validate and canonicalize an HTTPS GitLab clone URL against exact hosts."""

    if not isinstance(url, str) or not url or url != url.strip() or len(url) > 2_048:
        raise ValueError("GitLab URL must be a non-empty bounded string")
    hosts = frozenset(_validate_allowed_host(host) for host in allowed_hosts)
    if not hosts:
        raise ValueError("at least one GitLab host must be allowlisted")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("GitLab URL is malformed") from error
    if parsed.scheme != "https":
        raise ValueError("GitLab URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("GitLab URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("GitLab URL must not contain a query string or fragment")
    if port not in (None, 443):
        raise ValueError("GitLab URL must use the HTTPS default port")
    hostname = parsed.hostname
    if hostname is None or hostname != hostname.rstrip("."):
        raise ValueError("GitLab URL host is invalid")
    hostname = hostname.lower()
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("GitLab URL must not use an IP-literal host")
    if hostname not in hosts:
        raise ValueError("GitLab URL host is not allowlisted")
    if "%" in parsed.path or "\\" in parsed.path or "//" in parsed.path:
        raise ValueError("GitLab URL path contains unsafe encoding or separators")
    segments = parsed.path.strip("/").split("/")
    if len(segments) < 2 or any(
        not segment or segment in {".", ".."} or not _PATH_SEGMENT.fullmatch(segment) for segment in segments
    ):
        raise ValueError("GitLab URL must identify a namespace and repository")
    if segments[-1].endswith(".git"):
        segments[-1] = segments[-1][:-4]
    if not segments[-1] or segments[-1] in {".", ".."}:
        raise ValueError("GitLab URL repository name is invalid")
    canonical_path = f"/{'/'.join(segments)}.git"
    return urlunsplit(("https", hostname, canonical_path, "", ""))


def validate_git_ref(ref: str | None) -> str | None:
    """Validate a caller-supplied branch, tag, or exact commit without invoking Git."""

    _validate_ref(ref)
    return ref


def _validate_allowed_host(host: str) -> str:
    if not isinstance(host, str) or host != host.strip() or host != host.lower():
        raise ValueError("allowlisted GitLab hosts must be lowercase DNS names")
    if not _HOST.fullmatch(host) or host.endswith("."):
        raise ValueError("allowlisted GitLab host is invalid")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    raise ValueError("allowlisted GitLab hosts must not be IP literals")


def _validate_ref(ref: str | None) -> None:
    if ref is None:
        return
    if not isinstance(ref, str) or not ref or ref != ref.strip() or len(ref) > 255:
        raise ValueError("Git ref is invalid")
    if _COMMIT_SHA.fullmatch(ref):
        return
    value = ref
    if value.startswith("refs/heads/"):
        value = value.removeprefix("refs/heads/")
    elif value.startswith("refs/tags/"):
        value = value.removeprefix("refs/tags/")
    elif value.startswith("refs/"):
        raise ValueError("only branch and tag refs are supported")
    if (
        not value
        or ref in {"HEAD", "@"}
        or ref.startswith("-")
        or ref.endswith(("/", "."))
        or ref.startswith("/")
        or "//" in ref
        or ".." in ref
        or "@{" in ref
        or any(character in ref for character in " ~^:?*[\\")
        or _has_control_character(ref)
    ):
        raise ValueError("Git ref is invalid")
    components = value.split("/")
    if any(
        not component
        or component.startswith(".")
        or component.endswith(".lock")
        or len(component.encode("utf-8")) > 255
        for component in components
    ):
        raise ValueError("Git ref is invalid")


def _revision_candidates(ref: str | None) -> tuple[str, ...]:
    if ref is None:
        return ("refs/remotes/origin/HEAD",)
    if _COMMIT_SHA.fullmatch(ref):
        return (ref,)
    if ref.startswith("refs/heads/"):
        return (f"refs/remotes/origin/{ref.removeprefix('refs/heads/')}",)
    if ref.startswith("refs/tags/"):
        return (ref,)
    return (f"refs/remotes/origin/{ref}", f"refs/tags/{ref}")


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)
