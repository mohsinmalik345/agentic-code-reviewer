from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ai_code_intelligence.config import ScannerConfig
from ai_code_intelligence.domain.models import (
    FileKind,
    PackageManifestSummary,
    RepositoryDefinition,
    ScannedDirectory,
    ScannedFile,
    ScannedRepository,
)

_SOURCE_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py"}
_DOCUMENTATION_EXTENSIONS = {".md", ".mdx", ".rst", ".adoc"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


class LocalRepositoryScanner:
    """Inventories local repositories without modifying them or exposing environment values."""

    def __init__(self, config: ScannerConfig) -> None:
        self._config = config
        self._ignored = set(config.ignored_directories)

    def scan_all(self, repositories: tuple[RepositoryDefinition, ...]) -> tuple[ScannedRepository, ...]:
        return tuple(self.scan(repository) for repository in repositories)

    def scan(self, definition: RepositoryDefinition) -> ScannedRepository:
        root = Path(definition.path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"repository is not a directory: {root}")

        directories: list[ScannedDirectory] = []
        files: list[ScannedFile] = []
        environment_names: set[str] = set()
        total_bytes = 0

        for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in self._ignored
                and not (current_path / name).is_symlink()
                and _inside(root, (current_path / name).resolve())
            )
            relative_directory = current_path.relative_to(root).as_posix()
            if relative_directory != ".":
                parent = Path(relative_directory).parent.as_posix()
                directories.append(
                    ScannedDirectory(
                        relative_path=relative_directory,
                        parent_path=None if parent == "." else parent,
                    )
                )

            for name in sorted(file_names):
                absolute = current_path / name
                if absolute.is_symlink() or not absolute.is_file():
                    continue
                resolved = absolute.resolve()
                if not _inside(root, resolved):
                    continue
                relative = resolved.relative_to(root).as_posix()
                kind = classify_file(relative)
                stat = resolved.stat()
                if len(files) >= self._config.max_repository_files:
                    raise RuntimeError(f"repository {definition.id} exceeds configured file-count limit")
                total_bytes += stat.st_size
                if total_bytes > self._config.max_repository_bytes:
                    raise RuntimeError(f"repository {definition.id} exceeds configured total-byte limit")
                files.append(
                    ScannedFile(
                        absolute_path=str(resolved),
                        relative_path=relative,
                        kind=kind,
                        extension=absolute.suffix.lower(),
                        size_bytes=stat.st_size,
                        content_sha256=_sha256(resolved),
                    )
                )
                if kind is FileKind.ENVIRONMENT and stat.st_size <= self._config.max_file_bytes:
                    environment_names.update(_environment_names(resolved))

        manifest = _package_manifest(root / "package.json", self._config.max_file_bytes)
        return ScannedRepository(
            definition=definition,
            root_path=str(root),
            directories=tuple(directories),
            files=tuple(files),
            package_manifest=manifest,
            environment_variable_names=tuple(sorted(environment_names)),
        )


def classify_file(relative_path: str) -> FileKind:
    """Classify a repository file for safe downstream content routing."""

    path = Path(relative_path)
    name = path.name.lower()
    extension = path.suffix.lower()
    if name == "package.json":
        return FileKind.MANIFEST
    if name.startswith("tsconfig") and extension == ".json":
        return FileKind.TSCONFIG
    if name == ".env" or name.startswith(".env."):
        return FileKind.ENVIRONMENT
    if name.startswith("readme") and extension in _DOCUMENTATION_EXTENSIONS:
        return FileKind.README
    if extension in _SOURCE_EXTENSIONS:
        return FileKind.SOURCE
    if extension in _DOCUMENTATION_EXTENSIONS:
        return FileKind.DOCUMENTATION
    return FileKind.OTHER


def _inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _environment_names(path: Path) -> set[str]:
    names: set[str] = set()
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return names
    for line in content.splitlines():
        match = _ENV_LINE.match(line)
        if match and _ENV_NAME.match(match.group(1)):
            names.add(match.group(1))
    return names


def _package_manifest(path: Path, max_bytes: int) -> PackageManifestSummary | None:
    if not path.exists() or path.stat().st_size > max_bytes:
        return None
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    def strings(key: str) -> dict[str, str]:
        value = raw.get(key)
        if not isinstance(value, dict):
            return {}
        return {str(name): str(version) for name, version in value.items() if isinstance(version, str)}

    return PackageManifestSummary(
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        version=raw.get("version") if isinstance(raw.get("version"), str) else None,
        description=raw.get("description") if isinstance(raw.get("description"), str) else None,
        scripts=strings("scripts"),
        dependencies=strings("dependencies"),
        dev_dependencies=strings("devDependencies"),
    )
