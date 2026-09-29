from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ai_code_intelligence.config import AgentConfig, ScannerConfig
from ai_code_intelligence.domain.models import FileKind, ScannedFile, ScannedRepository
from ai_code_intelligence.utils.redaction import redact_sensitive_text

_CONFIGURATION_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}
_EXCLUDED_CONFIG_NAMES = {"package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml"}


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One redacted, line-addressable repository source document."""

    repository_id: str
    file_path: str
    content: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """A bounded contiguous source range supplied to one agent batch."""

    id: str
    repository_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """A size- and file-count-bounded group of source chunks."""

    id: str
    repository_id: str
    chunks: tuple[SourceChunk, ...]
    character_count: int


@dataclass(frozen=True, slots=True)
class RepositoryAgentContent:
    """All safe model inputs and loader warnings for one repository."""

    documents: tuple[SourceDocument, ...]
    batches: tuple[SourceBatch, ...]
    discovery_context: str
    warnings: tuple[str, ...]


class RepositoryContentLoader:
    """Reads allowlisted text, redacts credentials, and creates line-addressable Bedrock batches."""

    def __init__(self, agent_config: AgentConfig, scanner_config: ScannerConfig) -> None:
        self._agents = agent_config
        self._scanner = scanner_config

    def load(
        self,
        repository: ScannedRepository,
        included_source_paths: frozenset[str] | None = None,
    ) -> RepositoryAgentContent:
        """Load all source files, or only the exact paths selected by an incremental plan."""

        documents: list[SourceDocument] = []
        warnings: list[str] = []
        for file in repository.files:
            if file.kind is not FileKind.SOURCE:
                continue
            if included_source_paths is not None and file.relative_path not in included_source_paths:
                continue
            if file.size_bytes > self._scanner.max_file_bytes:
                warnings.append(
                    f"{file.relative_path}: skipped because {file.size_bytes} bytes exceeds max_file_bytes"
                )
                continue
            content = _safe_text(file)
            if content is None:
                warnings.append(f"{file.relative_path}: skipped because it is not valid UTF-8 text")
                continue
            documents.append(
                SourceDocument(
                    repository_id=repository.definition.id,
                    file_path=file.relative_path,
                    content=content,
                    lines=tuple(content.splitlines()),
                )
            )

        chunks = [
            chunk
            for document in documents
            for chunk in _chunk_document(
                document,
                self._agents.max_batch_characters,
                self._agents.chunk_overlap_lines,
            )
        ]
        batches = _batch_chunks(
            repository.definition.id,
            chunks,
            self._agents.max_batch_characters,
            self._agents.max_files_per_batch,
        )
        return RepositoryAgentContent(
            documents=tuple(documents),
            batches=tuple(batches),
            discovery_context=self._discovery_context(repository),
            warnings=tuple(warnings),
        )

    def _discovery_context(self, repository: ScannedRepository) -> str:
        inventory = {
            "repository": {
                "id": repository.definition.id,
                "name": repository.definition.name,
                "description": repository.definition.description,
            },
            "package_manifest": (
                repository.package_manifest.model_dump(mode="json") if repository.package_manifest else None
            ),
            "environment_variable_names": repository.environment_variable_names,
            "directories": [directory.relative_path for directory in repository.directories],
            "files": [
                {"path": file.relative_path, "kind": file.kind.value, "size_bytes": file.size_bytes}
                for file in repository.files
            ],
        }
        sections = [json.dumps(inventory, separators=(",", ":"))]
        used = len(sections[0])
        for file in (candidate for candidate in repository.files if _is_discovery_file(candidate)):
            if file.size_bytes > self._scanner.max_file_bytes:
                continue
            content = _safe_text(file)
            if content is None:
                continue
            remaining = self._agents.max_discovery_characters - used
            if remaining <= 200:
                break
            section = f'\n<file path="{file.relative_path}">\n{content[: remaining - 100]}\n</file>'
            sections.append(section)
            used += len(section)
        return "\n".join(sections)


def format_batch(batch: SourceBatch) -> str:
    """Render source with immutable original line numbers for evidence extraction."""

    values: list[str] = []
    for chunk in batch.chunks:
        numbered = "\n".join(
            f"{chunk.start_line + index}|{line}" for index, line in enumerate(chunk.content.splitlines())
        )
        values.append(
            json.dumps(
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "numbered_source": numbered,
                },
                separators=(",", ":"),
            )
        )
    return "\n\n".join(values)


def _safe_text(file: ScannedFile) -> str | None:
    if file.kind is FileKind.ENVIRONMENT:
        return None
    try:
        content = Path(file.absolute_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in content:
        return None
    return redact_sensitive_text(content)


def _is_discovery_file(file: ScannedFile) -> bool:
    if file.kind in {FileKind.MANIFEST, FileKind.TSCONFIG, FileKind.README, FileKind.DOCUMENTATION}:
        return True
    name = Path(file.relative_path).name.lower()
    return file.extension in _CONFIGURATION_EXTENSIONS and name not in _EXCLUDED_CONFIG_NAMES


def _chunk_document(document: SourceDocument, maximum: int, overlap_lines: int) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    start = 0
    while start < len(document.lines):
        end = start
        characters = 0
        while end < len(document.lines):
            next_size = len(document.lines[end]) + 1
            if end > start and characters + next_size > maximum:
                break
            characters += next_size
            end += 1
        chunks.append(
            SourceChunk(
                id=f"{document.file_path}:{start + 1}-{end}",
                repository_id=document.repository_id,
                file_path=document.file_path,
                start_line=start + 1,
                end_line=end,
                content="\n".join(document.lines[start:end]),
            )
        )
        if end >= len(document.lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def _batch_chunks(
    repository_id: str,
    chunks: list[SourceChunk],
    maximum: int,
    max_files: int,
) -> list[SourceBatch]:
    batches: list[SourceBatch] = []
    current: list[SourceChunk] = []
    characters = 0
    paths: set[str] = set()

    def flush() -> None:
        nonlocal current, characters, paths
        if current:
            batches.append(
                SourceBatch(
                    id=f"{repository_id}-batch-{len(batches) + 1:04d}",
                    repository_id=repository_id,
                    chunks=tuple(current),
                    character_count=characters,
                )
            )
        current = []
        characters = 0
        paths = set()

    for chunk in chunks:
        next_paths = paths | {chunk.file_path}
        if current and (characters + len(chunk.content) > maximum or len(next_paths) > max_files):
            flush()
        current.append(chunk)
        characters += len(chunk.content)
        paths.add(chunk.file_path)
    flush()
    return batches
