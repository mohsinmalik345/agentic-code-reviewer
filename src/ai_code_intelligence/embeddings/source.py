from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ai_code_intelligence.domain.knowledge import KnowledgeDocument
from ai_code_intelligence.domain.models import FileKind, KnowledgeGraph, NodeType, ScannedRepository
from ai_code_intelligence.embeddings.models import EmbeddingDocument
from ai_code_intelligence.utils.ids import stable_id
from ai_code_intelligence.utils.redaction import redact_sensitive_text

_IMPORTANT_TYPES = {
    NodeType.CLASS,
    NodeType.FUNCTION,
    NodeType.METHOD,
    NodeType.ENDPOINT,
    NodeType.BUSINESS_RULE,
}


class EmbeddingSourceCollector:
    """Selects knowledge, documentation, and graph-proven important source files."""

    def __init__(self, max_file_bytes: int) -> None:
        self._max_file_bytes = max_file_bytes

    def collect(
        self,
        repositories: tuple[ScannedRepository, ...],
        graph: KnowledgeGraph,
        knowledge: tuple[KnowledgeDocument, ...],
        included_source_paths: Mapping[str, frozenset[str]] | None = None,
    ) -> tuple[EmbeddingDocument, ...]:
        documents = [
            EmbeddingDocument(
                id=stable_id("embedding-document", item.id),
                kind=f"knowledge-{item.kind}",
                source_uri=f"knowledge://{item.id}",
                content=item.markdown,
                repository_id=item.repository_id,
                metadata={"title": item.title},
            )
            for item in knowledge
        ]
        important_paths = {
            (node.repository_id, node.file_path)
            for node in graph.nodes
            if node.type in _IMPORTANT_TYPES and node.repository_id and node.file_path
        }
        seen: set[tuple[str, str]] = set()
        for repository in repositories:
            included_paths = (
                included_source_paths.get(repository.definition.id)
                if included_source_paths is not None
                else None
            )
            for file in repository.files:
                key = (repository.definition.id, file.relative_path)
                if included_paths is not None and file.relative_path not in included_paths:
                    continue
                selected = file.kind in {FileKind.README, FileKind.DOCUMENTATION} or key in important_paths
                if not selected or key in seen or file.size_bytes > self._max_file_bytes:
                    continue
                seen.add(key)
                try:
                    content = Path(file.absolute_path).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if "\x00" in content:
                    continue
                documents.append(
                    EmbeddingDocument(
                        id=stable_id("embedding-document", *key, file.content_sha256),
                        kind="source" if file.kind is FileKind.SOURCE else "documentation",
                        source_uri=f"repository://{repository.definition.id}/{file.relative_path}",
                        content=redact_sensitive_text(content),
                        repository_id=repository.definition.id,
                        metadata={"file_path": file.relative_path, "sha256": file.content_sha256},
                    )
                )
        return tuple(documents)
