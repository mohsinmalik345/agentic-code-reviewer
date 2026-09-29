from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from ai_code_intelligence.config import ArchitectureChatConfig
from ai_code_intelligence.domain.models import GraphNode, GraphRelationship
from ai_code_intelligence.graph.neo4j_store import Neo4jGraphStore
from ai_code_intelligence.knowledge.graph_retrieval import GraphEvidence
from ai_code_intelligence.utils.redaction import redact_json_strings, redact_sensitive_text

_QUERY_TOKEN = re.compile(r"[A-Za-z0-9_./:-]{3,}")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_IGNORED_TERMS = {
    "all",
    "about",
    "architecture",
    "code",
    "does",
    "explain",
    "from",
    "have",
    "into",
    "list",
    "show",
    "that",
    "their",
    "there",
    "these",
    "this",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "with",
    "work",
}
_CANONICAL_ENTITY_TERMS = {
    "apis": "api",
    "classes": "class",
    "constants": "constant",
    "directories": "directory",
    "endpoints": "endpoint",
    "enums": "enum",
    "events": "socketevent",
    "files": "file",
    "functions": "function",
    "interfaces": "interface",
    "methods": "method",
    "repositories": "repository",
    "services": "service",
    "socketevents": "socketevent",
    "variables": "variable",
}


class Neo4jGraphEvidenceRetriever:
    """Retrieves and renders bounded facts from an exact published Neo4j snapshot."""

    def __init__(
        self,
        store_factory: Callable[[], Neo4jGraphStore],
        config: ArchitectureChatConfig,
    ) -> None:
        self._store_factory = store_factory
        self._config = config

    def retrieve(
        self,
        question: str,
        *,
        repository_id: str | None,
        snapshot_id: str,
    ) -> tuple[GraphEvidence, ...]:
        terms = _query_terms(question)
        if not terms:
            return ()
        store = self._store_factory()
        try:
            result = store.search_subgraph(
                terms,
                repository_id=repository_id,
                snapshot_id=snapshot_id,
                seed_limit=self._config.graph_seed_nodes,
                max_nodes=self._config.graph_max_nodes,
                max_relationships=self._config.graph_max_relationships,
            )
        finally:
            store.close()

        node_by_id = {node.id: node for node in result.graph.nodes}
        seed_ids = set(result.seed_node_ids)
        ordered_nodes = sorted(
            result.graph.nodes,
            key=lambda node: (node.id not in seed_ids, node.type.value, node.qualified_name, node.id),
        )
        evidence = [_node_evidence(node, self._config.graph_evidence_characters) for node in ordered_nodes]
        evidence.extend(
            _relationship_evidence(
                relationship,
                node_by_id,
                self._config.graph_evidence_characters,
            )
            for relationship in sorted(
                result.graph.relationships,
                key=lambda item: (
                    item.type.value,
                    node_by_id[item.source_id].qualified_name,
                    node_by_id[item.target_id].qualified_name,
                    item.id,
                ),
            )
        )
        return tuple(evidence)


def _query_terms(question: str) -> tuple[str, ...]:
    terms: set[str] = set()
    for token in _QUERY_TOKEN.findall(question):
        normalized = token.casefold()
        if normalized not in _IGNORED_TERMS:
            terms.add(_CANONICAL_ENTITY_TERMS.get(normalized, normalized))
        for part in _CAMEL_BOUNDARY.split(token):
            part = part.casefold().strip("_./:-")
            if len(part) >= 3 and part not in _IGNORED_TERMS:
                terms.add(_CANONICAL_ENTITY_TERMS.get(part, part))
    return tuple(sorted(terms, key=lambda value: (-len(value), value))[:12])


def _node_evidence(node: GraphNode, limit: int) -> GraphEvidence:
    lines = [
        f"Entity type: {node.type.value}",
        f"Name: {node.name}",
        f"Qualified name: {node.qualified_name}",
    ]
    if node.repository_id is not None:
        lines.append(f"Repository: {node.repository_id}")
    if node.file_path is not None:
        location = node.file_path
        if node.start_line is not None:
            location += f":{node.start_line}"
        if node.end_line is not None and node.end_line != node.start_line:
            location += f"-{node.end_line}"
        lines.append(f"Source location: {location}")
    metadata = _public_metadata(node.metadata)
    if metadata:
        lines.append(f"Metadata: {_bounded_json(metadata, limit // 2)}")
    return GraphEvidence(
        title=f"Neo4j {node.type.value}: {node.name}",
        repository_id=node.repository_id,
        section=f"{node.type.value} {node.name}",
        source_uri="neo4j://published-snapshot/entity",
        content=_bounded_text("\n".join(lines), limit),
    )


def _relationship_evidence(
    relationship: GraphRelationship,
    node_by_id: Mapping[str, GraphNode],
    limit: int,
) -> GraphEvidence:
    source = node_by_id[relationship.source_id]
    target = node_by_id[relationship.target_id]
    location = f"{relationship.evidence.file_path}:{relationship.evidence.start_line}"
    lines = [
        f"Relationship: {source.type.value} {source.qualified_name} "
        f"--{relationship.type.value}--> {target.type.value} {target.qualified_name}",
        f"Repository: {relationship.repository_id or source.repository_id or target.repository_id or 'cross-service'}",
        f"Source evidence: {location}, column {relationship.evidence.start_column}",
        f"Analyzer: {relationship.evidence.analyzer}",
        f"Confidence: {relationship.evidence.confidence}",
    ]
    if relationship.evidence.excerpt:
        lines.append(f"Excerpt: {relationship.evidence.excerpt}")
    metadata = _public_metadata(relationship.metadata)
    if metadata:
        lines.append(f"Metadata: {_bounded_json(metadata, limit // 2)}")
    return GraphEvidence(
        title=f"Neo4j {relationship.type.value}: {source.name} to {target.name}",
        repository_id=relationship.repository_id or source.repository_id,
        section=f"{source.name} {relationship.type.value} {target.name}",
        source_uri="neo4j://published-snapshot/relationship",
        content=_bounded_text("\n".join(lines), limit),
    )


def _public_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): redact_json_strings(value)
        for key, value in metadata.items()
        if not _internal_identifier_key(str(key))
    }


def _internal_identifier_key(key: str) -> bool:
    normalized = key.casefold()
    return normalized in {"id", "ids"} or normalized.endswith(("_id", "_ids"))


def _bounded_json(value: Mapping[str, Any], limit: int) -> str:
    return _bounded_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str),
        limit,
    )


def _bounded_text(value: str, limit: int) -> str:
    redacted = redact_sensitive_text(value).strip()
    if len(redacted) <= limit:
        return redacted
    return redacted[: max(0, limit - 1)].rstrip() + "…"
