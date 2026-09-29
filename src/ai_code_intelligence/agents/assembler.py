from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ai_code_intelligence.agents.content import SourceDocument
from ai_code_intelligence.agents.contracts import (
    AgentNode,
    AgentRelationship,
    CodeGraphBatchResult,
    attributes_to_dict,
    has_dynamic_identity_markers,
)
from ai_code_intelligence.domain.models import (
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    NodeType,
    RelationshipEvidence,
    RelationshipType,
    ScannedRepository,
)
from ai_code_intelligence.graph.builder import GraphBuilder, structural_graph
from ai_code_intelligence.utils.ids import stable_id

_REFERENCE_LABELS_BY_NODE_TYPE: dict[NodeType, frozenset[str]] = {
    NodeType.CLASS: frozenset({"Class"}),
    NodeType.FUNCTION: frozenset({"Function", "Method", "Middleware"}),
    NodeType.METHOD: frozenset({"Method", "Function", "Constructor"}),
    NodeType.INTERFACE: frozenset({"Interface", "TypeAlias"}),
    NodeType.ENDPOINT: frozenset({"Endpoint"}),
    NodeType.SOCKET_EVENT: frozenset({"SocketEvent"}),
    NodeType.ENVIRONMENT_VARIABLE: frozenset({"EnvironmentVariable"}),
    NodeType.DATABASE_TABLE: frozenset({"DatabaseTable", "DatabaseCollection", "DatabaseIndex"}),
    NodeType.API: frozenset({"API", "OutboundAPI"}),
    NodeType.BUSINESS_RULE: frozenset({"BusinessRule"}),
    NodeType.VARIABLE: frozenset({"Variable", "Property", "Field", "Import", "Function", "Router"}),
    NodeType.CONSTANT: frozenset({"Constant"}),
    NodeType.ENUM: frozenset({"Enum"}),
}
_LEXICAL_REFERENCE_NODE_TYPES = frozenset(
    {
        NodeType.CLASS,
        NodeType.FUNCTION,
        NodeType.METHOD,
        NodeType.INTERFACE,
        NodeType.VARIABLE,
        NodeType.CONSTANT,
        NodeType.ENUM,
    }
)


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Validated repository graph fragment plus evidence rejection metrics."""

    graph: KnowledgeGraph
    evidence_warnings: tuple[str, ...]
    candidate_fact_count: int
    rejected_fact_count: int
    rejection_reason_counts: tuple[tuple[str, int], ...]


class AgentGraphAssembler:
    """Accepts model facts only when exact source locations and excerpts validate."""

    def __init__(self, strict_evidence: bool = True) -> None:
        self._strict = strict_evidence

    def assemble(
        self,
        repository: ScannedRepository,
        documents: tuple[SourceDocument, ...],
        results: list[CodeGraphBatchResult],
        reference_graph: KnowledgeGraph | None = None,
    ) -> AssemblyResult:
        builder = GraphBuilder.from_graph(structural_graph(repository))
        document_index = {document.file_path: document for document in documents}
        reference_index = _ReferenceIndex()
        for node in builder.build().nodes:
            _register_node_aliases(reference_index, node)
        reference_nodes = {node.id: node for node in reference_graph.nodes} if reference_graph else {}
        for node in reference_nodes.values():
            _register_node_aliases(reference_index, node)

        warnings: list[str] = []
        accepted_nodes: list[tuple[AgentNode, CodeGraphBatchResult, GraphNode]] = []
        candidates = sum(len(item.output.nodes) + len(item.output.relationships) for item in results)
        rejected = 0

        for result in results:
            warnings.extend(f"{result.batch_id}: {warning}" for warning in result.output.warnings)
            for candidate in result.output.nodes:
                reason = self._node_rejection(
                    candidate,
                    result,
                    document_index,
                    repository.definition.id,
                )
                if reason:
                    warnings.append(f"Rejected node {candidate.ref}: {reason}")
                    rejected += 1
                    continue
                node = _canonical_node(candidate, repository.definition.id, result.invocation.role.value)
                try:
                    builder.add_node(node)
                except ValueError as error:
                    warnings.append(f"Rejected node {candidate.ref}: {error}")
                    rejected += 1
                    continue
                accepted_nodes.append((candidate, result, node))
                reference_index.register(candidate.ref, node.id)
                _register_node_aliases(reference_index, node)

        self._add_semantic_structure(builder, repository, accepted_nodes)

        for result in results:
            for relationship_candidate in result.output.relationships:
                reason = self._relationship_rejection(relationship_candidate, result, document_index)
                source_id = reference_index.resolve(relationship_candidate.source_ref)
                target_id = reference_index.resolve(relationship_candidate.target_ref)
                if reason is None and source_id is None:
                    reason = f"unresolved or ambiguous source_ref {relationship_candidate.source_ref}"
                if reason is None and target_id is None:
                    reason = f"unresolved or ambiguous target_ref {relationship_candidate.target_ref}"
                if reason:
                    warnings.append(
                        f"Rejected relationship {relationship_candidate.type.value} "
                        f"{relationship_candidate.source_ref} -> "
                        f"{relationship_candidate.target_ref}: {reason}"
                    )
                    rejected += 1
                    continue
                assert source_id is not None and target_id is not None
                for identifier in (source_id, target_id):
                    if not builder.contains_node(identifier) and identifier in reference_nodes:
                        builder.add_node(reference_nodes[identifier])
                relationship = GraphRelationship(
                    id=stable_id(
                        "relationship",
                        relationship_candidate.type.value,
                        source_id,
                        target_id,
                        relationship_candidate.file_path,
                        relationship_candidate.start_line,
                        relationship_candidate.start_column,
                    ),
                    type=relationship_candidate.type,
                    source_id=source_id,
                    target_id=target_id,
                    repository_id=repository.definition.id,
                    evidence=RelationshipEvidence(
                        file_path=relationship_candidate.file_path,
                        start_line=relationship_candidate.start_line,
                        start_column=relationship_candidate.start_column,
                        analyzer=f"bedrock-{result.invocation.role.value}",
                        confidence=relationship_candidate.confidence,
                        resolution=f"bedrock-validated:{relationship_candidate.resolution}",
                        excerpt=relationship_candidate.excerpt,
                    ),
                    metadata={
                        **attributes_to_dict(relationship_candidate.attributes),
                        "evidence_validated": True,
                        "batch_id": result.batch_id,
                    },
                )
                try:
                    builder.add_relationship(relationship)
                except ValueError as error:
                    warnings.append(f"Rejected relationship {relationship.id}: {error}")
                    rejected += 1

        return AssemblyResult(
            graph=builder.build(),
            evidence_warnings=tuple(warnings),
            candidate_fact_count=candidates,
            rejected_fact_count=rejected,
            rejection_reason_counts=_rejection_reason_counts(warnings),
        )

    def _add_semantic_structure(
        self,
        builder: GraphBuilder,
        repository: ScannedRepository,
        accepted: list[tuple[AgentNode, CodeGraphBatchResult, GraphNode]],
    ) -> None:
        repository_id = repository.definition.id
        service_id = stable_id("node", NodeType.SERVICE.value, repository_id)
        added_relationship_ids: set[str] = set()
        for candidate, result, node in accepted:
            file_id = stable_id("node", NodeType.FILE.value, repository_id, candidate.file_path)
            parent = _nearest_lexical_parent(candidate, node, accepted)
            parent_id = parent.id if parent is not None else file_id
            if not builder.contains_node(parent_id) or not builder.contains_node(service_id):
                raise RuntimeError("semantic node references missing structural scaffold")
            evidence = RelationshipEvidence(
                file_path=candidate.file_path,
                start_line=candidate.start_line,
                start_column=1,
                end_line=candidate.end_line,
                analyzer="deterministic-source-location",
                confidence="high",
                resolution="validated lexical containment",
                excerpt=candidate.evidence_excerpt,
            )
            contains_id = stable_id(
                "relationship",
                "CONTAINS",
                parent_id,
                node.id,
                candidate.file_path,
                candidate.start_line,
            )
            if contains_id not in added_relationship_ids:
                builder.add_relationship(
                    GraphRelationship(
                        id=contains_id,
                        type=RelationshipType.CONTAINS,
                        source_id=parent_id,
                        target_id=node.id,
                        repository_id=repository_id,
                        evidence=evidence,
                        metadata={"derived_from_validated_location": True},
                    )
                )
                added_relationship_ids.add(contains_id)
            belongs_to_id = stable_id(
                "relationship",
                "BELONGS_TO",
                node.id,
                service_id,
                candidate.file_path,
                candidate.start_line,
            )
            if belongs_to_id not in added_relationship_ids:
                builder.add_relationship(
                    GraphRelationship(
                        id=belongs_to_id,
                        type=RelationshipType.BELONGS_TO,
                        source_id=node.id,
                        target_id=service_id,
                        repository_id=repository_id,
                        evidence=evidence.model_copy(
                            update={"resolution": "validated source file belongs to service"}
                        ),
                        metadata={
                            "derived_from_validated_location": True,
                            "batch_id": result.batch_id,
                        },
                    )
                )
                added_relationship_ids.add(belongs_to_id)

    def _node_rejection(
        self,
        node: AgentNode,
        result: CodeGraphBatchResult,
        documents: dict[str, SourceDocument],
        repository_id: str,
    ) -> str | None:
        if not _node_reference_matches(node, repository_id):
            return "ref does not match <type>|<qualified_name>"
        if not node.qualified_name.startswith(f"{node.file_path}::"):
            return "qualified_name is not rooted in file_path"
        if not _line_was_supplied(result, node.file_path, node.start_line):
            return "claimed source line was not supplied to this invocation"
        if node.end_line is not None and node.end_line < node.start_line:
            return "end_line precedes start_line"
        if node.end_line is not None and not _line_was_supplied(result, node.file_path, node.end_line):
            return "claimed end line was not supplied to this invocation"
        return self._evidence_rejection(
            documents.get(node.file_path),
            node.start_line,
            node.end_line or node.start_line,
            node.evidence_excerpt,
        )

    def _relationship_rejection(
        self,
        relationship: AgentRelationship,
        result: CodeGraphBatchResult,
        documents: dict[str, SourceDocument],
    ) -> str | None:
        if not _line_was_supplied(result, relationship.file_path, relationship.start_line):
            return "claimed source line was not supplied to this invocation"
        document = documents.get(relationship.file_path)
        line = (
            document.lines[relationship.start_line - 1]
            if document and relationship.start_line <= len(document.lines)
            else None
        )
        if line is None or relationship.start_column > len(line) + 1:
            return "source column is outside claimed line"
        return self._evidence_rejection(
            document,
            relationship.start_line,
            relationship.start_line,
            relationship.excerpt,
        )

    def _evidence_rejection(
        self,
        document: SourceDocument | None,
        start_line: int,
        end_line: int,
        excerpt: str,
    ) -> str | None:
        if document is None:
            return "unknown source file"
        if start_line < 1 or end_line > len(document.lines):
            return "source line is outside file"
        if not self._strict:
            return None
        start = max(0, start_line - 3)
        end = min(len(document.lines), max(end_line + 3, start_line + 12))
        nearby = _normalize_evidence("\n".join(document.lines[start:end]))
        claimed = _normalize_evidence(excerpt)
        if not claimed or claimed not in nearby:
            return "evidence excerpt does not occur near claimed line"
        return None


class _ReferenceIndex:
    def __init__(self) -> None:
        self._values: dict[str, str | None] = {}

    def register(self, reference: str, node_id: str) -> None:
        if reference not in self._values:
            self._values[reference] = node_id
            return
        existing = self._values[reference]
        if existing != node_id:
            self._values[reference] = None

    def resolve(self, reference: str) -> str | None:
        return self._values.get(reference)


def _register_node_aliases(index: _ReferenceIndex, node: GraphNode) -> None:
    index.register(node.id, node.id)
    index.register(f"{node.type.value}|{node.qualified_name}", node.id)


def _line_was_supplied(result: CodeGraphBatchResult, file_path: str, line: int) -> bool:
    return any(path == file_path and start <= line <= end for path, start, end in result.supplied_ranges)


def _normalize_evidence(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _node_reference_matches(node: AgentNode, repository_id: str) -> bool:
    """Accept only lossless formatting aliases for the node's redundant reference field."""

    label, separator, payload = node.ref.partition("|")
    if separator != "|" or label not in _REFERENCE_LABELS_BY_NODE_TYPE[node.type]:
        return False
    for prefix in (f"{repository_id}:file:", f"{repository_id}:"):
        if payload.startswith(prefix):
            payload = payload.removeprefix(prefix)
            break
    if payload == node.qualified_name:
        return True
    if node.type not in _LEXICAL_REFERENCE_NODE_TYPES:
        return False
    root = f"{node.file_path}::"
    if not payload.startswith(root) or not node.qualified_name.startswith(root):
        return False

    def lexical_parts(value: str) -> tuple[str, ...]:
        return tuple(part for part in re.split(r"::|\.", value.removeprefix(root)) if part)

    return lexical_parts(payload) == lexical_parts(node.qualified_name)


def _rejection_reason_counts(warnings: list[str]) -> tuple[tuple[str, int], ...]:
    categories = Counter(
        _rejection_reason_category(warning)
        for warning in warnings
        if warning.startswith("Rejected ")
    )
    return tuple(sorted(categories.items(), key=lambda item: (-item[1], item[0])))


def _rejection_reason_category(warning: str) -> str:
    categories = (
        ("ref does not match", "invalid_node_reference"),
        ("qualified_name is not rooted", "invalid_qualified_name"),
        ("claimed source line was not supplied", "source_line_not_supplied"),
        ("claimed end line was not supplied", "end_line_not_supplied"),
        ("end_line precedes", "invalid_line_range"),
        ("evidence excerpt does not occur", "evidence_excerpt_mismatch"),
        ("unknown source file", "unknown_source_file"),
        ("source line is outside file", "source_line_outside_file"),
        ("source column is outside", "source_column_outside_line"),
        ("unresolved or ambiguous source_ref", "unresolved_source_reference"),
        ("unresolved or ambiguous target_ref", "unresolved_target_reference"),
    )
    for fragment, category in categories:
        if fragment in warning:
            return category
    return "graph_conflict"


def _canonical_node(candidate: AgentNode, repository_id: str, agent_role: str) -> GraphNode:
    metadata = {
        **attributes_to_dict(candidate.attributes),
        "extracted_by": agent_role,
        "source_excerpt": candidate.evidence_excerpt[:500],
    }
    if candidate.type is NodeType.ENVIRONMENT_VARIABLE:
        return GraphNode(
            id=stable_id("node", NodeType.ENVIRONMENT_VARIABLE.value, repository_id, candidate.name),
            type=candidate.type,
            name=candidate.name,
            qualified_name=f"{repository_id}:env:{candidate.name}",
            repository_id=repository_id,
            file_path=candidate.file_path,
            start_line=candidate.start_line,
            end_line=candidate.end_line,
            metadata={**metadata, "value_captured": False},
        )

    if candidate.type in {NodeType.ENDPOINT, NodeType.API}:
        method = metadata.get("method")
        identity = metadata.get("url", metadata.get("path"))
        dynamic = (
            metadata.get("dynamic_url") is True
            or metadata.get("dynamic_path") is True
            or not isinstance(method, str)
            or not isinstance(identity, str)
            or has_dynamic_identity_markers(identity)
        )
        static_identity = False
        if not dynamic and isinstance(method, str) and isinstance(identity, str):
            static_identity = _contains_quoted_literal(
                candidate.evidence_excerpt, identity
            ) and _http_method_evidenced(candidate.evidence_excerpt, method)
        metadata = {
            **metadata,
            "static_http_identity_validated": static_identity,
        }
        if not static_identity:
            metadata = {
                **metadata,
                "dynamic_url": "url" in metadata,
                "dynamic_path": "path" in metadata,
            }

    if candidate.type is NodeType.SOCKET_EVENT:
        identity = str(metadata.get("event_name", candidate.name))
        if metadata.get("dynamic_event_name") is not True and _contains_quoted_literal(
            candidate.evidence_excerpt, identity
        ):
            return GraphNode(
                id=stable_id("node", NodeType.SOCKET_EVENT.value, repository_id, identity),
                type=candidate.type,
                name=identity,
                qualified_name=f"{repository_id}:socket-event:{identity}",
                repository_id=repository_id,
                file_path=candidate.file_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                metadata={**metadata, "static_identity_validated": True},
            )
        metadata = {**metadata, "dynamic_event_name": True, "static_identity_validated": False}

    if candidate.type is NodeType.DATABASE_TABLE:
        identity = str(metadata.get("collection", metadata.get("table", candidate.name)))
        if metadata.get("dynamic_collection") is not True and _contains_quoted_literal(
            candidate.evidence_excerpt, identity
        ):
            database_kind = str(metadata.get("database_kind", "database"))
            return GraphNode(
                id=stable_id("node", NodeType.DATABASE_TABLE.value, repository_id, database_kind, identity),
                type=candidate.type,
                name=identity,
                qualified_name=f"{repository_id}:{database_kind}:{identity}",
                repository_id=repository_id,
                file_path=candidate.file_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                metadata={**metadata, "static_identity_validated": True},
            )
        metadata = {**metadata, "dynamic_collection": True, "static_identity_validated": False}

    return GraphNode(
        id=stable_id("node", candidate.type.value, repository_id, candidate.qualified_name),
        type=candidate.type,
        name=candidate.name,
        qualified_name=candidate.qualified_name,
        repository_id=repository_id,
        file_path=candidate.file_path,
        start_line=candidate.start_line,
        end_line=candidate.end_line,
        metadata=metadata,
    )


def _contains_quoted_literal(excerpt: str, value: str) -> bool:
    return any(f"{quote}{value}{quote}" in excerpt for quote in ('"', "'", "`"))


def _http_method_evidenced(excerpt: str, method: str) -> bool:
    lowered = excerpt.lower()
    normalized = method.lower()
    call_patterns = (f".{normalized}(", f".{normalized} (")
    configured_patterns = (
        f'method: "{normalized}"',
        f"method: '{normalized}'",
        f'method = "{normalized}"',
        f"method = '{normalized}'",
    )
    return any(pattern in lowered for pattern in (*call_patterns, *configured_patterns)) or (
        normalized == "get" and "fetch(" in lowered and "method" not in lowered
    )


def _nearest_lexical_parent(
    candidate: AgentNode,
    node: GraphNode,
    accepted: list[tuple[AgentNode, CodeGraphBatchResult, GraphNode]],
) -> GraphNode | None:
    containers: list[GraphNode] = []
    for other_candidate, _result, other_node in accepted:
        if other_node.id == node.id or other_candidate.file_path != candidate.file_path:
            continue
        if other_node.type not in {
            NodeType.CLASS,
            NodeType.FUNCTION,
            NodeType.METHOD,
            NodeType.INTERFACE,
        }:
            continue
        remainder = candidate.qualified_name.removeprefix(other_candidate.qualified_name)
        if remainder == candidate.qualified_name or not remainder.startswith(("::", ".", "#", "/")):
            continue
        if other_candidate.start_line > candidate.start_line:
            continue
        if (
            other_candidate.end_line is not None
            and candidate.end_line is not None
            and other_candidate.end_line < candidate.end_line
        ):
            continue
        containers.append(other_node)
    return max(containers, key=lambda item: len(item.qualified_name), default=None)
