from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ai_code_intelligence.domain.change import DiffHunk, FileChangeStatus, RepositoryChangeSet
from ai_code_intelligence.domain.models import (
    FileKind,
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    NodeType,
    RelationshipType,
    ScannedRepository,
)

_SEMANTIC_RELATIONSHIPS = frozenset(
    {
        RelationshipType.CALLS,
        RelationshipType.USES,
        RelationshipType.DEPENDS_ON,
        RelationshipType.IMPORTS,
        RelationshipType.EXPOSES,
        RelationshipType.IMPLEMENTS,
        RelationshipType.INVOKES,
        RelationshipType.READS,
        RelationshipType.WRITES,
        RelationshipType.EMITS,
        RelationshipType.LISTENS,
    }
)
_SEMANTIC_NODE_TYPES = frozenset(
    {
        NodeType.CLASS,
        NodeType.FUNCTION,
        NodeType.METHOD,
        NodeType.INTERFACE,
        NodeType.ENDPOINT,
        NodeType.SOCKET_EVENT,
        NodeType.DATABASE_TABLE,
        NodeType.API,
        NodeType.BUSINESS_RULE,
        NodeType.VARIABLE,
        NodeType.CONSTANT,
        NodeType.ENUM,
    }
)
_CENTRAL_NODE_TYPES = frozenset(
    {
        NodeType.SERVICE,
        NodeType.REPOSITORY,
        NodeType.ENDPOINT,
        NodeType.API,
        NodeType.SOCKET_EVENT,
        NodeType.DATABASE_TABLE,
        NodeType.BUSINESS_RULE,
    }
)
_FULL_REINDEX_NAMES = frozenset(
    {
        "package.json",
        "tsconfig.json",
        "tsconfig.base.json",
        "tsconfig.build.json",
    }
)


@dataclass(frozen=True, slots=True)
class RepositoryIncrementalPlan:
    """Deterministic scope for updating one repository at an exact Git commit."""

    repository_id: str
    mode: Literal["full", "selective"]
    change_set: RepositoryChangeSet
    directly_affected_node_ids: frozenset[str]
    dependency_node_ids: frozenset[str]
    affected_source_paths: frozenset[str]
    invalidated_paths: frozenset[str]
    knowledge_sections: frozenset[str]
    reason: str

    @property
    def is_selective(self) -> bool:
        """Return whether existing facts outside the invalidated slice can be retained."""

        return self.mode == "selective"


class IncrementalIndexPlanner:
    """Maps exact Git changes to AST-backed graph entities and their dependency closure."""

    def __init__(
        self,
        *,
        max_dependency_depth: int = 12,
        max_dependency_nodes: int = 10_000,
        max_affected_source_ratio: float = 0.75,
    ) -> None:
        if max_dependency_depth < 1 or max_dependency_nodes < 1:
            raise ValueError("incremental dependency limits must be positive")
        if not 0 < max_affected_source_ratio <= 1:
            raise ValueError("incremental affected-source ratio must be in (0, 1]")
        self._max_depth = max_dependency_depth
        self._max_nodes = max_dependency_nodes
        self._max_ratio = max_affected_source_ratio

    def plan(
        self,
        repository: ScannedRepository,
        previous_graph: KnowledgeGraph,
        change_set: RepositoryChangeSet,
    ) -> RepositoryIncrementalPlan:
        """Produce a safe selective plan, falling back to a complete repository rebuild."""

        repository_id = repository.definition.id
        if change_set.repository_id != repository_id:
            raise ValueError("change set does not belong to the scanned repository")
        source_paths = frozenset(
            file.relative_path for file in repository.files if file.kind is FileKind.SOURCE
        )
        if change_set.requires_full_reindex:
            return self._full(repository_id, change_set, source_paths, change_set.basis.value)
        if any(_requires_full_reindex(path) for path in change_set.affected_paths):
            return self._full(
                repository_id,
                change_set,
                source_paths,
                "repository-wide manifest or TypeScript configuration changed",
            )

        repository_nodes = {
            node.id: node for node in previous_graph.nodes if node.repository_id == repository_id
        }
        directly_affected = _directly_affected_nodes(repository_nodes.values(), change_set)
        closure, exceeded = self._dependency_closure(
            previous_graph,
            repository_id,
            directly_affected,
        )
        if exceeded:
            return self._full(
                repository_id,
                change_set,
                source_paths,
                "dependency closure exceeded configured node or depth limit",
            )

        current_changed_sources = {
            file.current_path
            for file in change_set.files
            if file.current_path is not None and file.current_path in source_paths
        }
        closure_source_paths = {
            path
            for node_id in closure
            if node_id in repository_nodes
            for path in _node_occurrence_paths(repository_nodes[node_id])
            if path in source_paths
        }
        affected_source_paths = frozenset(
            path for path in (*current_changed_sources, *closure_source_paths) if path is not None
        )
        if source_paths and len(affected_source_paths) / len(source_paths) > self._max_ratio:
            return self._full(
                repository_id,
                change_set,
                source_paths,
                "dependency closure covers too much of the repository",
            )

        invalidated_paths = frozenset(
            {
                *change_set.affected_paths,
                *(
                    path
                    for node_id in closure
                    if (node := repository_nodes.get(node_id)) is not None
                    for path in _node_occurrence_paths(node)
                ),
            }
        )
        affected_types = {
            node.type
            for node_id in directly_affected | closure
            if (node := repository_nodes.get(node_id)) is not None
        }
        sections = _knowledge_sections(change_set, affected_types)
        return RepositoryIncrementalPlan(
            repository_id=repository_id,
            mode="selective",
            change_set=change_set,
            directly_affected_node_ids=frozenset(directly_affected),
            dependency_node_ids=frozenset(closure.difference(directly_affected)),
            affected_source_paths=affected_source_paths,
            invalidated_paths=invalidated_paths,
            knowledge_sections=frozenset(sections),
            reason="exact Git diff expanded through validated graph dependencies",
        )

    def _full(
        self,
        repository_id: str,
        change_set: RepositoryChangeSet,
        source_paths: frozenset[str],
        reason: str,
    ) -> RepositoryIncrementalPlan:
        return RepositoryIncrementalPlan(
            repository_id=repository_id,
            mode="full",
            change_set=change_set,
            directly_affected_node_ids=frozenset(),
            dependency_node_ids=frozenset(),
            affected_source_paths=source_paths,
            invalidated_paths=frozenset(change_set.affected_paths),
            knowledge_sections=frozenset(_ALL_REPOSITORY_SECTIONS),
            reason=reason,
        )

    def _dependency_closure(
        self,
        graph: KnowledgeGraph,
        repository_id: str,
        seeds: set[str],
    ) -> tuple[set[str], bool]:
        nodes = {node.id: node for node in graph.nodes}
        adjacency: dict[str, set[str]] = {}
        for relationship in graph.relationships:
            source = nodes[relationship.source_id]
            target = nodes[relationship.target_id]
            include = relationship.type in _SEMANTIC_RELATIONSHIPS
            if relationship.type is RelationshipType.CONTAINS:
                include = (
                    source.type in _SEMANTIC_NODE_TYPES
                    and target.type in _SEMANTIC_NODE_TYPES
                    and source.file_path == target.file_path
                )
            if not include:
                continue
            adjacency.setdefault(source.id, set()).add(target.id)
            adjacency.setdefault(target.id, set()).add(source.id)

        visited = set(seeds)
        queue = deque((seed, 0) for seed in sorted(seeds))
        exceeded_depth = False
        while queue:
            current, depth = queue.popleft()
            neighbors = adjacency.get(current, set())
            if depth >= self._max_depth:
                exceeded_depth = exceeded_depth or any(neighbor not in visited for neighbor in neighbors)
                continue
            for neighbor in sorted(neighbors):
                node = nodes.get(neighbor)
                if node is None or node.repository_id != repository_id or neighbor in visited:
                    continue
                visited.add(neighbor)
                if len(visited) > self._max_nodes:
                    return visited, True
                queue.append((neighbor, depth + 1))
        return visited, exceeded_depth


def refresh_plan_knowledge_sections(
    plan: RepositoryIncrementalPlan,
    graph: KnowledgeGraph,
) -> RepositoryIncrementalPlan:
    """Include entity kinds introduced by the newly analyzed affected files."""

    current_types = {
        node.type
        for node in graph.nodes
        if node.repository_id == plan.repository_id
        and plan.invalidated_paths.intersection(_node_occurrence_paths(node))
    }
    sections = plan.knowledge_sections | frozenset(
        _knowledge_sections(plan.change_set, current_types)
    )
    if sections == plan.knowledge_sections:
        return plan
    return RepositoryIncrementalPlan(
        repository_id=plan.repository_id,
        mode=plan.mode,
        change_set=plan.change_set,
        directly_affected_node_ids=plan.directly_affected_node_ids,
        dependency_node_ids=plan.dependency_node_ids,
        affected_source_paths=plan.affected_source_paths,
        invalidated_paths=plan.invalidated_paths,
        knowledge_sections=sections,
        reason=plan.reason,
    )


def central_architecture_fingerprint(graph: KnowledgeGraph) -> str:
    """Hash only facts represented in the portfolio-wide knowledge document."""

    projection = central_graph_projection(graph)
    payload = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def central_graph_projection(graph: KnowledgeGraph) -> dict[str, object]:
    """Return the canonical graph evidence consumed by the central knowledge agent."""

    node_index = {node.id: node for node in graph.nodes}
    important_nodes = tuple(node for node in graph.nodes if node.type in _CENTRAL_NODE_TYPES)
    important_ids = {node.id for node in important_nodes}
    included_relationships: list[GraphRelationship] = []
    for relationship in graph.relationships:
        if relationship.type in {RelationshipType.BELONGS_TO, RelationshipType.CONTAINS}:
            continue
        source = node_index[relationship.source_id]
        target = node_index[relationship.target_id]
        cross_repository = (
            source.repository_id is not None
            and target.repository_id is not None
            and source.repository_id != target.repository_id
        )
        if (
            relationship.source_id in important_ids
            or relationship.target_id in important_ids
            or cross_repository
        ):
            included_relationships.append(relationship)
    nodes = tuple(sorted(important_nodes, key=lambda node: node.id))
    relationships = tuple(sorted(included_relationships, key=lambda item: item.id))
    return {
        "nodes": [_central_node_dump(node) for node in nodes],
        "relationships": [
            _central_relationship_dump(relationship, node_index)
            for relationship in relationships
        ],
        "statistics": {
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "nodes_by_type": dict(sorted(Counter(node.type.value for node in nodes).items())),
            "relationships_by_type": dict(
                sorted(Counter(item.type.value for item in relationships).items())
            ),
        },
    }


def _directly_affected_nodes(
    nodes: Iterable[GraphNode],
    change_set: RepositoryChangeSet,
) -> set[str]:
    repository_nodes = tuple(nodes)
    direct: set[str] = set()
    for changed_file in change_set.files:
        previous_path = changed_file.previous_path
        current_path = changed_file.current_path
        paths = {path for path in (previous_path, current_path) if path is not None}
        path_nodes = [
            node for node in repository_nodes if paths.intersection(_node_occurrence_paths(node))
        ]
        if changed_file.status in {
            FileChangeStatus.ADDED,
            FileChangeStatus.DELETED,
            FileChangeStatus.RENAMED,
        } or changed_file.binary or not changed_file.hunks:
            direct.update(node.id for node in path_nodes)
            continue
        matched = {
            node.id
            for node in path_nodes
            if node.type in _SEMANTIC_NODE_TYPES
            and (
                node.file_path != previous_path
                or _node_intersects_old_hunks(node, changed_file.hunks)
            )
        }
        direct.update(matched)
        if not matched:
            direct.update(node.id for node in path_nodes if node.type in _SEMANTIC_NODE_TYPES)
    return direct


def _node_intersects_old_hunks(node: GraphNode, hunks: tuple[DiffHunk, ...]) -> bool:
    if node.start_line is None:
        return False
    node_end = node.end_line or node.start_line
    for raw_hunk in hunks:
        old_start = raw_hunk.old_start
        old_lines = raw_hunk.old_lines
        hunk_end = old_start + max(old_lines, 1) - 1
        if node.start_line <= hunk_end and node_end >= old_start:
            return True
    return False


def _node_occurrence_paths(node: GraphNode) -> frozenset[str]:
    values = {node.file_path} if node.file_path is not None else set()
    raw_occurrences = node.metadata.get("occurrence_file_paths")
    if isinstance(raw_occurrences, list):
        values.update(path for path in raw_occurrences if isinstance(path, str))
    return frozenset(values)


def _requires_full_reindex(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return name in _FULL_REINDEX_NAMES or (name.startswith("tsconfig.") and name.endswith(".json"))


def _knowledge_sections(
    change_set: RepositoryChangeSet,
    node_types: set[NodeType],
) -> set[str]:
    sections = {"Folder Structure", "Code Statistics"}
    if any(path.lower().endswith((".md", ".mdx")) for path in change_set.affected_paths):
        sections.update({"Purpose", "Architecture", "Known Risks"})
    if any(".env" in path.rsplit("/", 1)[-1].lower() for path in change_set.affected_paths):
        sections.add("Environment Variables")
    if node_types.intersection(
        {NodeType.CLASS, NodeType.FUNCTION, NodeType.METHOD, NodeType.INTERFACE}
    ):
        sections.update(
            {
                "Architecture",
                "Services",
                "Business Flow",
                "Internal Dependency Tree",
                "Important Functions",
                "Main Business Logic",
                "Known Risks",
            }
        )
    if NodeType.ENDPOINT in node_types:
        sections.update({"Controllers", "API List", "Business Flow", "Known Risks"})
    if NodeType.API in node_types:
        sections.update({"API List", "External Dependencies", "Business Flow", "Known Risks"})
    if NodeType.SOCKET_EVENT in node_types:
        sections.update({"Socket Events", "Business Flow", "Known Risks"})
    if NodeType.DATABASE_TABLE in node_types:
        sections.update({"Database Usage", "Business Flow", "Known Risks"})
    if NodeType.BUSINESS_RULE in node_types:
        sections.update({"Business Flow", "Main Business Logic", "Known Risks"})
    if not node_types and any(
        path.lower().endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"))
        for path in change_set.affected_paths
    ):
        sections.update(
            {
                "Architecture",
                "Services",
                "Important Functions",
                "Main Business Logic",
                "Known Risks",
            }
        )
    return sections


def _node_fingerprint(node: GraphNode) -> dict[str, object]:
    return {
        "id": node.id,
        "type": node.type.value,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "repository_id": node.repository_id,
        "file_path": node.file_path,
        "metadata": node.metadata,
    }


def _central_node_dump(node: GraphNode) -> dict[str, object]:
    omitted_metadata = {
        "batch_id",
        "extracted_by",
        "root_path",
        "sha256",
        "source_excerpt",
    }
    return {
        "id": node.id,
        "type": node.type.value,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "file_path": node.file_path,
        "start_line": node.start_line,
        "metadata": {
            key: value for key, value in node.metadata.items() if key not in omitted_metadata
        },
    }


def _central_relationship_dump(
    relationship: GraphRelationship,
    node_index: dict[str, GraphNode],
) -> dict[str, object]:
    # Kept separate from graph persistence metadata: this is the exact evidence
    # envelope supplied to the central knowledge agent.
    source_id = relationship.source_id
    target_id = relationship.target_id
    result: dict[str, object] = {
        "id": relationship.id,
        "type": relationship.type.value,
        "source_id": source_id,
        "target_id": target_id,
        "file_path": relationship.evidence.file_path,
        "start_line": relationship.evidence.start_line,
    }
    for key, identifier in (("source", source_id), ("target", target_id)):
        node = node_index.get(identifier)
        if node is not None:
            result[key] = {
                "type": node.type.value,
                "name": node.name,
                "repository_id": node.repository_id,
            }
    return result


_ALL_REPOSITORY_SECTIONS = (
    "Purpose",
    "Architecture",
    "Folder Structure",
    "Controllers",
    "Services",
    "Business Flow",
    "API List",
    "Environment Variables",
    "Database Usage",
    "External Dependencies",
    "Socket Events",
    "Internal Dependency Tree",
    "Important Functions",
    "Main Business Logic",
    "Known Risks",
    "Code Statistics",
)
