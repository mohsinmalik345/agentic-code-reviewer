from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from ai_code_intelligence.domain.analysis import GraphPath, ImpactAnalysis, RiskSignal
from ai_code_intelligence.domain.change import CodeChangeAnalysis
from ai_code_intelligence.domain.models import (
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    NodeType,
    RelationshipType,
)


@dataclass(frozen=True, slots=True)
class _State:
    node_id: str
    depth: int
    node_ids: tuple[str, ...]
    relationship_ids: tuple[str, ...]


class ImpactAnalyzer:
    """Propagates change impact through a bounded, relationship-aware graph traversal."""

    def __init__(self, max_depth: int, max_nodes: int = 10_000) -> None:
        self._max_depth = max_depth
        self._max_nodes = max_nodes

    def start_node_ids(self, change: CodeChangeAnalysis, graph: KnowledgeGraph) -> tuple[str, ...]:
        entity_ids = {
            node.id
            for values in (
                change.changed_functions,
                change.changed_classes,
                change.changed_endpoints,
                change.changed_apis,
                change.changed_business_rules,
            )
            for node in values
        }
        paths = {
            path for file in change.changed_files for path in (file.path, file.old_path) if path is not None
        }
        file_ids = {
            node.id
            for node in graph.nodes
            if node.type is NodeType.FILE
            and node.repository_id == change.repository_id
            and node.file_path in paths
        }
        return tuple(sorted(entity_ids | file_ids))

    def analyze(self, change: CodeChangeAnalysis, graph: KnowledgeGraph) -> ImpactAnalysis:
        starts = self.start_node_ids(change, graph)
        node_by_id = {node.id: node for node in graph.nodes}
        adjacency: dict[str, list[GraphRelationship]] = defaultdict(list)
        for relationship in graph.relationships:
            adjacency[relationship.source_id].append(relationship)
            adjacency[relationship.target_id].append(relationship)

        visited = dict.fromkeys(starts, 0)
        reached_relationships: dict[str, GraphRelationship] = {}
        paths: list[GraphPath] = []
        frontier = deque(_State(node_id, 0, (node_id,), ()) for node_id in starts)
        truncated = False

        while frontier:
            current = frontier.popleft()
            current_node = node_by_id.get(current.node_id)
            if (
                current.depth >= self._max_depth
                or current_node is None
                or _terminal(current_node, current.depth)
            ):
                continue
            for relationship in adjacency[current.node_id]:
                if not _transition_allowed(current.node_id, relationship):
                    continue
                next_id = (
                    relationship.target_id
                    if relationship.source_id == current.node_id
                    else relationship.source_id
                )
                reached_relationships[relationship.id] = relationship
                depth = current.depth + 1
                previous = visited.get(next_id)
                if previous is None or depth < previous:
                    visited[next_id] = depth
                    next_state = _State(
                        next_id,
                        depth,
                        (*current.node_ids, next_id),
                        (*current.relationship_ids, relationship.id),
                    )
                    paths.append(
                        GraphPath(
                            node_ids=next_state.node_ids,
                            relationship_ids=next_state.relationship_ids,
                        )
                    )
                    frontier.append(next_state)
                if len(visited) >= self._max_nodes:
                    truncated = True
                    frontier.clear()
                    break

        direct = _nodes_at_depth(1, visited, node_by_id)
        indirect = tuple(
            node_by_id[node_id] for node_id, depth in visited.items() if depth > 1 and node_id in node_by_id
        )
        affected = _unique((*direct, *indirect))
        direct_functions = tuple(node for node in direct if _function(node))
        indirect_functions = tuple(node for node in indirect if _function(node))
        endpoints = tuple(node for node in affected if node.type is NodeType.ENDPOINT)
        rest_calls = tuple(
            node for node in affected if node.type is NodeType.API and node.metadata.get("kind") == "http"
        )
        socket_events = tuple(node for node in affected if node.type is NodeType.SOCKET_EVENT)
        repository_ids = {
            change.repository_id,
            *(node.repository_id for node in affected if node.repository_id),
        }
        services = tuple(
            node
            for node in graph.nodes
            if node.type is NodeType.SERVICE and node.repository_id in repository_ids
        )
        relationships = tuple(reached_relationships.values())
        signals = _risk_signals(
            change, direct, indirect, endpoints, socket_events, services, relationships, truncated
        )
        return ImpactAnalysis(
            directly_affected=direct,
            indirectly_affected=indirect,
            directly_affected_functions=direct_functions,
            indirectly_affected_functions=indirect_functions,
            affected_endpoints=endpoints,
            affected_rest_calls=rest_calls,
            affected_socket_events=socket_events,
            affected_services=services,
            dependency_paths=tuple(paths),
            relationships=relationships,
            deterministic_risk_signals=signals,
            potential_regressions=_regressions(change, affected, signals),
            traversal_truncated=truncated,
        )


def _transition_allowed(current: str, relationship: GraphRelationship) -> bool:
    from_source = relationship.source_id == current
    if relationship.type in {
        RelationshipType.CONTAINS,
        RelationshipType.EXPOSES,
        RelationshipType.IMPORTS,
    }:
        return not from_source
    if relationship.type is RelationshipType.BELONGS_TO:
        return from_source
    if relationship.type is RelationshipType.DEPENDS_ON:
        return not from_source or "via" in relationship.metadata
    return True


def _terminal(node: GraphNode, depth: int) -> bool:
    if depth == 0:
        return False
    return node.type is NodeType.REPOSITORY or (
        node.type is NodeType.API and node.metadata.get("kind") in {"package", "package-operation"}
    )


def _nodes_at_depth(
    depth: int,
    visited: dict[str, int],
    nodes: dict[str, GraphNode],
) -> tuple[GraphNode, ...]:
    return tuple(nodes[node_id] for node_id, value in visited.items() if value == depth and node_id in nodes)


def _risk_signals(
    change: CodeChangeAnalysis,
    direct: tuple[GraphNode, ...],
    indirect: tuple[GraphNode, ...],
    endpoints: tuple[GraphNode, ...],
    socket_events: tuple[GraphNode, ...],
    services: tuple[GraphNode, ...],
    relationships: tuple[GraphRelationship, ...],
    truncated: bool,
) -> tuple[RiskSignal, ...]:
    values: list[RiskSignal] = []
    if any(file.status == "deleted" for file in change.changed_files):
        values.append(_signal("deleted-code", "high", "One or more files were deleted."))
    if change.changed_dependencies:
        values.append(
            _signal(
                "dependency-change",
                "high",
                f"{len(change.changed_dependencies)} package dependency declaration(s) changed.",
            )
        )
    if len(services) > 1:
        values.append(
            _signal(
                "cross-service-impact",
                "high",
                f"The dependency graph reaches {len(services)} services.",
                tuple(node.id for node in services),
            )
        )
    if endpoints:
        values.append(
            _signal(
                "endpoint-impact",
                "medium",
                f"{len(endpoints)} exposed endpoint(s) are in the impact cone.",
                tuple(node.id for node in endpoints),
            )
        )
    if socket_events:
        values.append(
            _signal(
                "socket-impact",
                "medium",
                f"{len(socket_events)} socket event(s) are in the impact cone.",
                tuple(node.id for node in socket_events),
            )
        )
    writes = tuple(rel for rel in relationships if rel.type is RelationshipType.WRITES)
    if writes:
        values.append(
            _signal(
                "database-write-impact",
                "high",
                f"{len(writes)} database write relationship(s) are reachable.",
                tuple(rel.target_id for rel in writes),
            )
        )
    blast_radius = len(direct) + len(indirect)
    if blast_radius > 100:
        values.append(_signal("large-blast-radius", "high", f"Traversal reaches {blast_radius} entities."))
    elif blast_radius > 25:
        values.append(
            _signal("moderate-blast-radius", "medium", f"Traversal reaches {blast_radius} entities.")
        )
    if truncated:
        values.append(
            _signal(
                "traversal-truncated",
                "critical",
                "Impact traversal hit its safety bound; blast radius is incomplete.",
            )
        )
    return tuple(values)


def _regressions(
    change: CodeChangeAnalysis,
    affected: tuple[GraphNode, ...],
    signals: tuple[RiskSignal, ...],
) -> tuple[str, ...]:
    values = [
        *(f"Regression-test endpoint {node.name}." for node in affected if node.type is NodeType.ENDPOINT),
        *(
            f"Regression-test producers and consumers of socket event {node.name}."
            for node in affected
            if node.type is NodeType.SOCKET_EVENT
        ),
        *(
            "Regression-test database write success, validation, and failure paths."
            for signal in signals
            if signal.code == "database-write-impact"
        ),
        *(
            f"Run compatibility tests for {item.package_name} ({item.change})."
            for item in change.changed_dependencies
        ),
    ]
    if not values:
        values.append("Run focused tests for every changed function and its direct callers.")
    return tuple(dict.fromkeys(values))


def _signal(
    code: str,
    severity: str,
    reason: str,
    evidence_node_ids: tuple[str, ...] = (),
) -> RiskSignal:
    return RiskSignal(
        code=code,
        severity=severity,
        reason=reason,
        evidence_node_ids=evidence_node_ids,
    )


def _function(node: GraphNode) -> bool:
    return node.type in {NodeType.FUNCTION, NodeType.METHOD}


def _unique(nodes: tuple[GraphNode, ...]) -> tuple[GraphNode, ...]:
    values: dict[str, GraphNode] = {}
    for node in nodes:
        values[node.id] = node
    return tuple(values.values())
