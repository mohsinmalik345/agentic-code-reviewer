from __future__ import annotations

from pathlib import PurePosixPath

from ai_code_intelligence.domain.models import (
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    NodeType,
    RelationshipEvidence,
    RelationshipType,
    ScannedRepository,
)
from ai_code_intelligence.utils.ids import stable_id


class GraphBuilder:
    """Deduplicates graph values and prevents conflicting facts for stable identifiers."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._relationships: dict[str, GraphRelationship] = {}

    @classmethod
    def from_graph(cls, graph: KnowledgeGraph) -> GraphBuilder:
        builder = cls()
        for node in graph.nodes:
            builder.add_node(node)
        for relationship in graph.relationships:
            builder.add_relationship(relationship)
        return builder

    def add_node(self, node: GraphNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None or existing == node:
            self._nodes[node.id] = node
            return
        identity = (existing.type, existing.name, existing.qualified_name, existing.repository_id)
        candidate_identity = (node.type, node.name, node.qualified_name, node.repository_id)
        if identity != candidate_identity:
            raise ValueError(f"conflicting graph node id: {node.id}")
        metadata = {**existing.metadata}
        raw_conflicts = metadata.get("conflicting_metadata_keys")
        conflicting_keys = (
            {str(value) for value in raw_conflicts} if isinstance(raw_conflicts, list) else set()
        )
        for key, value in node.metadata.items():
            if key not in metadata:
                metadata[key] = value
            elif metadata[key] != value:
                conflicting_keys.add(key)
        raw_occurrences = metadata.get("occurrence_file_paths")
        prior_occurrences = raw_occurrences if isinstance(raw_occurrences, list) else []
        occurrence_paths = {
            path
            for path in (
                *prior_occurrences,
                existing.file_path,
                node.file_path,
            )
            if isinstance(path, str)
        }
        if len(occurrence_paths) > 1:
            metadata["occurrence_file_paths"] = sorted(occurrence_paths)
        if conflicting_keys:
            metadata["conflicting_metadata_keys"] = sorted(conflicting_keys)
        self._nodes[node.id] = existing.model_copy(
            update={
                "file_path": existing.file_path or node.file_path,
                "start_line": existing.start_line or node.start_line,
                "end_line": existing.end_line or node.end_line,
                "metadata": metadata,
            }
        )

    def node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def add_relationship(self, relationship: GraphRelationship) -> None:
        existing = self._relationships.get(relationship.id)
        if existing is not None and existing != relationship:
            raise ValueError(f"conflicting graph relationship id: {relationship.id}")
        self._relationships[relationship.id] = relationship

    def contains_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def build(self) -> KnowledgeGraph:
        nodes = sorted(self._nodes.values(), key=lambda node: node.id)
        relationships = sorted(self._relationships.values(), key=lambda relationship: relationship.id)
        return KnowledgeGraph.create(nodes, relationships)


def structural_graph(repository: ScannedRepository) -> KnowledgeGraph:
    """Build only inventory facts that can be proven without semantic interpretation."""

    builder = GraphBuilder()
    repository_id = repository.definition.id
    repository_node = GraphNode(
        id=stable_id("node", NodeType.REPOSITORY.value, repository_id),
        type=NodeType.REPOSITORY,
        name=repository.definition.name,
        qualified_name=f"{repository_id}:repository",
        repository_id=repository_id,
        metadata={"root_path": repository.root_path, "description": repository.definition.description},
    )
    service_node = GraphNode(
        id=stable_id("node", NodeType.SERVICE.value, repository_id),
        type=NodeType.SERVICE,
        name=repository.definition.name,
        qualified_name=f"{repository_id}:service",
        repository_id=repository_id,
        metadata={},
    )
    builder.add_node(repository_node)
    builder.add_node(service_node)
    builder.add_relationship(
        _structural_relationship(
            RelationshipType.BELONGS_TO,
            service_node.id,
            repository_node.id,
            repository_id,
            "<repository-inventory>",
            "service is configured for repository",
        )
    )

    directory_ids: dict[str, str] = {}
    for directory in repository.directories:
        node = GraphNode(
            id=stable_id("node", NodeType.DIRECTORY.value, repository_id, directory.relative_path),
            type=NodeType.DIRECTORY,
            name=PurePosixPath(directory.relative_path).name,
            qualified_name=f"{repository_id}:directory:{directory.relative_path}",
            repository_id=repository_id,
            file_path=directory.relative_path,
            metadata={},
        )
        builder.add_node(node)
        directory_ids[directory.relative_path] = node.id
        parent_id = (
            repository_node.id
            if directory.parent_path is None
            else directory_ids.get(directory.parent_path, repository_node.id)
        )
        builder.add_relationship(
            _structural_relationship(
                RelationshipType.CONTAINS,
                parent_id,
                node.id,
                repository_id,
                directory.relative_path,
                "directory hierarchy from local filesystem",
            )
        )

    for file in repository.files:
        node = GraphNode(
            id=stable_id("node", NodeType.FILE.value, repository_id, file.relative_path),
            type=NodeType.FILE,
            name=PurePosixPath(file.relative_path).name,
            qualified_name=f"{repository_id}:file:{file.relative_path}",
            repository_id=repository_id,
            file_path=file.relative_path,
            metadata={
                "kind": file.kind.value,
                "extension": file.extension,
                "size_bytes": file.size_bytes,
                "sha256": file.content_sha256,
            },
        )
        builder.add_node(node)
        parent_path = PurePosixPath(file.relative_path).parent.as_posix()
        parent_id = (
            repository_node.id if parent_path == "." else directory_ids.get(parent_path, repository_node.id)
        )
        builder.add_relationship(
            _structural_relationship(
                RelationshipType.CONTAINS,
                parent_id,
                node.id,
                repository_id,
                file.relative_path,
                "file location from local filesystem",
            )
        )
        builder.add_relationship(
            _structural_relationship(
                RelationshipType.BELONGS_TO,
                node.id,
                service_node.id,
                repository_id,
                file.relative_path,
                "file belongs to configured service",
            )
        )

    for name in repository.environment_variable_names:
        node = GraphNode(
            id=stable_id("node", NodeType.ENVIRONMENT_VARIABLE.value, repository_id, name),
            type=NodeType.ENVIRONMENT_VARIABLE,
            name=name,
            qualified_name=f"{repository_id}:env:{name}",
            repository_id=repository_id,
            metadata={"value_captured": False, "discovered_from_environment_file": True},
        )
        builder.add_node(node)
        builder.add_relationship(
            _structural_relationship(
                RelationshipType.BELONGS_TO,
                node.id,
                service_node.id,
                repository_id,
                "<environment-name-inventory>",
                "environment variable name belongs to configured service",
            )
        )

    manifest = repository.package_manifest
    if manifest:
        for package_name, version in sorted({**manifest.dev_dependencies, **manifest.dependencies}.items()):
            package_node = GraphNode(
                id=stable_id("node", NodeType.API.value, "package", package_name),
                type=NodeType.API,
                name=package_name,
                qualified_name=f"package:{package_name}",
                metadata={"kind": "package"},
            )
            builder.add_node(package_node)
            builder.add_relationship(
                _structural_relationship(
                    RelationshipType.DEPENDS_ON,
                    service_node.id,
                    package_node.id,
                    repository_id,
                    "package.json",
                    "exact package manifest dependency",
                    metadata={"declared_version": version},
                )
            )
    return builder.build()


def merge_graphs(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    builder = GraphBuilder()
    for graph in graphs:
        for node in graph.nodes:
            builder.add_node(node)
        for relationship in graph.relationships:
            builder.add_relationship(relationship)
    return builder.build()


def retain_unaffected_graph(
    graph: KnowledgeGraph,
    changed_repository_ids: frozenset[str],
) -> KnowledgeGraph:
    """Keep reusable repository slices while removing every edge touching changed code.

    Repository-less nodes, such as shared package identities, are retained only when
    an unaffected relationship still references them. This prevents orphaned global
    identities from surviving after the repository that introduced them is rebuilt.
    Cross-repository relationships involving a changed repository are deliberately
    removed so the dependency agent can recompute only those affected links.
    """

    if not changed_repository_ids:
        return graph
    nodes = {node.id: node for node in graph.nodes}
    retained_relationships = [
        relationship
        for relationship in graph.relationships
        if relationship.repository_id not in changed_repository_ids
        and nodes[relationship.source_id].repository_id not in changed_repository_ids
        and nodes[relationship.target_id].repository_id not in changed_repository_ids
    ]
    referenced_ids = {
        identifier
        for relationship in retained_relationships
        for identifier in (relationship.source_id, relationship.target_id)
    }
    retained_nodes = [
        node
        for node in graph.nodes
        if (
            node.repository_id is not None
            and node.repository_id not in changed_repository_ids
        )
        or (node.repository_id is None and node.id in referenced_ids)
    ]
    return KnowledgeGraph.create(retained_nodes, retained_relationships)


def patch_repository_graph(
    previous: KnowledgeGraph,
    replacement: KnowledgeGraph,
    repository_id: str,
    invalidated_paths: frozenset[str],
) -> KnowledgeGraph:
    """Replace one repository's inventory and only invalidated semantic file slices.

    ``replacement`` must contain the repository's complete deterministic inventory,
    but may contain semantic facts only for affected source files. Publication still
    occurs as one complete graph snapshot; this function never mutates Neo4j in place.
    """

    replacement_nodes = {node.id: node for node in replacement.nodes}
    structural_types = {
        NodeType.REPOSITORY,
        NodeType.SERVICE,
        NodeType.DIRECTORY,
        NodeType.FILE,
        NodeType.ENVIRONMENT_VARIABLE,
    }
    removed_node_ids = {
        node.id
        for node in previous.nodes
        if node.repository_id == repository_id
        and (node.type in structural_types or node.file_path in invalidated_paths)
    }

    retained_relationships: list[GraphRelationship] = []
    for relationship in previous.relationships:
        if (
            relationship.repository_id == repository_id
            and relationship.evidence.analyzer == "deterministic-repository-inventory"
        ):
            continue
        if (
            relationship.repository_id == repository_id
            and relationship.evidence.file_path in invalidated_paths
        ):
            continue
        endpoint_was_removed = (
            relationship.source_id in removed_node_ids
            or relationship.target_id in removed_node_ids
        )
        removed_endpoints_were_recreated = (
            relationship.source_id not in removed_node_ids
            or relationship.source_id in replacement_nodes
        ) and (
            relationship.target_id not in removed_node_ids
            or relationship.target_id in replacement_nodes
        )
        if endpoint_was_removed and not removed_endpoints_were_recreated:
            continue
        retained_relationships.append(relationship)

    retained_nodes = [node for node in previous.nodes if node.id not in removed_node_ids]
    builder = GraphBuilder()
    for node in (*retained_nodes, *replacement.nodes):
        builder.add_node(node)
    for relationship in (*retained_relationships, *replacement.relationships):
        builder.add_relationship(relationship)
    merged = builder.build()

    referenced_ids = {
        identifier
        for relationship in merged.relationships
        for identifier in (relationship.source_id, relationship.target_id)
    }
    nodes = [
        node
        for node in merged.nodes
        if node.repository_id is not None or node.id in referenced_ids
    ]
    node_ids = {node.id for node in nodes}
    relationships = [
        relationship
        for relationship in merged.relationships
        if relationship.source_id in node_ids and relationship.target_id in node_ids
    ]
    result = KnowledgeGraph.create(nodes, relationships)
    stale = [
        node.id
        for node in result.nodes
        if node.repository_id == repository_id
        and node.file_path in invalidated_paths
        and node.id not in replacement_nodes
    ]
    if stale:
        raise RuntimeError(
            "incremental graph patch retained invalidated nodes: " + ", ".join(stale[:5])
        )
    return result


def _structural_relationship(
    relationship_type: RelationshipType,
    source_id: str,
    target_id: str,
    repository_id: str,
    file_path: str,
    resolution: str,
    *,
    metadata: dict[str, object] | None = None,
) -> GraphRelationship:
    return GraphRelationship(
        id=stable_id("relationship", relationship_type.value, source_id, target_id, file_path, resolution),
        type=relationship_type,
        source_id=source_id,
        target_id=target_id,
        repository_id=repository_id,
        evidence=RelationshipEvidence(
            file_path=file_path,
            start_line=1,
            start_column=1,
            analyzer="deterministic-repository-inventory",
            confidence="high",
            resolution=resolution,
        ),
        metadata=metadata or {},
    )
