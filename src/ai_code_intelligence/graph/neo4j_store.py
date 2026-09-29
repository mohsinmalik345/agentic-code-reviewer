from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from neo4j import Driver, GraphDatabase, ManagedTransaction

from ai_code_intelligence.domain.models import (
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    NodeType,
    RelationshipEvidence,
    RelationshipType,
)


@dataclass(frozen=True, slots=True)
class GraphSearchResult:
    """A bounded Neo4j neighborhood and the relevance-ranked nodes that seeded it."""

    graph: KnowledgeGraph
    seed_node_ids: tuple[str, ...]


class Neo4jGraphStore:
    """Stores validated snapshots using allowlisted labels and relationship types only."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str,
        *,
        driver: Driver | None = None,
    ) -> None:
        self._driver = driver or GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def active_snapshot_id(self) -> str | None:
        """Return the single snapshot ID shared by the active managed graph."""

        with self._driver.session(database=self._database) as session:
            rows = session.run(
                "CALL () { MATCH (n:CodeIntelligence) "
                "RETURN collect(DISTINCT n.snapshotId) AS nodeSnapshotIds } "
                "CALL () { MATCH ()-[r]->() WHERE r.codeIntelligence = true "
                "RETURN collect(DISTINCT r.snapshotId) AS relationshipSnapshotIds } "
                "RETURN nodeSnapshotIds, relationshipSnapshotIds"
            ).data()
        if not rows:
            return None
        node_ids = {str(value) for value in rows[0].get("nodeSnapshotIds", []) if value}
        relationship_ids = {str(value) for value in rows[0].get("relationshipSnapshotIds", []) if value}
        combined = node_ids | relationship_ids
        if not combined:
            return None
        if len(combined) != 1 or (relationship_ids and relationship_ids != node_ids):
            raise RuntimeError("Neo4j contains more than one active CodeIntelligence snapshot")
        return next(iter(combined))

    def clear(self) -> None:
        """Delete only the managed CodeIntelligence graph in one transaction."""

        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n:CodeIntelligence) DETACH DELETE n").consume()

    def replace(self, graph: KnowledgeGraph) -> str:
        snapshot_id = str(uuid4())
        nodes_by_type: dict[NodeType, list[dict[str, object]]] = defaultdict(list)
        for node in graph.nodes:
            nodes_by_type[node.type].append({"props": _node_properties(node, snapshot_id)})

        relationships_by_type: dict[RelationshipType, list[dict[str, object]]] = defaultdict(list)
        for relationship in graph.relationships:
            relationships_by_type[relationship.type].append(
                {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "props": _relationship_properties(relationship, snapshot_id),
                }
            )

        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT code_intelligence_id IF NOT EXISTS "
                "FOR (n:CodeIntelligence) REQUIRE n.id IS UNIQUE"
            ).consume()
            session.execute_write(
                _replace_snapshot,
                snapshot_id,
                dict(nodes_by_type),
                dict(relationships_by_type),
            )
        return snapshot_id

    def patch(
        self,
        previous: KnowledgeGraph,
        graph: KnowledgeGraph,
        *,
        expected_snapshot_id: str,
    ) -> str:
        """Apply an exact graph delta while atomically advancing the active snapshot ID."""

        if not expected_snapshot_id:
            raise ValueError("expected_snapshot_id cannot be empty")
        snapshot_id = str(uuid4())
        previous_nodes = {node.id: node for node in previous.nodes}
        current_nodes = {node.id: node for node in graph.nodes}
        previous_relationships = {item.id: item for item in previous.relationships}
        current_relationships = {item.id: item for item in graph.relationships}

        changed_nodes = tuple(
            node
            for node in graph.nodes
            if previous_nodes.get(node.id) != node
        )
        changed_relationships = tuple(
            relationship
            for relationship in graph.relationships
            if previous_relationships.get(relationship.id) != relationship
        )
        deleted_node_ids = tuple(sorted(previous_nodes.keys() - current_nodes.keys()))
        deleted_relationship_ids = tuple(
            sorted(
                (previous_relationships.keys() - current_relationships.keys())
                | {relationship.id for relationship in changed_relationships}
            )
        )
        relabelled_nodes = tuple(
            (node.id, previous_nodes[node.id].type)
            for node in changed_nodes
            if node.id in previous_nodes and previous_nodes[node.id].type is not node.type
        )
        nodes_by_type: dict[NodeType, list[dict[str, object]]] = defaultdict(list)
        for node in changed_nodes:
            nodes_by_type[node.type].append({"props": _node_properties(node, snapshot_id)})
        relationships_by_type: dict[RelationshipType, list[dict[str, object]]] = defaultdict(list)
        for relationship in changed_relationships:
            relationships_by_type[relationship.type].append(
                {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "props": _relationship_properties(relationship, snapshot_id),
                }
            )

        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT code_intelligence_id IF NOT EXISTS "
                "FOR (n:CodeIntelligence) REQUIRE n.id IS UNIQUE"
            ).consume()
            session.execute_write(
                _patch_snapshot,
                expected_snapshot_id,
                snapshot_id,
                deleted_node_ids,
                deleted_relationship_ids,
                relabelled_nodes,
                dict(nodes_by_type),
                dict(relationships_by_type),
            )
        return snapshot_id

    def list_nodes(
        self,
        *,
        repository_id: str | None = None,
        node_type: NodeType | None = None,
        limit: int = 100,
    ) -> tuple[GraphNode, ...]:
        """Return a bounded, optionally filtered node list from the active snapshot."""

        _validate_list_limit(limit)
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                "MATCH (n:CodeIntelligence) "
                "WHERE ($repository_id IS NULL OR n.repositoryId = $repository_id) "
                "AND ($node_type IS NULL OR n.type = $node_type) "
                "RETURN properties(n) AS props ORDER BY n.id LIMIT $limit",
                repository_id=repository_id,
                node_type=node_type.value if node_type else None,
                limit=limit,
            ).data()
        return tuple(_nodes_from_rows(rows))

    def list_relationships(
        self,
        *,
        repository_id: str | None = None,
        relationship_type: RelationshipType | None = None,
        limit: int = 100,
    ) -> tuple[GraphRelationship, ...]:
        """Return a bounded, optionally filtered relationship list from the active snapshot."""

        _validate_list_limit(limit)
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                "MATCH (source:CodeIntelligence)-[r]->(target:CodeIntelligence) "
                "WHERE r.codeIntelligence = true "
                "AND ($repository_id IS NULL OR r.repositoryId = $repository_id) "
                "AND ($relationship_type IS NULL OR r.type = $relationship_type) "
                "RETURN source.id AS sourceId, target.id AS targetId, type(r) AS type, "
                "properties(r) AS props ORDER BY r.id LIMIT $limit",
                repository_id=repository_id,
                relationship_type=relationship_type.value if relationship_type else None,
                limit=limit,
            ).data()
        return tuple(_relationships_from_rows(rows))

    def search_subgraph(
        self,
        terms: tuple[str, ...],
        *,
        repository_id: str | None,
        snapshot_id: str,
        seed_limit: int,
        max_nodes: int,
        max_relationships: int,
    ) -> GraphSearchResult:
        """Find name-matched entities and return their bounded one-hop neighborhood."""

        if not terms:
            return GraphSearchResult(KnowledgeGraph.create([], []), ())
        if not snapshot_id:
            raise ValueError("snapshot_id cannot be empty")
        _validate_search_limit("seed_limit", seed_limit, maximum=50)
        _validate_search_limit("max_nodes", max_nodes, maximum=200)
        _validate_search_limit("max_relationships", max_relationships, maximum=500)
        if max_nodes < seed_limit:
            raise ValueError("max_nodes cannot be smaller than seed_limit")
        query = (
            "MATCH (seed:CodeIntelligence) "
            "WHERE seed.snapshotId = $snapshot_id "
            "AND ($repository_id IS NULL OR seed.repositoryId = $repository_id) "
            "AND any(term IN $terms WHERE "
            "toLower(seed.name) CONTAINS term "
            "OR toLower(seed.qualifiedName) CONTAINS term "
            "OR toLower(coalesce(seed.filePath, '')) CONTAINS term "
            "OR toLower(seed.type) = term) "
            "WITH seed, reduce(score = 0, term IN $terms | score + "
            "CASE WHEN toLower(seed.name) = term THEN 100 ELSE 0 END + "
            "CASE WHEN toLower(seed.name) CONTAINS term THEN 30 ELSE 0 END + "
            "CASE WHEN toLower(seed.qualifiedName) CONTAINS term THEN 10 ELSE 0 END + "
            "CASE WHEN toLower(coalesce(seed.filePath, '')) CONTAINS term THEN 3 ELSE 0 END + "
            "CASE WHEN toLower(seed.type) = term THEN 5 ELSE 0 END) AS score "
            "ORDER BY score DESC, seed.id LIMIT $seed_limit "
            "WITH collect(seed) AS seeds "
            "UNWIND seeds AS seed "
            "OPTIONAL MATCH (seed)-[candidate]-(neighbor:CodeIntelligence) "
            "WHERE candidate.codeIntelligence = true "
            "AND candidate.snapshotId = $snapshot_id "
            "AND neighbor.snapshotId = $snapshot_id "
            "WITH seeds, neighbor ORDER BY neighbor.id "
            "WITH seeds, collect(DISTINCT neighbor) AS neighbors "
            "WITH seeds, seeds + [node IN neighbors WHERE NOT node IN seeds] AS candidates "
            "WITH seeds, candidates[..$max_nodes] AS selected "
            "UNWIND selected AS node "
            "WITH seeds, selected, collect(properties(node)) AS nodeRows "
            "UNWIND selected AS source "
            "OPTIONAL MATCH (source)-[relationship]->(target:CodeIntelligence) "
            "WHERE target IN selected "
            "AND relationship.codeIntelligence = true "
            "AND relationship.snapshotId = $snapshot_id "
            "WITH seeds, nodeRows, source, relationship, target "
            "ORDER BY source.id, relationship.id "
            "WITH seeds, nodeRows, collect(DISTINCT CASE WHEN relationship IS NULL THEN null ELSE {"
            "sourceId: source.id, targetId: target.id, type: type(relationship), "
            "props: properties(relationship)} END) AS relationshipRows "
            "RETURN nodeRows AS nodes, "
            "[row IN relationshipRows WHERE row IS NOT NULL][..$max_relationships] "
            "AS relationships, [seed IN seeds | seed.id] AS seedIds"
        )
        with self._driver.session(database=self._database) as session:
            record = session.run(
                query,
                terms=list(terms),
                repository_id=repository_id,
                snapshot_id=snapshot_id,
                seed_limit=seed_limit,
                max_nodes=max_nodes,
                max_relationships=max_relationships,
            ).single()
        if record is None:
            return GraphSearchResult(KnowledgeGraph.create([], []), ())
        node_rows = [{"props": props} for props in record["nodes"]]
        relationship_rows = list(record["relationships"])
        return GraphSearchResult(
            _graph_from_rows(node_rows, relationship_rows),
            tuple(str(identifier) for identifier in record["seedIds"]),
        )

    def load(self) -> KnowledgeGraph:
        with self._driver.session(database=self._database) as session:
            node_rows = session.run(
                "MATCH (n:CodeIntelligence) RETURN properties(n) AS props ORDER BY n.id"
            ).data()
            relationship_rows = session.run(
                "MATCH (source:CodeIntelligence)-[r]->(target:CodeIntelligence) "
                "WHERE r.codeIntelligence = true "
                "RETURN source.id AS sourceId, target.id AS targetId, type(r) AS type, "
                "properties(r) AS props ORDER BY r.id"
            ).data()
        return _graph_from_rows(node_rows, relationship_rows)

    def impact_subgraph(
        self,
        start_node_ids: tuple[str, ...],
        max_depth: int,
        max_nodes: int,
    ) -> KnowledgeGraph:
        """Load a bounded undirected neighborhood; domain traversal rules run afterward in memory."""

        if not 1 <= max_depth <= 20:
            raise ValueError("max_depth must be between 1 and 20")
        if not start_node_ids:
            return KnowledgeGraph.create([], [])
        query = (
            f"MATCH p=(start:CodeIntelligence)-[*0..{max_depth}]-(node:CodeIntelligence) "
            "WHERE start.id IN $start_ids "
            "WITH collect(DISTINCT node)[..$max_nodes] AS nodes "
            "UNWIND nodes AS n WITH collect(n) AS selected "
            "UNWIND selected AS n "
            "OPTIONAL MATCH (n)-[r]-(other:CodeIntelligence) "
            "WHERE other IN selected AND r.codeIntelligence = true "
            "RETURN [item IN selected | properties(item)] AS nodes, "
            "collect(DISTINCT {sourceId: startNode(r).id, targetId: endNode(r).id, "
            "type: type(r), props: properties(r)}) AS relationships"
        )
        with self._driver.session(database=self._database) as session:
            record = session.run(
                query,
                start_ids=list(start_node_ids),
                max_nodes=max_nodes,
            ).single()
        if record is None:
            return KnowledgeGraph.create([], [])
        node_rows = [{"props": props} for props in record["nodes"]]
        relationship_rows = [
            {
                "sourceId": row["sourceId"],
                "targetId": row["targetId"],
                "type": row["type"],
                "props": row["props"],
            }
            for row in record["relationships"]
            if row.get("props")
        ]
        return _graph_from_rows(node_rows, relationship_rows)


def _node_properties(node: GraphNode, snapshot_id: str) -> dict[str, object]:
    values: dict[str, object] = {
        "id": node.id,
        "type": node.type.value,
        "name": node.name,
        "qualifiedName": node.qualified_name,
        "metadataJson": json.dumps(node.metadata, separators=(",", ":"), default=str),
        "snapshotId": snapshot_id,
    }
    if node.repository_id is not None:
        values["repositoryId"] = node.repository_id
    if node.file_path is not None:
        values["filePath"] = node.file_path
    if node.start_line is not None:
        values["startLine"] = node.start_line
    if node.end_line is not None:
        values["endLine"] = node.end_line
    return values


def _relationship_properties(
    relationship: GraphRelationship,
    snapshot_id: str,
) -> dict[str, object]:
    values: dict[str, object] = {
        "id": relationship.id,
        "type": relationship.type.value,
        "metadataJson": json.dumps(relationship.metadata, separators=(",", ":"), default=str),
        "evidenceJson": relationship.evidence.model_dump_json(),
        "snapshotId": snapshot_id,
        "codeIntelligence": True,
    }
    if relationship.repository_id is not None:
        values["repositoryId"] = relationship.repository_id
    return values


def _replace_snapshot(
    transaction: ManagedTransaction,
    snapshot_id: str,
    nodes_by_type: dict[NodeType, list[dict[str, object]]],
    relationships_by_type: dict[RelationshipType, list[dict[str, object]]],
) -> None:
    """Write and clean a snapshot within one all-or-nothing Neo4j transaction."""

    for node_type, rows in nodes_by_type.items():
        query = (
            f"UNWIND $rows AS row MERGE (n:CodeIntelligence:{node_type.value} {{id: row.props.id}}) "
            "SET n = row.props"
        )
        for batch in _batches(rows, 500):
            transaction.run(query, rows=batch).consume()

    for relationship_type, rows in relationships_by_type.items():
        query = (
            "UNWIND $rows AS row "
            "MATCH (source:CodeIntelligence {id: row.source_id}) "
            "MATCH (target:CodeIntelligence {id: row.target_id}) "
            f"MERGE (source)-[r:{relationship_type.value} {{id: row.props.id}}]->(target) "
            "SET r = row.props"
        )
        for batch in _batches(rows, 500):
            transaction.run(query, rows=batch).consume()

    transaction.run(
        "MATCH ()-[r]->() WHERE r.codeIntelligence = true AND r.snapshotId <> $snapshot_id DELETE r",
        snapshot_id=snapshot_id,
    ).consume()
    transaction.run(
        "MATCH (n:CodeIntelligence) WHERE n.snapshotId <> $snapshot_id DETACH DELETE n",
        snapshot_id=snapshot_id,
    ).consume()


def _patch_snapshot(
    transaction: ManagedTransaction,
    expected_snapshot_id: str,
    snapshot_id: str,
    deleted_node_ids: tuple[str, ...],
    deleted_relationship_ids: tuple[str, ...],
    relabelled_nodes: tuple[tuple[str, NodeType], ...],
    nodes_by_type: dict[NodeType, list[dict[str, object]]],
    relationships_by_type: dict[RelationshipType, list[dict[str, object]]],
) -> None:
    """Validate and patch the active managed graph in one transaction."""

    record = transaction.run(
        "CALL () { MATCH (n:CodeIntelligence) "
        "RETURN collect(DISTINCT n.snapshotId) AS nodeSnapshotIds } "
        "CALL () { MATCH ()-[r]->() WHERE r.codeIntelligence = true "
        "RETURN collect(DISTINCT r.snapshotId) AS relationshipSnapshotIds } "
        "RETURN nodeSnapshotIds, relationshipSnapshotIds"
    ).single()
    raw_node_snapshot_ids: Any = record.get("nodeSnapshotIds", []) if record is not None else []
    raw_relationship_snapshot_ids: Any = (
        record.get("relationshipSnapshotIds", []) if record is not None else []
    )
    node_snapshot_ids = {str(value) for value in raw_node_snapshot_ids if value}
    relationship_snapshot_ids = {
        str(value) for value in raw_relationship_snapshot_ids if value
    }
    combined = node_snapshot_ids | relationship_snapshot_ids
    if combined != {expected_snapshot_id} or (
        relationship_snapshot_ids and relationship_snapshot_ids != node_snapshot_ids
    ):
        raise RuntimeError("Neo4j active snapshot changed before incremental graph publication")

    if deleted_relationship_ids:
        transaction.run(
            "MATCH ()-[r]->() WHERE r.codeIntelligence = true AND r.id IN $ids DELETE r",
            ids=list(deleted_relationship_ids),
        ).consume()
    if deleted_node_ids:
        transaction.run(
            "MATCH (n:CodeIntelligence) WHERE n.id IN $ids DETACH DELETE n",
            ids=list(deleted_node_ids),
        ).consume()
    for node_id, previous_type in relabelled_nodes:
        transaction.run(
            f"MATCH (n:CodeIntelligence {{id: $id}}) REMOVE n:{previous_type.value}",
            id=node_id,
        ).consume()
    for node_type, rows in nodes_by_type.items():
        query = (
            f"UNWIND $rows AS row MERGE (n:CodeIntelligence:{node_type.value} {{id: row.props.id}}) "
            "SET n = row.props"
        )
        for batch in _batches(rows, 500):
            transaction.run(query, rows=batch).consume()
    for relationship_type, rows in relationships_by_type.items():
        query = (
            "UNWIND $rows AS row "
            "MATCH (source:CodeIntelligence {id: row.source_id}) "
            "MATCH (target:CodeIntelligence {id: row.target_id}) "
            f"MERGE (source)-[r:{relationship_type.value} {{id: row.props.id}}]->(target) "
            "SET r = row.props"
        )
        for batch in _batches(rows, 500):
            transaction.run(query, rows=batch).consume()
    transaction.run(
        "MATCH (n:CodeIntelligence) SET n.snapshotId = $snapshot_id",
        snapshot_id=snapshot_id,
    ).consume()
    transaction.run(
        "MATCH ()-[r]->() WHERE r.codeIntelligence = true SET r.snapshotId = $snapshot_id",
        snapshot_id=snapshot_id,
    ).consume()


def _graph_from_rows(
    node_rows: list[dict[str, Any]],
    relationship_rows: list[dict[str, Any]],
) -> KnowledgeGraph:
    return KnowledgeGraph.create(
        _nodes_from_rows(node_rows),
        _relationships_from_rows(relationship_rows),
    )


def _nodes_from_rows(node_rows: list[dict[str, Any]]) -> list[GraphNode]:
    return [
        GraphNode(
            id=props["id"],
            type=NodeType(props["type"]),
            name=props["name"],
            qualified_name=props["qualifiedName"],
            repository_id=props.get("repositoryId"),
            file_path=props.get("filePath"),
            start_line=props.get("startLine"),
            end_line=props.get("endLine"),
            metadata=json.loads(props.get("metadataJson", "{}")),
        )
        for row in node_rows
        if isinstance((props := row.get("props")), dict)
    ]


def _relationships_from_rows(
    relationship_rows: list[dict[str, Any]],
) -> list[GraphRelationship]:
    relationships: list[GraphRelationship] = []
    for row in relationship_rows:
        props = row.get("props")
        if not isinstance(props, dict):
            continue
        relationships.append(
            GraphRelationship(
                id=props["id"],
                type=RelationshipType(row["type"]),
                source_id=row["sourceId"],
                target_id=row["targetId"],
                repository_id=props.get("repositoryId"),
                evidence=RelationshipEvidence.model_validate_json(props["evidenceJson"]),
                metadata=json.loads(props.get("metadataJson", "{}")),
            )
        )
    return relationships


def _validate_list_limit(limit: int) -> None:
    if not 0 <= limit <= 1_000:
        raise ValueError("limit must be between 0 and 1000")


def _validate_search_limit(name: str, value: int, *, maximum: int) -> None:
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _batches(values: list[dict[str, object]], size: int) -> list[list[dict[str, object]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
