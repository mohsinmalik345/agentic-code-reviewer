# Neo4j graph model

Every node has labels `:CodeIntelligence` and its domain type. Every relationship carries `id`, `repositoryId` where applicable, serialized metadata/evidence, and a snapshot ID.

Publication is a whole-system snapshot replacement executed in one Neo4j write transaction. Snapshot
assembly is commit-aware: unchanged repository slices and unaffected cross-service edges are reused,
changed/new slices are rebuilt, and cross-links touching changed code are recomputed. Neo4j receives
the complete merged graph, so a partial analysis can never erase an unrelated repository. If the
published graph/manifest baseline cannot be validated, the worker falls back to a full scan.

## Node labels

`Service`, `Repository`, `Directory`, `File`, `Class`, `Function`, `Method`, `Interface`, `Endpoint`, `SocketEvent`, `EnvironmentVariable`, `DatabaseTable`, `API`, `BusinessRule`, `Variable`, `Constant`, `Enum`.

## Relationship types

`BELONGS_TO`, `CONTAINS`, `CALLS`, `USES`, `DEPENDS_ON`, `IMPORTS`, `EXPOSES`, `IMPLEMENTS`, `INVOKES`, `READS`, `WRITES`, `EMITS`, `LISTENS`.

## Example Cypher

Services and exposed endpoints:

```cypher
MATCH (service:CodeIntelligence:Service)<-[:BELONGS_TO*1..3]-(entity)
MATCH (entity)-[:EXPOSES]->(endpoint:Endpoint)
RETURN service.name, endpoint.name, endpoint.metadataJson
ORDER BY service.name, endpoint.name;
```

Cross-service REST dependencies:

```cypher
MATCH (outbound:API)-[r:INVOKES]->(endpoint:Endpoint)
WHERE outbound.repositoryId <> endpoint.repositoryId
RETURN outbound.repositoryId, outbound.name, endpoint.repositoryId, endpoint.name, r.evidenceJson;
```

Socket producers and consumers with the same validated event identity:

```cypher
MATCH (producer)-[:EMITS]->(emitted:SocketEvent)
MATCH (consumer)-[:LISTENS]->(listened:SocketEvent)
WHERE emitted.name = listened.name
  AND emitted.repositoryId <> listened.repositoryId
RETURN emitted.name, producer.repositoryId, consumer.repositoryId;
```

Database write blast radius:

```cypher
MATCH path=(changed)-[:CALLS|USES|INVOKES|WRITES*1..8]->(table:DatabaseTable)
WHERE changed.id IN $changedNodeIds
RETURN path;
```

Endpoint callers:

```cypher
MATCH path=(endpoint:Endpoint)<-[:EXPOSES|CALLS|INVOKES*1..8]-(caller)
WHERE endpoint.id = $endpointId
RETURN path;
```

