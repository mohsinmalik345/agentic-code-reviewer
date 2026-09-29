from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from ai_code_intelligence.application.models import ScanResult
from ai_code_intelligence.domain.knowledge import (
    KnowledgeDocument,
    KnowledgeManifest,
    StoredDocument,
)
from ai_code_intelligence.domain.models import (
    AgentInvocation,
    FrozenModel,
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    NodeType,
)
from ai_code_intelligence.domain.repositories import (
    IndexingJobStatus,
    RepositoryProvider,
    RepositoryRecord,
    RepositoryStatus,
)
from ai_code_intelligence.knowledge.human import human_readable_document
from ai_code_intelligence.persistence.catalog import IndexingJobStore, RepositoryCatalog

_LOGGER = logging.getLogger(__name__)
_CITATION = re.compile(r"\[(node|relationship):([^\]]*)\]")
_MAX_GRAPH_BYTES = 256 * 1024 * 1024
_MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


class AdoptionDocumentStore(Protocol):
    """Knowledge destination required by the legacy-snapshot adoption workflow."""

    def put(self, document: KnowledgeDocument) -> StoredDocument: ...

    def get(self, uri: str) -> str: ...

    def latest_manifest(self) -> KnowledgeManifest | None: ...

    def publish_manifest(self, manifest: Mapping[str, Any]) -> str: ...

    def delete_manifest(self) -> None: ...


class AdoptionGraphStore(Protocol):
    """Minimal durable graph operations used during adoption."""

    def verify_connectivity(self) -> None: ...

    def load(self) -> KnowledgeGraph: ...

    def active_snapshot_id(self) -> str | None: ...

    def replace(self, graph: KnowledgeGraph) -> str: ...

    def clear(self) -> None: ...

    def close(self) -> None: ...


class AdoptionMetadataStore(Protocol):
    """PostgreSQL audit operations used by the adoption workflow."""

    def initialize(self) -> None: ...

    def record_scan(
        self,
        result: ScanResult,
        invocations: tuple[AgentInvocation, ...],
        documents: tuple[KnowledgeDocument, ...],
        stored: tuple[StoredDocument, ...],
    ) -> None: ...

    def close(self) -> None: ...


class AdoptionGraphSnapshotStore(Protocol):
    """Local durable graph snapshot used by API and deployment reads."""

    @property
    def path(self) -> Path: ...

    def put(self, graph: KnowledgeGraph) -> str: ...

    def get(self) -> KnowledgeGraph: ...

    def exists(self) -> bool: ...

    def delete(self) -> None: ...


class AdoptionPlan(FrozenModel):
    """Validated, read-only plan for adopting an existing portfolio snapshot."""

    scan_run_id: str
    repository_ids: tuple[str, ...]
    graph_node_count: int = Field(ge=0)
    graph_relationship_count: int = Field(ge=0)
    neo4j_snapshot_id: str | None = None
    graph_action: Literal["preserve", "install", "replace"]


class AdoptionResult(AdoptionPlan):
    """Durable locations created by a completed no-Bedrock adoption."""

    knowledge_uris: tuple[str, ...]
    manifest_uri: str


@dataclass(frozen=True, slots=True)
class _PreparedAdoption:
    plan: AdoptionPlan
    graph_path: Path
    graph: KnowledgeGraph
    records: tuple[RepositoryRecord, ...]
    documents: tuple[KnowledgeDocument, ...]
    previous_manifest: KnowledgeManifest | None


class LegacySnapshotAdoptionService:
    """Adopts existing graph and Markdown artifacts without invoking an AI model.

    The stable knowledge manifest is the publication boundary. Versioned document
    objects, the idempotent PostgreSQL audit row, and recoverable catalog metadata
    are prepared first; ``latest.json`` is always the final commit marker.
    """

    def __init__(
        self,
        *,
        catalog: RepositoryCatalog,
        jobs: IndexingJobStore,
        documents: AdoptionDocumentStore,
        graph_snapshots: AdoptionGraphSnapshotStore,
        graph_factory: Callable[[], AdoptionGraphStore],
        metadata_factory: Callable[[], AdoptionMetadataStore],
    ) -> None:
        self._catalog = catalog
        self._jobs = jobs
        self._documents = documents
        self._graph_snapshots = graph_snapshots
        self._graph_factory = graph_factory
        self._metadata_factory = metadata_factory

    def preflight(
        self,
        graph_path: str | Path,
        knowledge_directory: str | Path,
        *,
        replace_neo4j: bool = False,
    ) -> AdoptionPlan:
        """Validate local artifacts and durable state without writing anything."""

        return self._prepare(graph_path, knowledge_directory, replace_neo4j).plan

    def adopt(
        self,
        graph_path: str | Path,
        knowledge_directory: str | Path,
        *,
        replace_neo4j: bool = False,
    ) -> AdoptionResult:
        """Publish a validated legacy snapshot using a recoverable commit sequence."""

        prepared = self._prepare(graph_path, knowledge_directory, replace_neo4j)
        stored = tuple(self._documents.put(document) for document in prepared.documents)
        evidence_count = len(prepared.records) + 1
        evidence_stored = stored[:evidence_count]
        human_stored = stored[evidence_count:]
        if len(human_stored) != evidence_count:
            raise RuntimeError("adoption did not produce a complete reader-edition portfolio")

        graph_store = self._graph_factory()
        previous_local_graph: KnowledgeGraph | None = None
        local_graph_changed = False
        persisted_graph_path = str(self._graph_snapshots.path)
        prior_graph: KnowledgeGraph | None = None
        graph_changed = prepared.plan.graph_action != "preserve"
        snapshot_id = prepared.plan.neo4j_snapshot_id
        catalog_published = False
        manifest_attempted = False
        try:
            if self._graph_snapshots.exists():
                previous_local_graph = self._graph_snapshots.get()
                local_graph_changed = previous_local_graph != prepared.graph
            else:
                local_graph_changed = True
            if local_graph_changed:
                persisted_graph_path = self._graph_snapshots.put(prepared.graph)

            if graph_changed:
                graph_store.verify_connectivity()
                prior_graph = graph_store.load()
                snapshot_id = graph_store.replace(prepared.graph)

            result = ScanResult(
                scan_run_id=prepared.plan.scan_run_id,
                indexing_mode="adopted-existing",
                repository_ids=prepared.plan.repository_ids,
                graph_path=persisted_graph_path,
                statistics=prepared.graph.statistics,
                knowledge_uris=tuple(item.uri for item in evidence_stored),
                embedded_chunk_count=0,
                agent_invocations=(),
                evidence_warnings=(
                    "Imported from a legacy snapshot; local Git commit SHAs were not "
                    "independently established.",
                ),
                neo4j_snapshot_id=snapshot_id,
            )
            self._record_metadata(result, prepared.documents, stored)

            updated_records = _indexed_records(
                prepared.records,
                prepared.graph,
                evidence_stored,
                result.scan_run_id,
            )
            manifest = _manifest(
                updated_records,
                result,
                prepared.graph.generated_at,
                human_stored,
            )
            KnowledgeManifest.model_validate(manifest)

            # The complete catalog update is atomic and recoverable. The stable
            # manifest remains the final commit marker across all publication writes.
            self._catalog.publish_index_metadata(updated_records)
            catalog_published = True
            manifest_attempted = True
            manifest_uri = self._documents.publish_manifest(manifest)
            return AdoptionResult(
                **prepared.plan.model_dump(exclude={"neo4j_snapshot_id"}),
                neo4j_snapshot_id=snapshot_id,
                knowledge_uris=result.knowledge_uris,
                manifest_uri=manifest_uri,
            )
        except Exception as error:
            rollback_errors = self._rollback(
                previous_manifest=prepared.previous_manifest,
                manifest_attempted=manifest_attempted,
                catalog_published=catalog_published,
                original_records=prepared.records,
                graph_store=graph_store,
                graph_changed=graph_changed,
                prior_graph=prior_graph,
                local_graph_changed=local_graph_changed,
                previous_local_graph=previous_local_graph,
            )
            if rollback_errors:
                names = ", ".join(type(item).__name__ for item in rollback_errors)
                raise RuntimeError(
                    f"snapshot adoption failed and rollback was incomplete ({names})"
                ) from error
            raise
        finally:
            graph_store.close()

    def _prepare(
        self,
        graph_path: str | Path,
        knowledge_directory: str | Path,
        replace_neo4j: bool,
    ) -> _PreparedAdoption:
        _reject_active_jobs(self._jobs)
        records = self._catalog.list_repositories(include_disabled=False)
        if not records:
            raise ValueError("at least one enabled repository must be registered")
        repository_ids = tuple(record.id for record in records)

        resolved_graph_path, graph = _load_graph(graph_path)
        _validate_graph_repositories(graph, repository_ids)
        markdown = _load_markdown(knowledge_directory, records, graph)
        scan_run_id = _scan_run_id(graph, markdown, records)
        documents = _knowledge_documents(markdown, records, graph.generated_at, scan_run_id)

        graph_store = self._graph_factory()
        try:
            graph_store.verify_connectivity()
            durable_graph = graph_store.load()
            snapshot_id = graph_store.active_snapshot_id()
        finally:
            graph_store.close()
        graph_action = _graph_action(graph, durable_graph, snapshot_id, replace_neo4j)
        previous_manifest = self._documents.latest_manifest()
        _validate_previous_manifest(previous_manifest, durable_graph, snapshot_id, records)
        return _PreparedAdoption(
            plan=AdoptionPlan(
                scan_run_id=scan_run_id,
                repository_ids=repository_ids,
                graph_node_count=graph.statistics.node_count,
                graph_relationship_count=graph.statistics.relationship_count,
                neo4j_snapshot_id=snapshot_id if graph_action == "preserve" else None,
                graph_action=graph_action,
            ),
            graph_path=resolved_graph_path,
            graph=graph,
            records=records,
            documents=documents,
            previous_manifest=previous_manifest,
        )

    def _record_metadata(
        self,
        result: ScanResult,
        documents: tuple[KnowledgeDocument, ...],
        stored: tuple[StoredDocument, ...],
    ) -> None:
        metadata = self._metadata_factory()
        try:
            metadata.initialize()
            metadata.record_scan(result, (), documents, stored)
        finally:
            metadata.close()

    def _rollback(
        self,
        *,
        previous_manifest: KnowledgeManifest | None,
        manifest_attempted: bool,
        catalog_published: bool,
        original_records: tuple[RepositoryRecord, ...],
        graph_store: AdoptionGraphStore,
        graph_changed: bool,
        prior_graph: KnowledgeGraph | None,
        local_graph_changed: bool,
        previous_local_graph: KnowledgeGraph | None,
    ) -> tuple[Exception, ...]:
        errors: list[Exception] = []
        restored_snapshot_id: str | None = None
        if graph_changed and prior_graph is not None:
            try:
                if prior_graph.nodes:
                    restored_snapshot_id = graph_store.replace(prior_graph)
                else:
                    graph_store.clear()
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore the prior Neo4j graph")
        if local_graph_changed:
            try:
                if previous_local_graph is None:
                    self._graph_snapshots.delete()
                else:
                    self._graph_snapshots.put(previous_local_graph)
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore the prior local graph snapshot")
        if catalog_published:
            try:
                self._catalog.publish_index_metadata(original_records)
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore the prior repository metadata")
        if manifest_attempted:
            try:
                if previous_manifest is None:
                    self._documents.delete_manifest()
                else:
                    self._documents.publish_manifest(
                        _manifest_mapping(
                            previous_manifest,
                            neo4j_snapshot_id=restored_snapshot_id,
                        )
                    )
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore the prior knowledge manifest")
        return tuple(errors)


def _load_graph(path: str | Path) -> tuple[Path, KnowledgeGraph]:
    target = Path(path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise ValueError("legacy graph path must be a regular file")
    size = target.stat().st_size
    if size <= 0 or size > _MAX_GRAPH_BYTES:
        raise ValueError(f"legacy graph must be between 1 and {_MAX_GRAPH_BYTES} bytes")
    graph = KnowledgeGraph.model_validate_json(target.read_bytes())
    if len({node.id for node in graph.nodes}) != len(graph.nodes):
        raise ValueError("legacy graph contains duplicate node IDs")
    if len({relationship.id for relationship in graph.relationships}) != len(graph.relationships):
        raise ValueError("legacy graph contains duplicate relationship IDs")
    computed = KnowledgeGraph.create(list(graph.nodes), list(graph.relationships))
    if computed.statistics != graph.statistics:
        raise ValueError("legacy graph statistics do not match its contents")
    return target, graph


def _validate_graph_repositories(graph: KnowledgeGraph, expected_ids: tuple[str, ...]) -> None:
    expected = set(expected_ids)
    node_ids = {node.repository_id for node in graph.nodes if node.repository_id is not None}
    relationship_ids = {
        relationship.repository_id
        for relationship in graph.relationships
        if relationship.repository_id is not None
    }
    if node_ids != expected:
        raise ValueError(
            "legacy graph repository IDs do not match enabled registrations: "
            f"expected={sorted(expected)}, actual={sorted(node_ids)}"
        )
    unknown_relationship_ids = relationship_ids - expected
    if unknown_relationship_ids:
        raise ValueError(
            f"legacy graph relationships contain unknown repository IDs: {sorted(unknown_relationship_ids)}"
        )
    repository_nodes = [node for node in graph.nodes if node.type is NodeType.REPOSITORY]
    by_repository: dict[str, list[GraphNode]] = {repository_id: [] for repository_id in expected}
    for node in repository_nodes:
        if node.repository_id is None or node.repository_id not in expected:
            raise ValueError("every Repository graph node must belong to an enabled repository")
        by_repository[node.repository_id].append(node)
    invalid = sorted(repository_id for repository_id, nodes in by_repository.items() if len(nodes) != 1)
    if invalid:
        raise ValueError(
            f"legacy graph must contain exactly one Repository node for each registration: {invalid}"
        )


def _load_markdown(
    directory: str | Path,
    records: tuple[RepositoryRecord, ...],
    graph: KnowledgeGraph,
) -> dict[str, str]:
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("legacy knowledge path must be a directory")
    repositories_root = (root / "repositories").resolve(strict=True)
    if repositories_root.parent != root or not repositories_root.is_dir():
        raise ValueError("legacy knowledge directory must contain repositories/")

    expected_names = {f"{record.id}.md" for record in records}
    actual_names = {item.name for item in repositories_root.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            "legacy repository Markdown files do not match enabled registrations: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )

    result: dict[str, str] = {}
    for record in records:
        text = _read_markdown(repositories_root, f"{record.id}.md")
        _validate_citations(text, graph, record.id)
        result[record.id] = text
    central = _read_markdown(root, "central.md")
    _validate_citations(central, graph, None)
    result["central"] = central
    return result


def _read_markdown(root: Path, filename: str) -> str:
    path = (root / filename).resolve(strict=True)
    if path.parent != root or not path.is_file():
        raise ValueError(f"legacy knowledge file escaped its expected directory: {filename}")
    payload = path.read_bytes()
    if not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"legacy knowledge file {filename} must be between 1 and {_MAX_DOCUMENT_BYTES} bytes"
        )
    text = payload.decode("utf-8")
    if not text.strip() or "\x00" in text:
        raise ValueError(f"legacy knowledge file {filename} is empty or contains NUL bytes")
    first_content_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_content_line.startswith("# "):
        raise ValueError(f"legacy knowledge file {filename} must begin with a level-one heading")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _validate_citations(
    markdown: str,
    graph: KnowledgeGraph,
    repository_id: str | None,
) -> None:
    node_by_id = {node.id: node for node in graph.nodes}
    relationship_by_id = {relationship.id: relationship for relationship in graph.relationships}
    citations = _CITATION.findall(markdown)
    if not citations:
        raise ValueError("legacy knowledge document contains no graph citations")

    cited_repository_ids: set[str] = set()
    unknown: list[str] = []
    for kind, identifier in citations:
        values: Mapping[str, GraphNode | GraphRelationship]
        values = node_by_id if kind == "node" else relationship_by_id
        value = values.get(identifier)
        if value is None:
            unknown.append(f"{kind}:{identifier}")
            continue
        if value.repository_id is not None:
            cited_repository_ids.add(value.repository_id)
    if unknown:
        raise ValueError("legacy knowledge document cites unknown graph IDs: " + ", ".join(unknown[:10]))

    expected_ids = {node.repository_id for node in graph.nodes if node.repository_id is not None}
    if repository_id is not None and repository_id not in cited_repository_ids:
        raise ValueError(f"legacy repository document has no graph evidence for {repository_id}")
    if repository_id is None and cited_repository_ids != expected_ids:
        raise ValueError(
            "legacy central document does not cite every enabled repository: "
            f"expected={sorted(expected_ids)}, actual={sorted(cited_repository_ids)}"
        )


def _scan_run_id(
    graph: KnowledgeGraph,
    markdown: Mapping[str, str],
    records: tuple[RepositoryRecord, ...],
) -> str:
    payload = {
        "graph": graph.model_dump(mode="json"),
        "markdown": dict(sorted(markdown.items())),
        "repositories": [{"id": record.id, "commitSha": record.exact_commit} for record in records],
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"adopted-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _knowledge_documents(
    markdown: Mapping[str, str],
    records: tuple[RepositoryRecord, ...],
    generated_at: datetime,
    scan_run_id: str,
) -> tuple[KnowledgeDocument, ...]:
    fingerprint = scan_run_id.removeprefix("adopted-")[:32]
    documents = [
        KnowledgeDocument(
            id=f"adopted-{fingerprint}-repository-{record.id}",
            kind="repository",
            title=_title(markdown[record.id]),
            markdown=markdown[record.id],
            repository_id=record.id,
            generated_at=generated_at,
        )
        for record in records
    ]
    documents.append(
        KnowledgeDocument(
            id=f"adopted-{fingerprint}-central",
            kind="central",
            title=_title(markdown["central"]),
            markdown=markdown["central"],
            generated_at=generated_at,
        )
    )
    evidence = tuple(documents)
    return (*evidence, *(human_readable_document(document) for document in evidence))


def _title(markdown: str) -> str:
    line = next(line.strip() for line in markdown.splitlines() if line.strip())
    return line.removeprefix("# ").strip()


def _graph_action(
    legacy: KnowledgeGraph,
    durable: KnowledgeGraph,
    snapshot_id: str | None,
    replace_neo4j: bool,
) -> Literal["preserve", "install", "replace"]:
    if not durable.nodes:
        if durable.relationships:
            raise RuntimeError("Neo4j contains relationships without CodeIntelligence nodes")
        if not replace_neo4j:
            raise RuntimeError(
                "Neo4j has no active graph; use --replace-neo4j only after reviewing the preflight"
            )
        return "install"
    if _same_graph_content(legacy, durable):
        if snapshot_id is None:
            raise RuntimeError("matching Neo4j graph has no single active snapshot ID")
        return "preserve"
    if not replace_neo4j:
        raise RuntimeError("Neo4j graph differs from the supplied legacy snapshot; refusing to replace it")
    return "replace"


def _same_graph_content(left: KnowledgeGraph, right: KnowledgeGraph) -> bool:
    return {node.id: node for node in left.nodes} == {node.id: node for node in right.nodes} and {
        relationship.id: relationship for relationship in left.relationships
    } == {relationship.id: relationship for relationship in right.relationships}


def _indexed_records(
    records: tuple[RepositoryRecord, ...],
    graph: KnowledgeGraph,
    stored: tuple[StoredDocument, ...],
    scan_run_id: str,
) -> tuple[RepositoryRecord, ...]:
    if len(stored) != len(records) + 1:
        raise RuntimeError("adoption did not store one document per repository and one central document")
    uri_by_repository = {record.id: stored[index].uri for index, record in enumerate(records)}
    now = datetime.now(UTC)
    values: list[RepositoryRecord] = []
    for record in records:
        if record.provider is RepositoryProvider.GITLAB and record.exact_commit is None:
            raise ValueError(
                f"GitLab repository {record.id} requires an exact commit before snapshot adoption"
            )
        node_count = sum(1 for node in graph.nodes if node.repository_id == record.id)
        relationship_count = sum(
            1 for relationship in graph.relationships if relationship.repository_id == record.id
        )
        indexed_at = record.indexed_at if record.last_scan_run_id == scan_run_id else now
        candidate = record.model_copy(
            update={
                "status": RepositoryStatus.INDEXED,
                "knowledge_base_uri": uri_by_repository[record.id],
                "graph_node_count": node_count,
                "graph_relationship_count": relationship_count,
                "last_scan_run_id": scan_run_id,
                "indexed_at": indexed_at,
            }
        )
        values.append(RepositoryRecord.model_validate(candidate.model_dump(mode="python")))
    return tuple(values)


def _manifest(
    records: tuple[RepositoryRecord, ...],
    result: ScanResult,
    generated_at: datetime,
    human_stored: tuple[StoredDocument, ...],
) -> dict[str, Any]:
    if len(human_stored) != len(records) + 1:
        raise RuntimeError("reader-edition manifest requires one document per repository and central")
    human_by_repository = {record.id: human_stored[index].uri for index, record in enumerate(records)}
    return {
        "schemaVersion": 2,
        "scanRunId": result.scan_run_id,
        "generatedAt": generated_at.isoformat(),
        "repositories": [
            {
                "id": record.id,
                "commitSha": record.exact_commit,
                "knowledgeUri": record.knowledge_base_uri,
                "humanKnowledgeUri": human_by_repository[record.id],
            }
            for record in records
        ],
        "centralKnowledgeUri": result.knowledge_uris[-1],
        "humanCentralKnowledgeUri": human_stored[-1].uri,
        "graph": {
            "neo4jSnapshotId": result.neo4j_snapshot_id,
            "statistics": result.statistics.model_dump(mode="json"),
        },
    }


def _manifest_mapping(
    manifest: KnowledgeManifest,
    *,
    neo4j_snapshot_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": manifest.schema_version,
        "scanRunId": manifest.scan_run_id,
        "generatedAt": manifest.generated_at.isoformat(),
        "repositories": [
            {
                "id": repository.id,
                "commitSha": repository.commit_sha,
                "knowledgeUri": repository.knowledge_uri,
                **(
                    {"humanKnowledgeUri": repository.human_knowledge_uri}
                    if repository.human_knowledge_uri is not None
                    else {}
                ),
            }
            for repository in manifest.repositories
        ],
        "centralKnowledgeUri": manifest.central_knowledge_uri,
        **(
            {"humanCentralKnowledgeUri": manifest.human_central_knowledge_uri}
            if manifest.human_central_knowledge_uri is not None
            else {}
        ),
        "graph": {
            "neo4jSnapshotId": neo4j_snapshot_id or manifest.graph.neo4j_snapshot_id,
            "statistics": manifest.graph.statistics.model_dump(mode="json"),
        },
    }


def _validate_previous_manifest(
    manifest: KnowledgeManifest | None,
    durable_graph: KnowledgeGraph,
    snapshot_id: str | None,
    records: tuple[RepositoryRecord, ...],
) -> None:
    if manifest is None:
        return
    expected_ids = {record.id for record in records}
    manifest_ids = {repository.id for repository in manifest.repositories}
    if manifest_ids != expected_ids:
        raise RuntimeError("existing knowledge manifest does not match enabled repository registrations")
    if manifest.graph.neo4j_snapshot_id != snapshot_id:
        raise RuntimeError("existing knowledge manifest does not reference the active Neo4j snapshot")
    if manifest.graph.statistics != durable_graph.statistics:
        raise RuntimeError("existing knowledge manifest graph statistics do not match Neo4j")


def _reject_active_jobs(jobs: IndexingJobStore) -> None:
    active = [
        job.id
        for job in jobs.list_jobs()
        if job.status in {IndexingJobStatus.QUEUED, IndexingJobStatus.RUNNING}
    ]
    if active:
        raise RuntimeError(
            "snapshot adoption requires an idle indexing queue; active jobs: " + ", ".join(active[:10])
        )
