from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ai_code_intelligence.application.adoption import AdoptionResult
from ai_code_intelligence.application.models import ScanResult
from ai_code_intelligence.domain.knowledge import (
    KnowledgeDocument,
    KnowledgeDocumentRecord,
    KnowledgeManifest,
    StoredDocument,
)
from ai_code_intelligence.domain.models import AgentInvocation, FrozenModel, KnowledgeGraph
from ai_code_intelligence.domain.repositories import (
    IndexingJobStatus,
    RepositoryRecord,
    RepositoryStatus,
)
from ai_code_intelligence.graph.builder import retain_unaffected_graph
from ai_code_intelligence.persistence.catalog import IndexingJobStore, RepositoryCatalog
from ai_code_intelligence.utils.ids import stable_id

_LOGGER = logging.getLogger(__name__)


class SeparationDocumentStore(Protocol):
    """Knowledge operations required to move one repository out of a portfolio."""

    def put(self, document: KnowledgeDocument) -> StoredDocument: ...

    def get(self, uri: str) -> str: ...

    def latest_manifest(self) -> KnowledgeManifest | None: ...

    def publish_manifest(self, manifest: Mapping[str, Any]) -> str: ...


class SeparationGraphStore(Protocol):
    """Atomic graph operations required by portfolio separation."""

    def verify_connectivity(self) -> None: ...

    def load(self) -> KnowledgeGraph: ...

    def active_snapshot_id(self) -> str | None: ...

    def patch(
        self,
        previous: KnowledgeGraph,
        graph: KnowledgeGraph,
        *,
        expected_snapshot_id: str,
    ) -> str: ...

    def close(self) -> None: ...


class SeparationGraphSnapshotStore(Protocol):
    """Local active-graph pointer updated with the durable Neo4j publication."""

    @property
    def path(self) -> Path: ...

    def put(self, graph: KnowledgeGraph) -> str: ...


class SeparationMetadataStore(Protocol):
    """Historical and audit operations used by portfolio separation."""

    def initialize(self) -> None: ...

    def get_scan(self, scan_run_id: str) -> ScanResult | None: ...

    def knowledge_documents_for_scan(
        self,
        scan_run_id: str,
    ) -> tuple[KnowledgeDocumentRecord, ...]: ...

    def record_scan(
        self,
        result: ScanResult,
        invocations: tuple[AgentInvocation, ...],
        documents: tuple[KnowledgeDocument, ...],
        stored: tuple[StoredDocument, ...],
    ) -> None: ...

    def delete_scan(self, scan_run_id: str) -> bool: ...

    def close(self) -> None: ...


class InternalAdoption(Protocol):
    """No-model adoption boundary for the destination internal index."""

    def adopt(
        self,
        graph_path: str | Path,
        knowledge_directory: str | Path,
        *,
        replace_neo4j: bool = False,
    ) -> AdoptionResult: ...


class EmbeddingScopeRefresher(Protocol):
    """Post-publication vector cleanup for a retired repository scope."""

    def replace_central_and_remove(
        self,
        central: KnowledgeDocument,
        repository_ids: frozenset[str],
    ) -> int: ...


class RepositorySeparationResult(FrozenModel):
    """Durable publication locations produced without re-running code-analysis agents."""

    repository_id: str
    public_scan_run_id: str
    public_manifest_uri: str
    public_graph_snapshot_id: str
    public_node_count: int
    public_relationship_count: int
    internal_scan_run_id: str
    internal_manifest_uri: str
    internal_graph_snapshot_id: str | None
    internal_node_count: int
    internal_relationship_count: int
    refreshed_embedding_chunks: int = 0
    warnings: tuple[str, ...] = ()


class RepositorySeparationService:
    """Moves one published repository into a distinct internal-only runtime.

    Existing graph and knowledge evidence are reused. The public portfolio is
    restored from an exact historical scan, while the latest manifest remains the
    commit marker. A failed pre-commit public update restores Neo4j, the local graph,
    catalog metadata, stable document aliases, and the prior manifest.
    """

    def __init__(
        self,
        *,
        catalog: RepositoryCatalog,
        jobs: IndexingJobStore,
        documents: SeparationDocumentStore,
        graph_snapshots: SeparationGraphSnapshotStore,
        graph_factory: Callable[[], SeparationGraphStore],
        metadata_factory: Callable[[], SeparationMetadataStore],
        embedding_refresher: EmbeddingScopeRefresher | None = None,
    ) -> None:
        self._catalog = catalog
        self._jobs = jobs
        self._documents = documents
        self._graph_snapshots = graph_snapshots
        self._graph_factory = graph_factory
        self._metadata_factory = metadata_factory
        self._embedding_refresher = embedding_refresher

    def separate(
        self,
        repository_id: str,
        restore_scan_run_id: str,
        *,
        staging_directory: str | Path,
        initialize_internal: Callable[[], None],
        internal_adoption: InternalAdoption,
    ) -> RepositorySeparationResult:
        """Publish the internal index first, then remove it from the public snapshot."""

        _reject_active_jobs(self._jobs)
        original_records = self._catalog.list_repositories()
        record_by_id = {record.id: record for record in original_records}
        retired_record = record_by_id.get(repository_id)
        if retired_record is None:
            raise ValueError(f"repository is not registered: {repository_id}")
        if retired_record.status is RepositoryStatus.DISABLED:
            raise ValueError(f"repository is already separated: {repository_id}")

        previous_manifest = self._documents.latest_manifest()
        if previous_manifest is None:
            raise RuntimeError("the public knowledge portfolio has not been published")
        manifest_by_id = {item.id: item for item in previous_manifest.repositories}
        retired_manifest = manifest_by_id.get(repository_id)
        if retired_manifest is None:
            raise ValueError(f"repository is not present in the active publication: {repository_id}")
        remaining_entries = tuple(
            item for item in previous_manifest.repositories if item.id != repository_id
        )
        if not remaining_entries:
            raise ValueError("separation cannot leave the public portfolio empty")
        remaining_ids = tuple(sorted(item.id for item in remaining_entries))
        remaining_records = tuple(record_by_id[item] for item in remaining_ids)

        metadata = self._metadata_factory()
        graph_store = self._graph_factory()
        try:
            metadata.initialize()
            restored_scan = metadata.get_scan(restore_scan_run_id)
            if restored_scan is None:
                raise ValueError(f"restore scan was not found: {restore_scan_run_id}")
            if tuple(sorted(restored_scan.repository_ids)) != remaining_ids:
                raise ValueError(
                    "restore scan repository IDs do not match the remaining public portfolio"
                )
            historical_pointers = metadata.knowledge_documents_for_scan(restore_scan_run_id)
            historical_content = _validated_historical_content(
                historical_pointers,
                remaining_ids,
                self._documents,
            )

            graph_store.verify_connectivity()
            previous_graph = graph_store.load()
            previous_snapshot_id = graph_store.active_snapshot_id()
            if previous_snapshot_id is None:
                raise RuntimeError("the active public graph has no snapshot ID")
            if previous_manifest.graph.neo4j_snapshot_id != previous_snapshot_id:
                raise RuntimeError("the public manifest does not reference active Neo4j")
            if previous_manifest.graph.statistics != previous_graph.statistics:
                raise RuntimeError("the public manifest statistics do not match active Neo4j")

            public_graph = retain_unaffected_graph(previous_graph, frozenset((repository_id,)))
            if public_graph.statistics != restored_scan.statistics:
                raise RuntimeError(
                    "removing the internal repository does not reproduce the restore scan graph"
                )
            internal_graph = retain_unaffected_graph(previous_graph, frozenset(remaining_ids))
            internal_repository_ids = {
                node.repository_id
                for node in internal_graph.nodes
                if node.repository_id is not None
            }
            if internal_repository_ids != {repository_id}:
                raise RuntimeError("internal graph extraction crossed repository boundaries")

            current_alias_documents = _manifest_documents(previous_manifest, self._documents)
            staging_root = _prepare_internal_staging(
                staging_directory,
                repository_id,
                internal_graph,
                self._documents.get(retired_manifest.knowledge_uri),
            )
            initialize_internal()
            internal = internal_adoption.adopt(
                staging_root / "graph.json",
                staging_root / "knowledge",
                replace_neo4j=True,
            )

            scan_run_id = _separation_scan_id(
                repository_id,
                previous_manifest.scan_run_id,
                restore_scan_run_id,
                public_graph,
            )
            restored_documents = _restored_documents(
                historical_content,
                remaining_ids,
                scan_run_id,
                public_graph.generated_at,
            )
            stored_documents: tuple[StoredDocument, ...] = ()
            metadata_recorded = False
            graph_patched = False
            catalog_published = False
            manifest_published = False
            public_snapshot_id: str | None = None
            try:
                stored_documents = tuple(
                    self._documents.put(document) for document in restored_documents
                )
                evidence_count = len(remaining_ids) + 1
                evidence_stored = stored_documents[:evidence_count]
                human_stored = stored_documents[evidence_count:]
                if len(human_stored) != evidence_count:
                    raise RuntimeError("restored portfolio is missing reader-edition documents")

                public_snapshot_id = graph_store.patch(
                    previous_graph,
                    public_graph,
                    expected_snapshot_id=previous_snapshot_id,
                )
                graph_patched = True
                graph_path = self._graph_snapshots.put(public_graph)
                result = ScanResult(
                    scan_run_id=scan_run_id,
                    indexing_mode="portfolio-separation",
                    repository_ids=remaining_ids,
                    graph_path=graph_path,
                    statistics=public_graph.statistics,
                    knowledge_uris=tuple(item.uri for item in evidence_stored),
                    embedded_chunk_count=0,
                    agent_invocations=(),
                    evidence_warnings=(
                        "Restored from a validated historical publication; no analysis model was invoked.",
                    ),
                    neo4j_snapshot_id=public_snapshot_id,
                )
                metadata.record_scan(result, (), restored_documents, stored_documents)
                metadata_recorded = True

                updated_records = _public_records(
                    remaining_records,
                    public_graph,
                    evidence_stored,
                    scan_run_id,
                )
                disabled_record = RepositoryRecord.model_validate(
                    retired_record.model_copy(
                        update={
                            "status": RepositoryStatus.DISABLED,
                            "knowledge_base_uri": None,
                            "graph_node_count": 0,
                            "graph_relationship_count": 0,
                            "last_scan_run_id": None,
                            "indexed_at": None,
                            "updated_at": datetime.now(UTC),
                        }
                    ).model_dump(mode="python")
                )
                self._catalog.upsert_repositories((*updated_records, disabled_record))
                catalog_published = True

                manifest = _public_manifest(
                    updated_records,
                    result,
                    human_stored,
                    public_graph,
                )
                public_manifest_uri = self._documents.publish_manifest(manifest)
                manifest_published = True
            except Exception as error:
                rollback_errors = self._rollback_publication(
                    previous_manifest=previous_manifest,
                    previous_graph=previous_graph,
                    previous_snapshot_id=previous_snapshot_id,
                    public_graph=public_graph,
                    public_snapshot_id=public_snapshot_id,
                    original_records=tuple(
                        record_by_id[item] for item in (*remaining_ids, repository_id)
                    ),
                    current_alias_documents=current_alias_documents,
                    scan_run_id=scan_run_id,
                    metadata=metadata,
                    graph_store=graph_store,
                    metadata_recorded=metadata_recorded,
                    graph_patched=graph_patched,
                    catalog_published=catalog_published,
                    manifest_published=manifest_published,
                    documents_written=bool(stored_documents),
                )
                if rollback_errors:
                    names = ", ".join(type(item).__name__ for item in rollback_errors)
                    raise RuntimeError(
                        f"repository separation failed and rollback was incomplete ({names})"
                    ) from error
                raise

            assert public_snapshot_id is not None
            refreshed_chunks = 0
            warnings: list[str] = []
            if self._embedding_refresher is not None:
                central_document = next(
                    item for item in restored_documents if item.kind == "central"
                )
                try:
                    refreshed_chunks = self._embedding_refresher.replace_central_and_remove(
                        central_document,
                        frozenset((repository_id,)),
                    )
                except Exception as error:
                    _LOGGER.exception(
                        "Public portfolio was separated, but embedding cleanup must be retried",
                        extra={"fields": {"repository_id": repository_id}},
                    )
                    warnings.append(
                        "The graph and knowledge publication are correct, but stale embedding "
                        f"scopes remain because {type(error).__name__} interrupted vector cleanup."
                    )

            return RepositorySeparationResult(
                repository_id=repository_id,
                public_scan_run_id=scan_run_id,
                public_manifest_uri=public_manifest_uri,
                public_graph_snapshot_id=public_snapshot_id,
                public_node_count=public_graph.statistics.node_count,
                public_relationship_count=public_graph.statistics.relationship_count,
                internal_scan_run_id=internal.scan_run_id,
                internal_manifest_uri=internal.manifest_uri,
                internal_graph_snapshot_id=internal.neo4j_snapshot_id,
                internal_node_count=internal.graph_node_count,
                internal_relationship_count=internal.graph_relationship_count,
                refreshed_embedding_chunks=refreshed_chunks,
                warnings=tuple(warnings),
            )
        finally:
            graph_store.close()
            metadata.close()

    def _rollback_publication(
        self,
        *,
        previous_manifest: KnowledgeManifest,
        previous_graph: KnowledgeGraph,
        previous_snapshot_id: str,
        public_graph: KnowledgeGraph,
        public_snapshot_id: str | None,
        original_records: tuple[RepositoryRecord, ...],
        current_alias_documents: tuple[KnowledgeDocument, ...],
        scan_run_id: str,
        metadata: SeparationMetadataStore,
        graph_store: SeparationGraphStore,
        metadata_recorded: bool,
        graph_patched: bool,
        catalog_published: bool,
        manifest_published: bool,
        documents_written: bool,
    ) -> tuple[Exception, ...]:
        errors: list[Exception] = []
        restored_snapshot_id = previous_snapshot_id
        if graph_patched and public_snapshot_id is not None:
            try:
                restored_snapshot_id = graph_store.patch(
                    public_graph,
                    previous_graph,
                    expected_snapshot_id=public_snapshot_id,
                )
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore public Neo4j after separation failure")
        if graph_patched:
            try:
                self._graph_snapshots.put(previous_graph)
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore the public local graph snapshot")
        if catalog_published:
            try:
                self._catalog.upsert_repositories(original_records)
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore public repository metadata")
        if metadata_recorded:
            try:
                metadata.delete_scan(scan_run_id)
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to remove the rolled-back scan audit")
        if documents_written:
            try:
                for document in current_alias_documents:
                    self._documents.put(document)
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore stable public document aliases")
        if graph_patched or catalog_published or manifest_published:
            try:
                self._documents.publish_manifest(
                    _manifest_mapping(
                        previous_manifest,
                        neo4j_snapshot_id=restored_snapshot_id,
                    )
                )
            except Exception as error:
                errors.append(error)
                _LOGGER.exception("Unable to restore the prior public manifest")
        return tuple(errors)


def _validated_historical_content(
    pointers: tuple[KnowledgeDocumentRecord, ...],
    repository_ids: tuple[str, ...],
    documents: SeparationDocumentStore,
) -> dict[tuple[str, str | None], str]:
    expected = {
        *(("repository", repository_id) for repository_id in repository_ids),
        ("central", None),
        *(("repository-human", repository_id) for repository_id in repository_ids),
        ("central-human", None),
    }
    by_key = {(pointer.kind, pointer.repository_id): pointer for pointer in pointers}
    if set(by_key) != expected or len(by_key) != len(pointers):
        raise RuntimeError("restore scan does not contain one complete evidence and reader portfolio")
    content: dict[tuple[str, str | None], str] = {}
    for key, pointer in by_key.items():
        markdown = documents.get(pointer.uri)
        payload = markdown.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != pointer.content_sha256:
            raise RuntimeError(f"restore document hash does not match its audit: {pointer.document_id}")
        if len(payload) != pointer.size_bytes:
            raise RuntimeError(f"restore document size does not match its audit: {pointer.document_id}")
        _title(markdown)
        content[key] = markdown
    return content


def _restored_documents(
    content: Mapping[tuple[str, str | None], str],
    repository_ids: tuple[str, ...],
    scan_run_id: str,
    generated_at: datetime,
) -> tuple[KnowledgeDocument, ...]:
    keys = (
        *(("repository", repository_id) for repository_id in repository_ids),
        ("central", None),
        *(("repository-human", repository_id) for repository_id in repository_ids),
        ("central-human", None),
    )
    return tuple(
        KnowledgeDocument(
            id=stable_id("knowledge", scan_run_id, kind, repository_id or "central", markdown),
            kind=kind,
            title=_title(markdown),
            markdown=markdown,
            repository_id=repository_id,
            generated_at=generated_at,
        )
        for kind, repository_id in keys
        for markdown in (content[(kind, repository_id)],)
    )


def _manifest_documents(
    manifest: KnowledgeManifest,
    store: SeparationDocumentStore,
) -> tuple[KnowledgeDocument, ...]:
    values: list[KnowledgeDocument] = []
    for repository in manifest.repositories:
        markdown = store.get(repository.knowledge_uri)
        values.append(
            KnowledgeDocument(
                id=stable_id("knowledge", "rollback", repository.id, "evidence", markdown),
                kind="repository",
                title=_title(markdown),
                markdown=markdown,
                repository_id=repository.id,
                generated_at=manifest.generated_at,
            )
        )
    central = store.get(manifest.central_knowledge_uri)
    values.append(
        KnowledgeDocument(
            id=stable_id("knowledge", "rollback", "central", "evidence", central),
            kind="central",
            title=_title(central),
            markdown=central,
            generated_at=manifest.generated_at,
        )
    )
    for repository in manifest.repositories:
        if repository.human_knowledge_uri is None:
            continue
        markdown = store.get(repository.human_knowledge_uri)
        values.append(
            KnowledgeDocument(
                id=stable_id("knowledge", "rollback", repository.id, "human", markdown),
                kind="repository-human",
                title=_title(markdown),
                markdown=markdown,
                repository_id=repository.id,
                generated_at=manifest.generated_at,
            )
        )
    if manifest.human_central_knowledge_uri is not None:
        central_human = store.get(manifest.human_central_knowledge_uri)
        values.append(
            KnowledgeDocument(
                id=stable_id("knowledge", "rollback", "central", "human", central_human),
                kind="central-human",
                title=_title(central_human),
                markdown=central_human,
                generated_at=manifest.generated_at,
            )
        )
    return tuple(values)


def _prepare_internal_staging(
    root: str | Path,
    repository_id: str,
    graph: KnowledgeGraph,
    repository_markdown: str,
) -> Path:
    target = Path(root).expanduser().resolve()
    knowledge = target / "knowledge"
    repositories = knowledge / "repositories"
    repositories.mkdir(parents=True, exist_ok=True)
    _atomic_write(target / "graph.json", graph.model_dump_json(indent=2))
    _atomic_write(repositories / f"{repository_id}.md", repository_markdown)
    _atomic_write(knowledge / "central.md", _internal_central(repository_markdown))
    return target


def _internal_central(repository_markdown: str) -> str:
    lines = repository_markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    heading_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("# ")),
        None,
    )
    if heading_index is None:
        raise ValueError("internal repository knowledge must contain a level-one heading")
    lines[heading_index] = "# AI Code Intelligence Internal Knowledge Base"
    lines[heading_index + 1 : heading_index + 1] = [
        "",
        "This private portfolio contains only the code-intelligence platform and is not part "
        "of the Beelinks CRM architecture publication.",
    ]
    return "\n".join(lines).strip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    resolved_parent = path.parent.resolve()
    resolved_parent.mkdir(parents=True, exist_ok=True)
    target = path.resolve()
    if target.parent != resolved_parent:
        raise ValueError("staging path escaped its expected directory")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(target)


def _public_records(
    records: tuple[RepositoryRecord, ...],
    graph: KnowledgeGraph,
    stored: tuple[StoredDocument, ...],
    scan_run_id: str,
) -> tuple[RepositoryRecord, ...]:
    if len(records) + 1 != len(stored):
        raise RuntimeError("public restore requires one evidence document per repository and central")
    now = datetime.now(UTC)
    values: list[RepositoryRecord] = []
    for index, record in enumerate(records):
        values.append(
            RepositoryRecord.model_validate(
                record.model_copy(
                    update={
                        "status": RepositoryStatus.INDEXED,
                        "knowledge_base_uri": stored[index].uri,
                        "graph_node_count": sum(
                            1 for node in graph.nodes if node.repository_id == record.id
                        ),
                        "graph_relationship_count": sum(
                            1
                            for relationship in graph.relationships
                            if relationship.repository_id == record.id
                        ),
                        "last_scan_run_id": scan_run_id,
                        "indexed_at": record.indexed_at or now,
                        "updated_at": now,
                    }
                ).model_dump(mode="python")
            )
        )
    return tuple(values)


def _public_manifest(
    records: tuple[RepositoryRecord, ...],
    result: ScanResult,
    human_stored: tuple[StoredDocument, ...],
    graph: KnowledgeGraph,
) -> dict[str, Any]:
    if len(human_stored) != len(records) + 1:
        raise RuntimeError("public manifest requires complete reader-edition documents")
    human_by_id = {
        record.id: human_stored[index].uri for index, record in enumerate(records)
    }
    return {
        "schemaVersion": 2,
        "scanRunId": result.scan_run_id,
        "generatedAt": graph.generated_at.isoformat(),
        "repositories": [
            {
                "id": record.id,
                "commitSha": record.exact_commit,
                "knowledgeUri": record.knowledge_base_uri,
                "humanKnowledgeUri": human_by_id[record.id],
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
    neo4j_snapshot_id: str,
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
            "neo4jSnapshotId": neo4j_snapshot_id,
            "statistics": manifest.graph.statistics.model_dump(mode="json"),
        },
    }


def _separation_scan_id(
    repository_id: str,
    previous_scan_run_id: str,
    restore_scan_run_id: str,
    graph: KnowledgeGraph,
) -> str:
    payload = json.dumps(
        {
            "repositoryId": repository_id,
            "previousScanRunId": previous_scan_run_id,
            "restoreScanRunId": restore_scan_run_id,
            "nodeIds": sorted(node.id for node in graph.nodes),
            "relationshipIds": sorted(item.id for item in graph.relationships),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"separated-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _title(markdown: str) -> str:
    first = next((line.strip() for line in markdown.splitlines() if line.strip()), "")
    if not first.startswith("# ") or not first.removeprefix("# ").strip():
        raise ValueError("knowledge document must begin with a level-one heading")
    return first.removeprefix("# ").strip()


def _reject_active_jobs(jobs: IndexingJobStore) -> None:
    active = [
        job.id
        for job in jobs.list_jobs()
        if job.status in {IndexingJobStatus.QUEUED, IndexingJobStatus.RUNNING}
    ]
    if active:
        raise RuntimeError(
            "repository separation requires an idle indexing queue; active jobs: "
            + ", ".join(active[:10])
        )
