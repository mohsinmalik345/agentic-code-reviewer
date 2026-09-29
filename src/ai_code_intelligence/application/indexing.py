from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from uuid import uuid4

from ai_code_intelligence.agents.supervisor import CodeIntelligenceSupervisor
from ai_code_intelligence.analysis.incremental import (
    IncrementalIndexPlanner,
    RepositoryIncrementalPlan,
    central_architecture_fingerprint,
    refresh_plan_knowledge_sections,
)
from ai_code_intelligence.application.models import ScanResult
from ai_code_intelligence.domain.change import RepositoryChangeSet
from ai_code_intelligence.domain.knowledge import KnowledgeDocument, KnowledgeManifest, StoredDocument
from ai_code_intelligence.domain.models import (
    FileKind,
    KnowledgeGraph,
    RepositoryDefinition,
    ScannedRepository,
)
from ai_code_intelligence.domain.repositories import (
    RepositoryProvider,
    RepositoryRecord,
    RepositoryStatus,
)
from ai_code_intelligence.embeddings.chunker import DocumentChunker
from ai_code_intelligence.embeddings.indexer import EmbeddingIndexer
from ai_code_intelligence.embeddings.provider import TitanEmbeddingProvider
from ai_code_intelligence.embeddings.source import EmbeddingSourceCollector
from ai_code_intelligence.embeddings.vector_store import PostgresVectorStore
from ai_code_intelligence.graph.builder import retain_unaffected_graph
from ai_code_intelligence.graph.neo4j_store import Neo4jGraphStore
from ai_code_intelligence.graph.snapshot import LocalGraphSnapshotStore
from ai_code_intelligence.knowledge.service import KnowledgeService
from ai_code_intelligence.persistence.postgres import PostgresMetadataStore
from ai_code_intelligence.scanner.local import LocalRepositoryScanner


@dataclass(frozen=True, slots=True)
class IndexingArtifacts:
    """In-process artifacts shared by scan and deployment workflows."""

    result: ScanResult
    repositories: tuple[ScannedRepository, ...]
    graph: KnowledgeGraph
    knowledge_documents: tuple[KnowledgeDocument, ...]
    human_knowledge_documents: tuple[KnowledgeDocument, ...] = ()
    human_stored_documents: tuple[StoredDocument, ...] = ()
    changed_repository_ids: tuple[str, ...] = ()
    reused_repository_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InfrastructureFactories:
    """Lazy factories for infrastructure that is optional per operation."""

    neo4j: Callable[[], Neo4jGraphStore]
    metadata: Callable[[], PostgresMetadataStore]
    vector_store: Callable[[], PostgresVectorStore]
    titan_provider: Callable[[], TitanEmbeddingProvider]


class IndexingPipeline:
    """Runs inventory, specialist agents, validation, knowledge, persistence, and embeddings."""

    def __init__(
        self,
        repositories: tuple[RepositoryDefinition, ...],
        scanner: LocalRepositoryScanner,
        supervisor: CodeIntelligenceSupervisor,
        graph_snapshots: LocalGraphSnapshotStore,
        knowledge: KnowledgeService,
        factories: InfrastructureFactories,
        embedding_chunk_size: int,
        embedding_chunk_overlap: int,
        max_file_bytes: int,
        embedding_concurrency: int,
        embeddings_enabled: bool,
        embedding_max_input_utf8_bytes: int | None = None,
        incremental_max_dependency_depth: int = 12,
        incremental_max_dependency_nodes: int = 10_000,
        incremental_max_affected_source_ratio: float = 0.75,
    ) -> None:
        self._repositories = repositories
        self._scanner = scanner
        self._supervisor = supervisor
        self._graph_snapshots = graph_snapshots
        self._knowledge = knowledge
        self._factories = factories
        self._chunk_size = embedding_chunk_size
        self._chunk_overlap = embedding_chunk_overlap
        self._embedding_max_input_utf8_bytes = embedding_max_input_utf8_bytes
        self._max_file_bytes = max_file_bytes
        self._embedding_concurrency = embedding_concurrency
        self._embeddings_enabled = embeddings_enabled
        self._incremental_planner = IncrementalIndexPlanner(
            max_dependency_depth=incremental_max_dependency_depth,
            max_dependency_nodes=incremental_max_dependency_nodes,
            max_affected_source_ratio=incremental_max_affected_source_ratio,
        )
        self._logger = logging.getLogger(__name__)

    def reusable_repository_ids(
        self,
        records: tuple[RepositoryRecord, ...],
    ) -> frozenset[str]:
        """Return records backed by the same commit in both published graph and manifest."""

        reusable, _published = self.repository_reuse_state(records)
        return reusable

    def repository_reuse_state(
        self,
        records: tuple[RepositoryRecord, ...],
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Return safely reusable records and all repositories in the active publication."""

        if not records:
            return frozenset(), frozenset()
        baseline = self._validated_published_snapshot()
        if baseline is None:
            return frozenset(), frozenset()
        graph, manifest = baseline
        manifest_by_id = {item.id: item for item in manifest.repositories}
        published_ids = frozenset(manifest_by_id)
        record_ids = {record.id for record in records}
        if not set(manifest_by_id).issubset(record_ids) or manifest.graph.statistics != graph.statistics:
            return frozenset(), published_ids
        graph_repository_ids = {
            node.repository_id for node in graph.nodes if node.repository_id is not None
        }
        node_counts = Counter(
            node.repository_id for node in graph.nodes if node.repository_id is not None
        )
        relationship_counts = Counter(
            relationship.repository_id
            for relationship in graph.relationships
            if relationship.repository_id is not None
        )
        reusable: set[str] = set()
        for record in records:
            entry = manifest_by_id.get(record.id)
            if (
                record.provider is not RepositoryProvider.GITLAB
                or record.status not in {RepositoryStatus.INDEXED, RepositoryStatus.READY}
                or record.exact_commit is None
                or record.knowledge_base_uri is None
                or record.last_scan_run_id != manifest.scan_run_id
                or record.indexed_at is None
                or record.id not in graph_repository_ids
                or record.graph_node_count != node_counts.get(record.id, 0)
                or record.graph_relationship_count != relationship_counts.get(record.id, 0)
                or entry is None
                or entry.commit_sha != record.exact_commit
                or entry.knowledge_uri != record.knowledge_base_uri
                or entry.human_knowledge_uri is None
                or not self._knowledge.documents_are_readable(
                    (entry.knowledge_uri, entry.human_knowledge_uri)
                )
            ):
                continue
            reusable.add(record.id)
        return frozenset(reusable), published_ids

    def run(
        self,
        *,
        repositories: tuple[RepositoryDefinition, ...] | None = None,
        changed_repository_ids: frozenset[str] | None = None,
        repository_change_sets: Mapping[str, RepositoryChangeSet] | None = None,
        persist_neo4j: bool = False,
        index_embeddings: bool = False,
        persist_metadata: bool = False,
        require_incremental_baseline: bool = False,
        cancellation_check: Callable[[], None] | None = None,
        begin_publication: Callable[[], None] | None = None,
    ) -> IndexingArtifacts:
        definitions = repositories if repositories is not None else self._repositories
        if not definitions:
            raise ValueError("at least one repository is required for indexing")
        definition_ids = tuple(item.id for item in definitions)
        if len(definition_ids) != len(set(definition_ids)):
            raise ValueError("repository definitions must have unique IDs")
        if changed_repository_ids is not None and (
            not changed_repository_ids or not changed_repository_ids.issubset(definition_ids)
        ):
            raise ValueError("changed repository IDs must be a nonempty portfolio subset")
        if repository_change_sets is not None and (
            changed_repository_ids is None
            or not set(repository_change_sets).issubset(changed_repository_ids)
        ):
            raise ValueError("repository change sets must belong to changed repositories")

        previous_graph: KnowledgeGraph | None = None
        previous_manifest: KnowledgeManifest | None = None
        effective_changed_ids = changed_repository_ids
        baseline_requested = changed_repository_ids is not None and (
            len(changed_repository_ids) < len(definitions) or bool(repository_change_sets)
        )
        if baseline_requested and changed_repository_ids is not None:
            previous_graph, previous_manifest = self._incremental_baseline(
                definition_ids,
                changed_repository_ids,
            )
            if previous_graph is None or previous_manifest is None:
                if require_incremental_baseline:
                    raise RuntimeError(
                        "repository-scoped indexing requires a complete validated published "
                        "portfolio baseline; run a full portfolio reindex first"
                    )
                self._logger.warning(
                    "Incremental baseline is incomplete; falling back to a full portfolio index",
                    extra={"fields": {"requested_changed_ids": sorted(changed_repository_ids)}},
                )
                effective_changed_ids = frozenset(definition_ids)

        incremental = (
            effective_changed_ids is not None
            and previous_graph is not None
            and previous_manifest is not None
            and (
                len(effective_changed_ids) < len(definitions)
                or bool(repository_change_sets)
            )
        )
        scan_definitions = (
            tuple(item for item in definitions if item.id in effective_changed_ids)
            if incremental and effective_changed_ids is not None
            else definitions
        )
        checkpoint = cancellation_check or (lambda: None)
        checkpoint()
        scanned_repositories = self._scanner.scan_all(scan_definitions)
        checkpoint()
        incremental_plans: dict[str, RepositoryIncrementalPlan] = {}
        if incremental and previous_graph is not None and repository_change_sets:
            for repository in scanned_repositories:
                change_set = repository_change_sets.get(repository.definition.id)
                if change_set is None:
                    continue
                plan = self._incremental_planner.plan(repository, previous_graph, change_set)
                incremental_plans[repository.definition.id] = plan
                self._logger.info(
                    "Incremental repository plan created",
                    extra={
                        "fields": {
                            "repository_id": plan.repository_id,
                            "mode": plan.mode,
                            "changed_file_count": len(change_set.files),
                            "direct_entity_count": len(plan.directly_affected_node_ids),
                            "dependency_entity_count": len(plan.dependency_node_ids),
                            "affected_source_file_count": len(plan.affected_source_paths),
                            "reason": plan.reason,
                        }
                    },
                )
        if repository_change_sets is not None:
            graph_result = self._supervisor.analyze(
                scanned_repositories,
                previous_graph=previous_graph if incremental else None,
                changed_repository_ids=effective_changed_ids if incremental else None,
                incremental_plans=incremental_plans if incremental_plans else None,
                cancellation_check=checkpoint,
            )
        else:
            graph_result = self._supervisor.analyze(
                scanned_repositories,
                previous_graph=previous_graph if incremental else None,
                changed_repository_ids=effective_changed_ids if incremental else None,
                cancellation_check=checkpoint,
            )
        checkpoint()
        incremental_plans = {
            repository_id: refresh_plan_knowledge_sections(plan, graph_result.graph)
            for repository_id, plan in incremental_plans.items()
        }
        regenerate_central = (
            previous_graph is None
            or central_architecture_fingerprint(previous_graph)
            != central_architecture_fingerprint(graph_result.graph)
        )
        if incremental:
            assert previous_manifest is not None
            if repository_change_sets is not None:
                knowledge_result = self._knowledge.generate_incremental(
                    scanned_repositories,
                    graph_result,
                    definition_ids,
                    previous_manifest,
                    incremental_plans=incremental_plans,
                    regenerate_central=regenerate_central,
                    cancellation_check=checkpoint,
                )
            else:
                knowledge_result = self._knowledge.generate_incremental(
                    scanned_repositories,
                    graph_result,
                    definition_ids,
                    previous_manifest,
                    cancellation_check=checkpoint,
                )
        else:
            knowledge_result = self._knowledge.generate(
                scanned_repositories,
                graph_result,
                cancellation_check=checkpoint,
            )
        checkpoint()
        documents = (*knowledge_result.repository_documents, knowledge_result.central_document)
        human_documents = (
            *knowledge_result.human_repository_documents,
            knowledge_result.human_central_document,
        )
        invocations = (*graph_result.invocations, *knowledge_result.invocations)

        if begin_publication is not None:
            begin_publication()
        else:
            checkpoint()

        embedded_count = 0
        if index_embeddings or self._embeddings_enabled:
            metadata = self._factories.metadata()
            vector_store: PostgresVectorStore | None = None
            try:
                metadata.initialize()
                vector_store = self._factories.vector_store()
                provider = self._factories.titan_provider()
                indexer = EmbeddingIndexer(
                    DocumentChunker(
                        self._chunk_size,
                        self._chunk_overlap,
                        max_utf8_bytes=self._embedding_max_input_utf8_bytes,
                    ),
                    provider,
                    vector_store,
                    self._embedding_concurrency,
                )
                selective_paths = {
                    repository_id: plan.affected_source_paths
                    | frozenset(
                        file.current_path
                        for file in plan.change_set.files
                        if file.current_path is not None
                        and any(
                            scanned_file.relative_path == file.current_path
                            and scanned_file.kind in {FileKind.README, FileKind.DOCUMENTATION}
                            for repository in scanned_repositories
                            if repository.definition.id == repository_id
                            for scanned_file in repository.files
                        )
                    )
                    for repository_id, plan in incremental_plans.items()
                    if plan.is_selective
                }
                embedding_documents = (
                    tuple(
                        document
                        for document in documents
                        if (
                            document.repository_id in knowledge_result.updated_repository_ids
                            or (document.repository_id is None and knowledge_result.central_updated)
                        )
                    )
                    if incremental and effective_changed_ids is not None
                    else documents
                )
                sources = EmbeddingSourceCollector(self._max_file_bytes).collect(
                    scanned_repositories,
                    graph_result.graph,
                    embedding_documents,
                    included_source_paths=selective_paths if selective_paths else None,
                )
                if incremental and effective_changed_ids is not None and incremental_plans:
                    source_selective_ids = frozenset(selective_paths)
                    knowledge_only_ids = frozenset(
                        knowledge_result.updated_repository_ids
                    ).difference(effective_changed_ids)
                    selective_ids = source_selective_ids | knowledge_only_ids
                    full_ids = effective_changed_ids.difference(source_selective_ids)
                    stale_source_uris = frozenset(
                        f"repository://{repository_id}/{path}"
                        for repository_id in source_selective_ids
                        for path in incremental_plans[repository_id].invalidated_paths
                    )
                    embedded_count = indexer.index_delta(
                        sources,
                        full_repository_ids=full_ids,
                        selective_repository_ids=selective_ids,
                        stale_source_uris=stale_source_uris,
                        replace_central=knowledge_result.central_updated,
                    )
                elif incremental and effective_changed_ids is not None:
                    embedded_count = indexer.index_scoped(sources, effective_changed_ids)
                else:
                    embedded_count = indexer.index(sources)
            finally:
                if vector_store:
                    vector_store.close()
                metadata.close()

        # Keep the active graph on the last complete publication until every
        # embedding has succeeded. This avoids exposing a graph snapshot that
        # the latest knowledge manifest and vector index do not describe.
        neo4j_snapshot_id: str | None = None
        if persist_neo4j:
            if (
                previous_graph is not None
                and previous_manifest is not None
                and _same_graph_content(previous_graph, graph_result.graph)
            ):
                neo4j_snapshot_id = previous_manifest.graph.neo4j_snapshot_id
            else:
                graph_store = self._factories.neo4j()
                try:
                    graph_store.verify_connectivity()
                    expected_snapshot_id = (
                        previous_manifest.graph.neo4j_snapshot_id
                        if previous_manifest is not None
                        else None
                    )
                    if (
                        incremental
                        and previous_graph is not None
                        and expected_snapshot_id is not None
                    ):
                        neo4j_snapshot_id = graph_store.patch(
                            previous_graph,
                            graph_result.graph,
                            expected_snapshot_id=expected_snapshot_id,
                        )
                    else:
                        neo4j_snapshot_id = graph_store.replace(graph_result.graph)
                finally:
                    graph_store.close()

        graph_path = self._graph_snapshots.put(graph_result.graph)

        result = ScanResult(
            scan_run_id=str(uuid4()),
            indexing_mode="bedrock-agents-incremental" if incremental else "bedrock-agents",
            repository_ids=definition_ids,
            graph_path=graph_path,
            statistics=graph_result.graph.statistics,
            knowledge_uris=tuple(item.uri for item in knowledge_result.stored_documents),
            embedded_chunk_count=embedded_count,
            agent_invocations=invocations,
            evidence_warnings=graph_result.evidence_warnings,
            neo4j_snapshot_id=neo4j_snapshot_id,
        )

        if persist_metadata:
            metadata = self._factories.metadata()
            try:
                metadata.initialize()
                metadata.record_scan(
                    result,
                    invocations,
                    (*documents, *human_documents),
                    (
                        *knowledge_result.stored_documents,
                        *knowledge_result.human_stored_documents,
                    ),
                )
            finally:
                metadata.close()

        return IndexingArtifacts(
            result,
            scanned_repositories,
            graph_result.graph,
            documents,
            human_documents,
            knowledge_result.human_stored_documents,
            tuple(sorted(effective_changed_ids or definition_ids)),
            tuple(
                repository_id
                for repository_id in definition_ids
                if incremental
                and effective_changed_ids is not None
                and repository_id not in effective_changed_ids
                and repository_id not in knowledge_result.updated_repository_ids
            ),
        )

    def _incremental_baseline(
        self,
        repository_ids: tuple[str, ...],
        changed_repository_ids: frozenset[str],
    ) -> tuple[KnowledgeGraph | None, KnowledgeManifest | None]:
        baseline = self._validated_published_snapshot()
        if baseline is None:
            return None, None
        graph, manifest = baseline
        manifest_by_id = {item.id: item for item in manifest.repositories}
        graph_repository_ids = {
            node.repository_id for node in graph.nodes if node.repository_id is not None
        }
        unchanged_ids = set(repository_ids).difference(changed_repository_ids)
        if (
            not unchanged_ids.issubset(graph_repository_ids)
            or not unchanged_ids.issubset(manifest_by_id)
            or any(
                manifest_by_id[repository_id].human_knowledge_uri is None
                for repository_id in unchanged_ids
            )
        ):
            return None, None
        removed_repository_ids = frozenset(graph_repository_ids.difference(repository_ids))
        if removed_repository_ids:
            graph = retain_unaffected_graph(graph, removed_repository_ids)
        return graph, manifest

    def _validated_published_snapshot(
        self,
    ) -> tuple[KnowledgeGraph, KnowledgeManifest] | None:
        """Bind the local graph to the manifest's exact active Neo4j snapshot."""

        if not self._graph_snapshots.exists():
            return None
        graph_store: Neo4jGraphStore | None = None
        try:
            graph = self._graph_snapshots.get()
            manifest = self._knowledge.latest_manifest()
            if (
                manifest is None
                or manifest.schema_version != 2
                or manifest.graph.neo4j_snapshot_id is None
                or manifest.graph.statistics != graph.statistics
            ):
                return None
            graph_store = self._factories.neo4j()
            graph_store.verify_connectivity()
            if graph_store.active_snapshot_id() != manifest.graph.neo4j_snapshot_id:
                return None
            durable_graph = graph_store.load()
            if not _same_graph_content(graph, durable_graph):
                return None
            return graph, manifest
        except Exception:
            self._logger.warning("Unable to validate the incremental indexing baseline", exc_info=True)
            return None
        finally:
            if graph_store is not None:
                graph_store.close()


def _same_graph_content(left: KnowledgeGraph, right: KnowledgeGraph) -> bool:
    return {node.id: node for node in left.nodes} == {node.id: node for node in right.nodes} and {
        relationship.id: relationship for relationship in left.relationships
    } == {relationship.id: relationship for relationship in right.relationships}
