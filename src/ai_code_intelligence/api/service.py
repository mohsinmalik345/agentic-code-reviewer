from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from threading import Event, RLock
from typing import Literal

from ai_code_intelligence.application.container import ApplicationContainer
from ai_code_intelligence.application.indexing import IndexingArtifacts
from ai_code_intelligence.application.models import DeploymentResult, ScanResult
from ai_code_intelligence.application.onboarding import GitLabRepositoryRegistration
from ai_code_intelligence.domain.knowledge import (
    KnowledgeDocumentView,
    KnowledgeManifest,
)
from ai_code_intelligence.domain.models import (
    AGENT_DESCRIPTORS,
    AgentDescriptor,
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    RepositoryDefinition,
)
from ai_code_intelligence.domain.repositories import (
    IndexingJob,
    IndexingJobEvent,
    IndexingJobStatus,
    RepositoryRecord,
    RepositoryStatus,
)
from ai_code_intelligence.jira.models import (
    JiraConnection,
    JiraConnectionTestResult,
    JiraProjectMapping,
)
from ai_code_intelligence.knowledge.chat import (
    ArchitectureChatAnswer,
    ArchitectureChatMessage,
    ArchitectureChatUnavailableError,
)

_LOGGER = logging.getLogger(__name__)


class UserInputError(ValueError):
    """A validated, credential-free error that may be returned to an API caller."""


class PlatformService:
    """Application facade; GraphQL resolvers contain no business logic."""

    def __init__(self, container: ApplicationContainer) -> None:
        self._container = container
        self._latest: IndexingArtifacts | None = None
        self._latest_result: ScanResult | None = None
        self._graph: KnowledgeGraph | None = None
        self._graph_source = "none"
        self._graph_mtime_ns: int | None = None
        self._lock = RLock()

    def startup(self) -> None:
        """Seed bootstrap repositories and hydrate graph reads without invoking Bedrock."""

        self._container.initialize_registry()
        if self._container.config.repository_registry.driver == "local":
            self._container.recover_interrupted_jobs()
        self._refresh_graph(force=True)
        try:
            self._latest_result = self._container.load_latest_scan()
        except Exception as error:
            _LOGGER.warning(
                "Unable to hydrate the latest scan audit",
                extra={"fields": {"error_type": type(error).__name__}},
            )

    def shutdown(self) -> None:
        """Release application resources."""

        self._container.close()

    def run_worker(self, stop: Event) -> None:
        """Run the local durable worker; production PostgreSQL mode uses its own process."""

        self._container.worker.run_forever(stop)

    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "indexingMode": "bedrock-agents",
            "modelId": self._container.config.aws.analysis_model_id,
            "chatModelId": self._container.config.chat.model_id,
            "runtime": "python-3.12",
        }

    def agents(self) -> tuple[AgentDescriptor, ...]:
        return AGENT_DESCRIPTORS

    def latest_scan(self) -> ScanResult | None:
        return self._latest.result if self._latest else self._latest_result

    def repositories(self, *, include_disabled: bool = False) -> tuple[RepositoryRecord, ...]:
        return self._container.repository_store.list_repositories(include_disabled=include_disabled)

    def repository(self, repository_id: str) -> RepositoryRecord | None:
        return self._container.repository_store.get_repository(repository_id)

    def jira_connections(
        self,
        *,
        include_disabled: bool = False,
    ) -> tuple[JiraConnection, ...]:
        """Return credential-free Jira connection metadata."""

        return self._container.jira.list_connections(include_disabled=include_disabled)

    def jira_credential_configured(self, connection: JiraConnection) -> bool:
        return self._container.jira.credential_configured(connection)

    def save_jira_connection(self, values: Mapping[str, object]) -> JiraConnection:
        try:
            return self._container.jira.save_connection(values)
        except ValueError as error:
            raise UserInputError(str(error)) from error

    def test_jira_connection(self, connection_id: str) -> JiraConnectionTestResult:
        try:
            return self._container.jira.test_connection(connection_id)
        except ValueError as error:
            raise UserInputError(str(error)) from error

    def delete_jira_connection(self, connection_id: str) -> bool:
        return self._container.jira.delete_connection(connection_id)

    def jira_project_mappings(
        self,
        *,
        repository_id: str | None = None,
    ) -> tuple[JiraProjectMapping, ...]:
        return self._container.jira.list_mappings(repository_id=repository_id)

    def save_jira_project_mapping(
        self,
        values: Mapping[str, object],
    ) -> JiraProjectMapping:
        try:
            return self._container.jira.save_mapping(values)
        except ValueError as error:
            raise UserInputError(str(error)) from error

    def delete_jira_project_mapping(self, mapping_id: str) -> bool:
        return self._container.jira.delete_mapping(mapping_id)

    def central_knowledge_document(
        self,
        *,
        human_readable: bool = False,
    ) -> KnowledgeDocumentView:
        """Return the latest central Markdown through the authenticated application boundary."""

        manifest = self._latest_knowledge_manifest()
        uri = manifest.human_central_knowledge_uri if human_readable else manifest.central_knowledge_uri
        if uri is None:
            raise UserInputError("human-readable knowledge base has not been generated yet")
        return self._read_knowledge_document(
            kind="central-human" if human_readable else "central",
            title=("Central Knowledge Base — Reader Edition" if human_readable else "Central Knowledge Base"),
            repository_id=None,
            uri=uri,
            manifest=manifest,
        )

    def repository_knowledge_document(
        self,
        repository_id: str,
        *,
        human_readable: bool = False,
    ) -> KnowledgeDocumentView:
        """Return one registered repository's latest published Markdown document."""

        record = self._container.repository_store.get_repository(repository_id)
        if record is None:
            raise UserInputError("repository was not found")
        if record.status is not RepositoryStatus.INDEXED or record.knowledge_base_uri is None:
            raise UserInputError("repository knowledge base is not available yet")

        manifest = self._latest_knowledge_manifest()
        entry = next(
            (item for item in manifest.repositories if item.id == repository_id),
            None,
        )
        if entry is None:
            raise UserInputError("repository knowledge base is not available yet")
        if not human_readable and entry.knowledge_uri != record.knowledge_base_uri:
            _LOGGER.error(
                "Repository and knowledge manifest document pointers do not match",
                extra={"fields": {"repository_id": repository_id}},
            )
            raise RuntimeError("knowledge base storage is unavailable")
        uri = entry.human_knowledge_uri if human_readable else entry.knowledge_uri
        if uri is None:
            raise UserInputError("human-readable knowledge base has not been generated yet")
        return self._read_knowledge_document(
            kind="repository-human" if human_readable else "repository",
            title=f"{record.name} — Reader Edition" if human_readable else record.name,
            repository_id=record.id,
            uri=uri,
            manifest=manifest,
        )

    def indexing_job(self, job_id: str) -> IndexingJob | None:
        return self._container.repository_store.get_job(job_id)

    def indexing_jobs(
        self,
        *,
        status: str | None,
        limit: int,
    ) -> tuple[IndexingJob, ...]:
        try:
            parsed_status = IndexingJobStatus(status) if status is not None else None
        except ValueError as error:
            raise UserInputError(f"unsupported indexing job status: {status}") from error
        jobs = self._container.repository_store.list_jobs(status=parsed_status)
        bounded_limit = min(max(limit, 0), 500)
        return tuple(
            sorted(jobs, key=lambda job: (job.created_at, job.id), reverse=True)[:bounded_limit]
        )

    def indexing_job_events(
        self,
        job_id: str,
        *,
        limit: int,
    ) -> tuple[IndexingJobEvent, ...]:
        if self._container.repository_store.get_job(job_id) is None:
            raise UserInputError("indexing job was not found")
        bounded = min(max(limit, 1), 1_000)
        return self._container.repository_store.list_job_events(job_id, limit=bounded)

    def onboard_repository(
        self,
        url: str,
        *,
        repository_id: str | None,
        name: str | None,
        description: str | None,
        ref: str | None,
    ) -> tuple[RepositoryRecord, IndexingJob]:
        try:
            result = self._container.onboarding.register_gitlab(
                url,
                repository_id=repository_id,
                name=name,
                description=description,
                ref=ref,
            )
        except ValueError as error:
            raise UserInputError(str(error)) from error
        return result.repository, result.job

    def onboard_repositories(
        self,
        inputs: tuple[Mapping[str, object], ...],
    ) -> tuple[tuple[RepositoryRecord, ...], IndexingJob]:
        """Register a validated batch atomically and return its one queued portfolio job."""

        registrations = tuple(
            GitLabRepositoryRegistration(
                url=_required_string(item, "gitlabUrl"),
                repository_id=_optional_string(item, "repositoryId"),
                name=_optional_string(item, "name"),
                description=_optional_string(item, "description"),
                ref=_optional_string(item, "ref"),
            )
            for item in inputs
        )
        try:
            result = self._container.onboarding.register_gitlab_repositories(registrations)
        except ValueError as error:
            raise UserInputError(str(error)) from error
        return result.repositories, result.job

    def reindex(self, repository_id: str | None = None) -> IndexingJob:
        try:
            return self._container.onboarding.enqueue_reindex(repository_id)
        except ValueError as error:
            raise UserInputError(str(error)) from error

    def cancel_indexing_job(self, job_id: str) -> IndexingJob:
        """Cancel queued indexing or request a cooperative stop from the worker."""

        try:
            return self._container.onboarding.cancel_job(job_id)
        except (KeyError, ValueError) as error:
            raise UserInputError(str(error)) from error

    def ask_architecture(
        self,
        question: str,
        *,
        repository_id: str | None,
        history: tuple[ArchitectureChatMessage, ...],
    ) -> ArchitectureChatAnswer:
        """Answer from the latest published reader editions through the chat agent."""

        if repository_id is not None:
            repository = self._container.repository_store.get_repository(repository_id)
            if repository is None:
                raise UserInputError("repository was not found")
            if repository.status is not RepositoryStatus.INDEXED:
                raise UserInputError("repository knowledge base is not available yet")
        try:
            return self._container.architecture_chat.ask(
                question,
                repository_id=repository_id,
                history=history,
            )
        except ValueError as error:
            raise UserInputError(str(error)) from error
        except ArchitectureChatUnavailableError as error:
            raise UserInputError(error.user_message) from error

    def graph_snapshot(self) -> dict[str, object]:
        graph = self._graph_view()
        return {
            "available": graph is not None,
            "source": self._graph_source,
            "generatedAt": graph.generated_at.isoformat() if graph else None,
            "repositoryIds": tuple(
                sorted({node.repository_id for node in graph.nodes if node.repository_id is not None})
            )
            if graph is not None
            else (),
            "statistics": graph.statistics if graph else None,
        }

    def scan(
        self,
        *,
        persist_neo4j: bool,
        index_embeddings: bool,
        persist_metadata: bool,
    ) -> ScanResult:
        with self._lock:
            self._latest = self._container.indexing.run(
                repositories=self._repository_definitions(),
                persist_neo4j=persist_neo4j,
                index_embeddings=index_embeddings,
                persist_metadata=persist_metadata,
            )
            self._latest_result = self._latest.result
            self._set_graph(self._latest.graph, "memory")
            return self._latest.result

    def analyze_deployment(
        self,
        repository_id: str,
        *,
        base_revision: str | None,
        head_revision: str | None,
        staged: bool,
        intent: str | None,
        use_ai: bool,
        use_neo4j_impact: bool,
        refresh_index: bool,
        persist_neo4j: bool,
        index_embeddings: bool,
        persist_metadata: bool,
    ) -> DeploymentResult:
        with self._lock:
            if refresh_index:
                self._latest = self._container.indexing.run(
                    repositories=self._repository_definitions(),
                    persist_neo4j=persist_neo4j or use_neo4j_impact,
                    index_embeddings=index_embeddings,
                    persist_metadata=persist_metadata,
                )
                self._latest_result = self._latest.result
                self._set_graph(self._latest.graph, "memory")
            if self._latest is None:
                raise RuntimeError("no graph is loaded; set refreshIndex=true or run scan first")
            return self._container.deployment.run(
                self._latest,
                repository_id,
                base_revision=base_revision,
                head_revision=head_revision,
                staged=staged,
                intent=intent,
                use_ai=use_ai,
                use_neo4j_impact=use_neo4j_impact,
                persist_metadata=persist_metadata,
            )

    def graph_nodes(
        self,
        repository_id: str | None,
        node_type: str | None,
        limit: int,
    ) -> tuple[GraphNode, ...]:
        graph = self._graph_view()
        if graph is None:
            return ()
        values = (
            node
            for node in graph.nodes
            if (repository_id is None or node.repository_id == repository_id)
            and (node_type is None or node.type.value == node_type)
        )
        return tuple(list(values)[: min(max(limit, 0), 1000)])

    def graph_relationships(
        self,
        repository_id: str | None,
        relationship_type: str | None,
        limit: int,
    ) -> tuple[GraphRelationship, ...]:
        graph = self._graph_view()
        if graph is None:
            return ()
        values = (
            relationship
            for relationship in graph.relationships
            if (repository_id is None or relationship.repository_id == repository_id)
            and (relationship_type is None or relationship.type.value == relationship_type)
        )
        return tuple(list(values)[: min(max(limit, 0), 1000)])

    def _graph_view(self) -> KnowledgeGraph | None:
        self._refresh_graph(force=False)
        return self._graph

    def _refresh_graph(self, *, force: bool) -> None:
        snapshot_path: Path = self._container.graph_snapshots.path
        try:
            mtime_ns = snapshot_path.stat().st_mtime_ns if snapshot_path.exists() else None
            if not force and mtime_ns == self._graph_mtime_ns:
                return
            graph = self._container.load_graph()
            if graph is not None:
                self._graph = graph
                self._graph_source = "local_snapshot" if mtime_ns is not None else "neo4j"
                self._graph_mtime_ns = mtime_ns
        except Exception as error:
            _LOGGER.warning(
                "Unable to hydrate the latest graph",
                extra={"fields": {"error_type": type(error).__name__}},
            )

    def _set_graph(self, graph: KnowledgeGraph, source: str) -> None:
        self._graph = graph
        self._graph_source = source
        path = self._container.graph_snapshots.path
        self._graph_mtime_ns = path.stat().st_mtime_ns if path.exists() else None

    def _repository_definitions(self) -> tuple[RepositoryDefinition, ...]:
        return tuple(
            record.to_repository_definition()
            for record in self._container.repository_store.list_repositories(include_disabled=False)
        )

    def _latest_knowledge_manifest(self) -> KnowledgeManifest:
        try:
            manifest = self._container.document_store.latest_manifest()
        except Exception as error:
            _LOGGER.error(
                "Unable to load the latest knowledge manifest",
                extra={"fields": {"error_type": type(error).__name__}},
            )
            raise RuntimeError("knowledge base storage is unavailable") from error
        if manifest is None:
            raise UserInputError("knowledge base has not been generated yet")
        return manifest

    def _read_knowledge_document(
        self,
        *,
        kind: Literal["repository", "central", "repository-human", "central-human"],
        title: str,
        repository_id: str | None,
        uri: str,
        manifest: KnowledgeManifest,
    ) -> KnowledgeDocumentView:
        try:
            markdown = self._container.document_store.get(uri)
        except Exception as error:
            _LOGGER.error(
                "Unable to load a published knowledge document",
                extra={"fields": {"error_type": type(error).__name__, "kind": kind}},
            )
            raise RuntimeError("knowledge base storage is unavailable") from error
        payload = markdown.encode("utf-8")
        return KnowledgeDocumentView(
            kind=kind,
            title=title,
            repository_id=repository_id,
            source_uri=uri,
            markdown=markdown,
            content_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            scan_run_id=manifest.scan_run_id,
            generated_at=manifest.generated_at,
        )


def _required_string(item: Mapping[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise UserInputError(f"repository input {key} must be a string")
    return value


def _optional_string(item: Mapping[str, object], key: str) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise UserInputError(f"repository input {key} must be a string")
    return value
