from __future__ import annotations

from pathlib import Path

from ai_code_intelligence.agents.bedrock import BedrockConverseClient
from ai_code_intelligence.agents.cache import LocalStructuredResponseCache
from ai_code_intelligence.agents.supervisor import CodeIntelligenceSupervisor
from ai_code_intelligence.analysis.deployment_agent import DeploymentRiskAgent
from ai_code_intelligence.analysis.impact import ImpactAnalyzer
from ai_code_intelligence.application.adoption import LegacySnapshotAdoptionService
from ai_code_intelligence.application.deployment import DeploymentAnalysisWorkflow
from ai_code_intelligence.application.indexing import IndexingPipeline, InfrastructureFactories
from ai_code_intelligence.application.models import ScanResult
from ai_code_intelligence.application.onboarding import IndexingWorker, RepositoryOnboardingService
from ai_code_intelligence.application.reader_backfill import ReaderEditionBackfillService
from ai_code_intelligence.application.separation import RepositorySeparationService
from ai_code_intelligence.config import ApplicationConfig, optional_secret, required_secret
from ai_code_intelligence.domain.models import KnowledgeGraph
from ai_code_intelligence.embeddings.chunker import DocumentChunker
from ai_code_intelligence.embeddings.indexer import KnowledgeEmbeddingScopeRefresher
from ai_code_intelligence.embeddings.provider import TitanEmbeddingProvider
from ai_code_intelligence.embeddings.vector_store import PostgresVectorStore
from ai_code_intelligence.git.change_analyzer import GitChangeAnalyzer
from ai_code_intelligence.git.gitlab_checkout import ManagedGitLabCheckout
from ai_code_intelligence.git.local import LocalGitProvider
from ai_code_intelligence.graph.neo4j_retrieval import Neo4jGraphEvidenceRetriever
from ai_code_intelligence.graph.neo4j_store import Neo4jGraphStore
from ai_code_intelligence.graph.snapshot import LocalGraphSnapshotStore
from ai_code_intelligence.jira.client import EnvironmentCredentialProvider, JiraClient
from ai_code_intelligence.jira.service import JiraIntegrationService
from ai_code_intelligence.jira.store import (
    FileJiraConfigurationStore,
    JiraConfigurationStore,
    PostgresJiraConfigurationStore,
)
from ai_code_intelligence.knowledge.agents import KnowledgeAgents
from ai_code_intelligence.knowledge.chat import ArchitectureChatService
from ai_code_intelligence.knowledge.service import KnowledgeService
from ai_code_intelligence.knowledge.store import LocalDocumentStore, S3DocumentStore
from ai_code_intelligence.persistence.catalog import FileRepositoryCatalog, RepositoryStateStore
from ai_code_intelligence.persistence.postgres import PostgresMetadataStore
from ai_code_intelligence.persistence.postgres_catalog import PostgresRepositoryCatalog
from ai_code_intelligence.reports.generator import LocalReportStore
from ai_code_intelligence.scanner.local import LocalRepositoryScanner


class ApplicationContainer:
    """Composition root; business layers depend on ports, while this class selects adapters."""

    def __init__(self, config: ApplicationConfig) -> None:
        self.config = config
        runtime_root = Path(config.analysis.report_directory).resolve().parent
        self.model = LocalStructuredResponseCache(
            BedrockConverseClient(config.aws, config.agents),
            runtime_root / "cache" / "agents",
            config.aws.analysis_model_id,
        )
        self.document_store = (
            LocalDocumentStore(config.knowledge.local_directory)
            if config.knowledge.driver == "local"
            else S3DocumentStore(
                required_secret(config.knowledge.s3_bucket_env),
                config.knowledge.s3_prefix,
                config.aws.region,
            )
        )
        chat_aws = config.aws.model_copy(
            update={
                "analysis_model_id": config.chat.model_id,
                "analysis_context_window_tokens": config.chat.context_window_tokens,
                "analysis_max_output_tokens": config.chat.max_output_tokens,
            }
        )
        chat_agents = config.agents.model_copy(
            update={
                "native_structured_output": True,
                "adaptive_thinking": False,
                "validation_retries": 1,
                "max_output_tokens": config.chat.max_output_tokens,
            }
        )

        def graph_store_factory() -> Neo4jGraphStore:
            return Neo4jGraphStore(
                config.neo4j.uri,
                config.neo4j.username,
                required_secret(config.neo4j.password_env),
                config.neo4j.database,
            )

        self.architecture_chat = ArchitectureChatService(
            BedrockConverseClient(chat_aws, chat_agents),
            self.document_store,
            config.chat,
            Neo4jGraphEvidenceRetriever(graph_store_factory, config.chat),
        )
        self.graph_snapshots = LocalGraphSnapshotStore(
            runtime_root / "graph" / "latest.json"
        )
        factories = InfrastructureFactories(
            neo4j=graph_store_factory,
            metadata=lambda: PostgresMetadataStore(
                required_secret(config.postgres.connection_string_env),
                config.postgres.schema_name,
            ),
            vector_store=lambda: PostgresVectorStore(
                required_secret(config.postgres.connection_string_env),
                config.postgres.schema_name,
            ),
            titan_provider=lambda: TitanEmbeddingProvider(
                config.aws.embedding_model_id,
                config.aws.region,
                config.aws.embedding_dimensions,
            ),
        )
        self.indexing = IndexingPipeline(
            repositories=tuple(config.repositories),
            scanner=LocalRepositoryScanner(config.scanner),
            supervisor=CodeIntelligenceSupervisor(self.model, config.agents, config.scanner),
            graph_snapshots=self.graph_snapshots,
            knowledge=KnowledgeService(
                KnowledgeAgents(self.model, config.agents.max_knowledge_context_characters),
                self.document_store,
            ),
            factories=factories,
            embedding_chunk_size=config.embeddings.chunk_size_characters,
            embedding_chunk_overlap=config.embeddings.chunk_overlap_characters,
            embedding_max_input_utf8_bytes=config.embeddings.max_input_utf8_bytes,
            max_file_bytes=config.scanner.max_file_bytes,
            embedding_concurrency=config.agents.max_concurrent_invocations,
            embeddings_enabled=config.embeddings.enabled,
            incremental_max_dependency_depth=config.analysis.max_traversal_depth,
            incremental_max_dependency_nodes=config.analysis.max_traversal_nodes,
        )
        self.repository_store: RepositoryStateStore
        if config.repository_registry.driver == "local":
            self.repository_store = FileRepositoryCatalog(config.repository_registry.local_path)
        else:
            self.repository_store = PostgresRepositoryCatalog(
                required_secret(config.postgres.connection_string_env),
                config.postgres.schema_name,
                config.repository_registry.job_lease_seconds,
            )
        self.jira_store: JiraConfigurationStore
        if config.repository_registry.driver == "local":
            self.jira_store = FileJiraConfigurationStore(config.jira.local_path)
        else:
            self.jira_store = PostgresJiraConfigurationStore(
                required_secret(config.postgres.connection_string_env),
                config.postgres.schema_name,
            )
        jira_credentials = EnvironmentCredentialProvider()
        self.jira = JiraIntegrationService(
            self.jira_store,
            self.repository_store,
            lambda connection: JiraClient(
                connection,
                config.jira.allowed_hosts,
                credential_provider=jira_credentials,
                connect_timeout_seconds=config.jira.connect_timeout_seconds,
                read_timeout_seconds=config.jira.read_timeout_seconds,
                max_response_bytes=config.jira.max_response_bytes,
            ),
            config.jira.allowed_hosts,
        )
        self.reader_backfill = ReaderEditionBackfillService(
            self.document_store,
            self.repository_store,
            factories.metadata,
        )
        self.checkout = ManagedGitLabCheckout(
            config.gitlab.checkout_root,
            config.gitlab.allowed_hosts,
            token=optional_secret(config.gitlab.token_env),
            username=config.gitlab.username,
            timeout_seconds=config.gitlab.clone_timeout_seconds,
            fetch_timeout_seconds=config.gitlab.fetch_timeout_seconds,
        )
        self.onboarding = RepositoryOnboardingService(
            catalog=self.repository_store,
            jobs=self.repository_store,
            events=self.repository_store,
            checkout=self.checkout,
            indexing=self.indexing,
            manifest_publisher=self.document_store,
            checkout_root=config.gitlab.checkout_root,
            allowed_gitlab_hosts=config.gitlab.allowed_hosts,
            index_embeddings=config.embeddings.enabled,
            heartbeat_seconds=max(
                30,
                min(300, config.repository_registry.job_lease_seconds / 4),
            ),
        )
        self.adoption = LegacySnapshotAdoptionService(
            catalog=self.repository_store,
            jobs=self.repository_store,
            documents=self.document_store,
            graph_snapshots=self.graph_snapshots,
            graph_factory=factories.neo4j,
            metadata_factory=factories.metadata,
        )
        self.separation = RepositorySeparationService(
            catalog=self.repository_store,
            jobs=self.repository_store,
            documents=self.document_store,
            graph_snapshots=self.graph_snapshots,
            graph_factory=factories.neo4j,
            metadata_factory=factories.metadata,
            embedding_refresher=KnowledgeEmbeddingScopeRefresher(
                DocumentChunker(
                    config.embeddings.chunk_size_characters,
                    config.embeddings.chunk_overlap_characters,
                    max_utf8_bytes=config.embeddings.max_input_utf8_bytes,
                ),
                factories.titan_provider,
                factories.vector_store,
                config.agents.max_concurrent_invocations,
            ),
        )
        self.worker = IndexingWorker(
            self.onboarding,
            config.repository_registry.worker_poll_seconds,
        )
        self.deployment = DeploymentAnalysisWorkflow(
            config=config,
            git_analyzer=GitChangeAnalyzer(LocalGitProvider()),
            impact_analyzer=ImpactAnalyzer(
                config.analysis.max_traversal_depth,
                config.analysis.max_traversal_nodes,
            ),
            risk_agent=DeploymentRiskAgent(
                self.model,
                config.agents.max_knowledge_context_characters,
            ),
            reports=LocalReportStore(config.analysis.report_directory),
            neo4j_factory=factories.neo4j,
            metadata_factory=factories.metadata,
        )

    def initialize_database(self) -> None:
        store = PostgresMetadataStore(
            required_secret(self.config.postgres.connection_string_env),
            self.config.postgres.schema_name,
        )
        try:
            store.initialize()
        finally:
            store.close()

    def initialize_registry(self) -> None:
        """Seed configured local repositories without starting an index operation."""

        self.repository_store.seed_local_repositories(self.config.repositories)
        self.jira.initialize()

    def recover_interrupted_jobs(self) -> None:
        """Recover abandoned jobs when this process owns the local worker."""

        recover = getattr(self.repository_store, "recover_interrupted_jobs", None)
        if callable(recover):
            recover()

    def load_graph(self) -> KnowledgeGraph | None:
        """Load the latest local snapshot, falling back to durable Neo4j."""

        if self.graph_snapshots.exists():
            try:
                return self.graph_snapshots.get()
            except (OSError, ValueError):
                pass
        graph_store = self._graph_store()
        try:
            graph_store.verify_connectivity()
            graph = graph_store.load()
            return graph if graph.nodes else None
        finally:
            graph_store.close()

    def load_latest_scan(self) -> ScanResult | None:
        """Load the latest PostgreSQL scan audit for restart-safe API reads."""

        store = PostgresMetadataStore(
            required_secret(self.config.postgres.connection_string_env),
            self.config.postgres.schema_name,
        )
        try:
            return store.latest_scan()
        finally:
            store.close()

    def close(self) -> None:
        """Release long-lived repository-store resources."""

        close = getattr(self.repository_store, "close", None)
        if callable(close):
            close()
        self.jira.close()

    def _graph_store(self) -> Neo4jGraphStore:
        return Neo4jGraphStore(
            self.config.neo4j.uri,
            self.config.neo4j.username,
            required_secret(self.config.neo4j.password_env),
            self.config.neo4j.database,
        )
