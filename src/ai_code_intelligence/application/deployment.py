from __future__ import annotations

from collections.abc import Callable

from ai_code_intelligence.analysis.deployment_agent import (
    DeploymentRiskAgent,
    fail_closed_assessment,
)
from ai_code_intelligence.analysis.impact import ImpactAnalyzer
from ai_code_intelligence.application.indexing import IndexingArtifacts
from ai_code_intelligence.application.models import DeploymentResult
from ai_code_intelligence.config import ApplicationConfig
from ai_code_intelligence.git.change_analyzer import GitChangeAnalyzer
from ai_code_intelligence.graph.neo4j_store import Neo4jGraphStore
from ai_code_intelligence.persistence.postgres import PostgresMetadataStore
from ai_code_intelligence.reports.generator import DeploymentReportGenerator, LocalReportStore


class DeploymentAnalysisWorkflow:
    """Combines Git, graph traversal, Bedrock reasoning, and deterministic deployment policy."""

    def __init__(
        self,
        config: ApplicationConfig,
        git_analyzer: GitChangeAnalyzer,
        impact_analyzer: ImpactAnalyzer,
        risk_agent: DeploymentRiskAgent,
        reports: LocalReportStore,
        neo4j_factory: Callable[[], Neo4jGraphStore],
        metadata_factory: Callable[[], PostgresMetadataStore],
    ) -> None:
        self._config = config
        self._git = git_analyzer
        self._impact = impact_analyzer
        self._risk_agent = risk_agent
        self._reports = reports
        self._neo4j_factory = neo4j_factory
        self._metadata_factory = metadata_factory

    def run(
        self,
        artifacts: IndexingArtifacts,
        repository_id: str,
        *,
        base_revision: str | None = None,
        head_revision: str | None = None,
        staged: bool = False,
        intent: str | None = None,
        use_ai: bool = True,
        use_neo4j_impact: bool = False,
        persist_metadata: bool = False,
    ) -> DeploymentResult:
        repository = next(
            (item for item in artifacts.repositories if item.definition.id == repository_id),
            None,
        )
        if repository is None:
            raise ValueError(f"unknown repository id: {repository_id}")
        change = self._git.analyze(
            repository_id,
            repository.root_path,
            artifacts.graph,
            base_revision=base_revision,
            head_revision=head_revision,
            staged=staged,
        )
        impact_graph = artifacts.graph
        if use_neo4j_impact:
            starts = self._impact.start_node_ids(change, artifacts.graph)
            neo4j = self._neo4j_factory()
            try:
                neo4j.verify_connectivity()
                impact_graph = neo4j.impact_subgraph(
                    starts,
                    self._config.analysis.max_traversal_depth,
                    self._config.analysis.max_traversal_nodes,
                )
            finally:
                neo4j.close()
        impact = self._impact.analyze(change, impact_graph)

        invocation = None
        if use_ai:
            try:
                assessment, invocation = self._risk_agent.analyze(
                    intent,
                    change,
                    impact,
                    artifacts.graph,
                    artifacts.knowledge_documents,
                )
            except Exception as error:
                if not self._config.analysis.fail_closed_on_ai_error:
                    raise
                assessment = fail_closed_assessment(f"AI deployment analysis failed: {type(error).__name__}")
        else:
            assessment = fail_closed_assessment(
                "AI deployment analysis was disabled; safety cannot be established."
            )

        report = DeploymentReportGenerator().create(change, impact, assessment)
        markdown_path, json_path = self._reports.put(report)
        if persist_metadata:
            metadata = self._metadata_factory()
            try:
                metadata.initialize()
                metadata.record_report(report, markdown_path, json_path)
            finally:
                metadata.close()
        return DeploymentResult(
            report=report,
            markdown_report_path=markdown_path,
            json_report_path=json_path,
            agent_invocation=invocation,
        )
