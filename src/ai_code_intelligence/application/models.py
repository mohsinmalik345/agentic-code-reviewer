from __future__ import annotations

from ai_code_intelligence.domain.analysis import DeploymentReport
from ai_code_intelligence.domain.models import AgentInvocation, FrozenModel, GraphStatistics


class ScanResult(FrozenModel):
    """Public result of one complete multi-repository scan."""

    scan_run_id: str
    indexing_mode: str = "bedrock-agents"
    repository_ids: tuple[str, ...]
    graph_path: str
    statistics: GraphStatistics
    knowledge_uris: tuple[str, ...]
    embedded_chunk_count: int
    agent_invocations: tuple[AgentInvocation, ...]
    evidence_warnings: tuple[str, ...]
    neo4j_snapshot_id: str | None = None


class DeploymentResult(FrozenModel):
    """Deployment report plus persisted artifact locations and telemetry."""

    report: DeploymentReport
    markdown_report_path: str
    json_report_path: str
    agent_invocation: AgentInvocation | None = None
