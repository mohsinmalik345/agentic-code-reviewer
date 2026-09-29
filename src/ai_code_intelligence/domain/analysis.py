from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from ai_code_intelligence.domain.change import CodeChangeAnalysis
from ai_code_intelligence.domain.models import (
    FrozenModel,
    GraphNode,
    GraphRelationship,
)


class GraphPath(FrozenModel):
    """Ordered node and relationship IDs for one impact path."""

    node_ids: tuple[str, ...]
    relationship_ids: tuple[str, ...]


class RiskSignal(FrozenModel):
    """Deterministic deployment risk with graph evidence."""

    code: str
    severity: Literal["low", "medium", "high", "critical"]
    reason: str
    evidence_node_ids: tuple[str, ...] = ()


class ImpactAnalysis(FrozenModel):
    """Bounded deterministic blast radius and regression guidance."""

    directly_affected: tuple[GraphNode, ...]
    indirectly_affected: tuple[GraphNode, ...]
    directly_affected_functions: tuple[GraphNode, ...]
    indirectly_affected_functions: tuple[GraphNode, ...]
    affected_endpoints: tuple[GraphNode, ...]
    affected_rest_calls: tuple[GraphNode, ...]
    affected_socket_events: tuple[GraphNode, ...]
    affected_services: tuple[GraphNode, ...]
    dependency_paths: tuple[GraphPath, ...]
    relationships: tuple[GraphRelationship, ...]
    deterministic_risk_signals: tuple[RiskSignal, ...]
    potential_regressions: tuple[str, ...]
    traversal_truncated: bool


class DeploymentReport(FrozenModel):
    """Auditable final deployment gate combining deterministic and AI results."""

    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recommendation: Literal["PASS", "BLOCK"]
    risk_score: int = Field(ge=0, le=100)
    implementation_satisfies_intent: bool
    breakage_risk: Literal["low", "medium", "high", "critical"]
    hidden_side_effects: tuple[str, ...]
    regression_test_node_ids: tuple[str, ...]
    reasoning: tuple[str, ...]
    assumptions: tuple[str, ...]
    change: CodeChangeAnalysis
    impact: ImpactAnalysis
