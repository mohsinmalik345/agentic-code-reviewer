from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from ai_code_intelligence.domain.models import (
    AgentInvocation,
    FrozenModel,
    GraphRelationship,
    KnowledgeGraph,
    NodeType,
    RelationshipType,
)

SEMANTIC_NODE_TYPES = tuple(
    value
    for value in NodeType
    if value not in {NodeType.REPOSITORY, NodeType.SERVICE, NodeType.DIRECTORY, NodeType.FILE}
)
SEMANTIC_RELATIONSHIP_TYPES = tuple(
    value
    for value in RelationshipType
    if value not in {RelationshipType.BELONGS_TO, RelationshipType.CONTAINS}
)

_DYNAMIC_IDENTITY_MARKERS = ("${", "`", "+", "{{")
KNOWLEDGE_EVIDENCE_SUMMARY_MAX_CHARACTERS = 8_000


def has_dynamic_identity_markers(value: str) -> bool:
    """Report whether a route, URL, or event literal is interpolated or concatenated.

    Such a value cannot be matched exactly across repositories, so cross-service
    identity validation must reject it rather than guess the runtime value.
    """

    return any(marker in value for marker in _DYNAMIC_IDENTITY_MARKERS)


class AgentAttribute(FrozenModel):
    """Scalar semantic metadata emitted by a specialist agent."""

    key: str = Field(min_length=1)
    value: str | int | float | bool | None


class AgentNode(FrozenModel):
    """Evidence-addressed semantic node proposed by a code graph agent."""

    ref: str = Field(min_length=1)
    type: NodeType
    name: str = Field(min_length=1)
    qualified_name: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)
    evidence_excerpt: str = Field(min_length=1)
    attributes: tuple[AgentAttribute, ...] = ()

    @field_validator("type")
    @classmethod
    def only_semantic_nodes(cls, value: NodeType) -> NodeType:
        if value not in SEMANTIC_NODE_TYPES:
            raise ValueError(f"agent may not emit structural node type {value.value}")
        return value

    @classmethod
    def semantic_json_schema(cls) -> dict[str, object]:
        schema = cls.model_json_schema()
        type_schema = schema["properties"]["type"]
        if isinstance(type_schema, dict):
            type_schema.clear()
            type_schema.update({"type": "string", "enum": [item.value for item in SEMANTIC_NODE_TYPES]})
        return schema


class AgentRelationship(FrozenModel):
    """Evidence-addressed directed relationship proposed by an agent."""

    type: RelationshipType
    source_ref: str = Field(min_length=1)
    target_ref: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    start_column: int = Field(ge=1)
    confidence: Literal["high", "medium"]
    resolution: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    attributes: tuple[AgentAttribute, ...] = ()

    @field_validator("type")
    @classmethod
    def only_semantic_relationships(cls, value: RelationshipType) -> RelationshipType:
        if value not in SEMANTIC_RELATIONSHIP_TYPES:
            raise ValueError(f"agent may not emit structural relationship type {value.value}")
        return value

    @field_validator("resolution")
    @classmethod
    def resolution_must_contain_evidence(cls, value: str) -> str:
        """Reject empty reasoning without manufacturing a replacement explanation."""

        if not value.strip():
            raise ValueError("relationship resolution must contain non-whitespace evidence")
        return value


class CodeGraphOutput(FrozenModel):
    """Validated output envelope for code graph and audit specialists."""

    nodes: tuple[AgentNode, ...] = ()
    relationships: tuple[AgentRelationship, ...] = ()
    warnings: tuple[str, ...] = ()


class RepositoryProfile(FrozenModel):
    """Grounded operational profile produced before detailed extraction."""

    purpose: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    frameworks: tuple[str, ...] = ()
    entry_points: tuple[str, ...] = ()
    configuration_files: tuple[str, ...] = ()
    database_technologies: tuple[str, ...] = ()
    external_systems: tuple[str, ...] = ()
    deployment_notes: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    evidence_files: tuple[str, ...] = ()


class DependencyLink(FrozenModel):
    """Cross-repository link proposal referencing existing graph IDs."""

    type: Literal["INVOKES", "DEPENDS_ON"]
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    evidence_relationship_ids: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)


class DependencyGraphOutput(FrozenModel):
    """Dependency specialist proposals and non-fatal warnings."""

    links: tuple[DependencyLink, ...] = ()
    warnings: tuple[str, ...] = ()


class KnowledgeOutput(FrozenModel):
    """Markdown knowledge output with explicit graph citations."""

    title: str = Field(min_length=1)
    markdown: str = Field(min_length=1, max_length=75_000)
    evidence_node_ids: tuple[str, ...] = ()
    evidence_relationship_ids: tuple[str, ...] = ()


class KnowledgeEvidenceSummary(FrozenModel):
    """Bounded evidence memo used by knowledge map/reduce stages."""

    summary: str = Field(
        min_length=1,
        max_length=KNOWLEDGE_EVIDENCE_SUMMARY_MAX_CHARACTERS,
    )
    evidence_node_ids: tuple[str, ...] = ()
    evidence_relationship_ids: tuple[str, ...] = ()


class DeploymentRiskOutput(FrozenModel):
    """Structured AI deployment assessment constrained by graph IDs."""

    implementation_satisfies_intent: bool
    breakage_risk: Literal["low", "medium", "high", "critical"]
    hidden_side_effects: tuple[str, ...] = ()
    affected_service_ids: tuple[str, ...] = ()
    affected_endpoint_ids: tuple[str, ...] = ()
    regression_test_node_ids: tuple[str, ...] = ()
    risk_score: int = Field(ge=0, le=100)
    recommendation: Literal["PASS", "BLOCK"]
    reasoning: tuple[str, ...] = Field(min_length=1)
    assumptions: tuple[str, ...] = ()
    evidence_node_ids: tuple[str, ...] = ()
    evidence_relationship_ids: tuple[str, ...] = ()


class AgentGraphResult(FrozenModel):
    """Complete validated graph, profiles, telemetry, and warnings."""

    graph: KnowledgeGraph
    profiles: dict[str, RepositoryProfile]
    invocations: tuple[AgentInvocation, ...]
    evidence_warnings: tuple[str, ...]


class DependencyLinkResult(FrozenModel):
    """Graph after exact dependency validation plus agent telemetry."""

    graph: KnowledgeGraph
    invocation: AgentInvocation
    evidence_warnings: tuple[str, ...]


class CodeGraphBatchResult(FrozenModel):
    """One semantic batch result and the exact source ranges supplied."""

    output: CodeGraphOutput
    invocation: AgentInvocation
    batch_id: str
    supplied_ranges: tuple[tuple[str, int, int], ...]


def attributes_to_dict(attributes: tuple[AgentAttribute, ...]) -> dict[str, object]:
    """Convert unique agent attributes to JSON-compatible metadata."""

    result: dict[str, object] = {}
    for attribute in attributes:
        result[attribute.key] = attribute.value
    return result


def evidence_ids_exist(
    output: KnowledgeOutput | DeploymentRiskOutput,
    graph: KnowledgeGraph,
) -> tuple[set[str], set[str]]:
    """Return invalid node and relationship citations emitted by an agent."""

    node_ids = {node.id for node in graph.nodes}
    relationship_ids = {relationship.id for relationship in graph.relationships}
    return (
        set(output.evidence_node_ids).difference(node_ids),
        set(output.evidence_relationship_ids).difference(relationship_ids),
    )


def relationship_index(graph: KnowledgeGraph) -> dict[str, GraphRelationship]:
    return {relationship.id: relationship for relationship in graph.relationships}
