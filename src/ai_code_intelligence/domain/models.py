from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    """Immutable base model used for values that cross architectural boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class FileKind(StrEnum):
    """Safe content-routing classification for repository files."""

    SOURCE = "source"
    MANIFEST = "manifest"
    TSCONFIG = "tsconfig"
    ENVIRONMENT = "environment"
    README = "readme"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class NodeType(StrEnum):
    """Allowlisted knowledge graph node labels."""

    SERVICE = "Service"
    REPOSITORY = "Repository"
    DIRECTORY = "Directory"
    FILE = "File"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    INTERFACE = "Interface"
    ENDPOINT = "Endpoint"
    SOCKET_EVENT = "SocketEvent"
    ENVIRONMENT_VARIABLE = "EnvironmentVariable"
    DATABASE_TABLE = "DatabaseTable"
    API = "API"
    BUSINESS_RULE = "BusinessRule"
    VARIABLE = "Variable"
    CONSTANT = "Constant"
    ENUM = "Enum"


class RelationshipType(StrEnum):
    """Allowlisted directed knowledge graph relationship types."""

    BELONGS_TO = "BELONGS_TO"
    CONTAINS = "CONTAINS"
    CALLS = "CALLS"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    IMPORTS = "IMPORTS"
    EXPOSES = "EXPOSES"
    IMPLEMENTS = "IMPLEMENTS"
    INVOKES = "INVOKES"
    READS = "READS"
    WRITES = "WRITES"
    EMITS = "EMITS"
    LISTENS = "LISTENS"


class AgentRole(StrEnum):
    """Stable identifiers for every specialist in the agent team."""

    REPOSITORY_DISCOVERY = "repository-discovery"
    CODE_GRAPH = "code-graph"
    CODE_GRAPH_AUDIT = "code-graph-audit"
    DEPENDENCY_GRAPH = "dependency-graph"
    REPOSITORY_KNOWLEDGE = "repository-knowledge"
    CENTRAL_KNOWLEDGE = "central-knowledge"
    ARCHITECTURE_CHAT = "architecture-chat"
    DEPLOYMENT_RISK = "deployment-risk"


class RepositoryDefinition(FrozenModel):
    """Configured local repository and human-readable service identity."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    description: str | None = None


class ScannedDirectory(FrozenModel):
    """Repository-relative directory discovered by the inventory layer."""

    relative_path: str
    parent_path: str | None = None


class ScannedFile(FrozenModel):
    """Repository-relative file metadata without retained content."""

    absolute_path: str
    relative_path: str
    kind: FileKind
    extension: str
    size_bytes: int = Field(ge=0)
    content_sha256: str


class PackageManifestSummary(FrozenModel):
    """Typed subset of package.json relevant to architecture analysis."""

    name: str | None = None
    version: str | None = None
    description: str | None = None
    scripts: dict[str, str] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    dev_dependencies: dict[str, str] = Field(default_factory=dict)


class ScannedRepository(FrozenModel):
    """Complete read-only inventory for one configured repository."""

    definition: RepositoryDefinition
    root_path: str
    directories: tuple[ScannedDirectory, ...]
    files: tuple[ScannedFile, ...]
    package_manifest: PackageManifestSummary | None = None
    environment_variable_names: tuple[str, ...] = ()
    scanned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RelationshipEvidence(FrozenModel):
    """Exact source location and resolution metadata for a graph edge."""

    file_path: str
    start_line: int = Field(ge=1)
    start_column: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)
    end_column: int | None = Field(default=None, ge=1)
    analyzer: str
    confidence: Literal["high", "medium"]
    resolution: str
    excerpt: str | None = None


class GraphNode(FrozenModel):
    """Canonical graph entity with a stable content-derived identifier."""

    id: str
    type: NodeType
    name: str
    qualified_name: str
    repository_id: str | None = None
    file_path: str | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(FrozenModel):
    """Canonical directed graph edge with mandatory source evidence."""

    id: str
    type: RelationshipType
    source_id: str
    target_id: str
    repository_id: str | None = None
    evidence: RelationshipEvidence
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphStatistics(FrozenModel):
    """Materialized node and relationship counts for a graph snapshot."""

    node_count: int
    relationship_count: int
    nodes_by_type: dict[str, int]
    relationships_by_type: dict[str, int]


class KnowledgeGraph(FrozenModel):
    """Versioned, internally consistent knowledge graph snapshot."""

    schema_version: Literal[1] = 1
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]
    statistics: GraphStatistics

    @model_validator(mode="after")
    def relationships_reference_existing_nodes(self) -> KnowledgeGraph:
        node_ids = {node.id for node in self.nodes}
        dangling = [
            rel.id
            for rel in self.relationships
            if rel.source_id not in node_ids or rel.target_id not in node_ids
        ]
        if dangling:
            raise ValueError(f"relationships reference unknown nodes: {', '.join(dangling[:5])}")
        return self

    @classmethod
    def create(cls, nodes: list[GraphNode], relationships: list[GraphRelationship]) -> KnowledgeGraph:
        return cls(
            nodes=tuple(nodes),
            relationships=tuple(relationships),
            statistics=GraphStatistics(
                node_count=len(nodes),
                relationship_count=len(relationships),
                nodes_by_type=dict(Counter(node.type.value for node in nodes)),
                relationships_by_type=dict(Counter(rel.type.value for rel in relationships)),
            ),
        )


class AgentInvocation(FrozenModel):
    """Token, latency, model, repository, and batch audit record."""

    role: AgentRole
    invocation_id: str
    model_id: str
    repository_id: str | None = None
    batch_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0
    cache_hit: bool = False


class AgentDescriptor(FrozenModel):
    """Public description of a specialist agent and its responsibility."""

    role: AgentRole
    name: str
    responsibility: str


AGENT_DESCRIPTORS: tuple[AgentDescriptor, ...] = (
    AgentDescriptor(
        role=AgentRole.REPOSITORY_DISCOVERY,
        name="Repository Discovery Agent",
        responsibility="Profiles repository purpose, runtime, architecture, entry points, configuration, and risks.",
    ),
    AgentDescriptor(
        role=AgentRole.CODE_GRAPH,
        name="Code Graph Agent",
        responsibility="Extracts evidence-backed code entities and relationships from bounded source batches.",
    ),
    AgentDescriptor(
        role=AgentRole.CODE_GRAPH_AUDIT,
        name="Code Graph Audit Agent",
        responsibility="Finds semantic facts omitted by the first extraction pass.",
    ),
    AgentDescriptor(
        role=AgentRole.DEPENDENCY_GRAPH,
        name="Dependency Graph Agent",
        responsibility="Proposes cross-repository links that deterministic validators must approve.",
    ),
    AgentDescriptor(
        role=AgentRole.REPOSITORY_KNOWLEDGE,
        name="Repository Knowledge Agent",
        responsibility="Writes repository documentation from validated graph evidence.",
    ),
    AgentDescriptor(
        role=AgentRole.CENTRAL_KNOWLEDGE,
        name="Central Knowledge Agent",
        responsibility="Writes system-wide architecture documentation from validated evidence.",
    ),
    AgentDescriptor(
        role=AgentRole.ARCHITECTURE_CHAT,
        name="Architecture Chat Agent",
        responsibility="Answers grounded architecture questions from published knowledge.",
    ),
    AgentDescriptor(
        role=AgentRole.DEPLOYMENT_RISK,
        name="Deployment Risk Agent",
        responsibility="Assesses regressions and deployment safety after deterministic impact analysis.",
    ),
)
