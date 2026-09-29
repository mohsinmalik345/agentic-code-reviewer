from __future__ import annotations

import asyncio
import json
from typing import Any

from graphql import GraphQLSchema, build_schema

from ai_code_intelligence.api.service import PlatformService, UserInputError
from ai_code_intelligence.application.models import DeploymentResult, ScanResult
from ai_code_intelligence.domain.models import (
    AgentInvocation,
    GraphNode,
    GraphRelationship,
    GraphStatistics,
)
from ai_code_intelligence.domain.repositories import IndexingJob, IndexingJobEvent, RepositoryRecord
from ai_code_intelligence.jira.models import (
    JiraConnection,
    JiraConnectionTestResult,
    JiraProjectMapping,
)
from ai_code_intelligence.knowledge.chat import (
    ArchitectureChatAnswer,
    ArchitectureChatMessage,
)

TYPE_DEFINITIONS = """
type Health {
  status: String!
  indexingMode: String!
  modelId: String!
  chatModelId: String!
  runtime: String!
}
type AgentDescriptor {
  role: String!
  name: String!
  responsibility: String!
}
type AgentInvocation {
  role: String!
  invocationId: ID!
  modelId: String!
  repositoryId: String
  batchId: String
  inputTokens: Int!
  outputTokens: Int!
  totalTokens: Int!
  latencyMs: Float!
}
type Count { key: String!, value: Int! }
type GraphStatistics {
  nodeCount: Int!
  relationshipCount: Int!
  nodesByType: [Count!]!
  relationshipsByType: [Count!]!
}
type Repository {
  id: ID!
  name: String!
  description: String
  provider: String!
  status: String!
  gitlabUrl: String
  ref: String
  exactCommit: String
  knowledgeBaseUri: String
  graphNodeCount: Int!
  graphRelationshipCount: Int!
  lastScanRunId: ID
  createdAt: String!
  updatedAt: String!
  indexedAt: String
}
type IndexingJob {
  id: ID!
  requestedRepositoryId: ID
  status: String!
  stage: String!
  scanRunId: ID
  errorCode: String
  createdAt: String!
  updatedAt: String!
  startedAt: String
  completedAt: String
  cancellationRequestedAt: String
  canCancel: Boolean!
}
type IndexingJobEvent {
  id: ID!
  jobId: ID!
  level: String!
  stage: String!
  code: String!
  message: String!
  errorType: String
  createdAt: String!
}
type ArchitectureChatSource {
  id: ID!
  title: String!
  repositoryId: ID
  section: String!
  sourceUri: String!
  evidenceKind: String!
}
type ArchitectureChatAnswer {
  answerMarkdown: String!
  sources: [ArchitectureChatSource!]!
  limitations: [String!]!
  invocation: AgentInvocation!
}
type JiraConnection {
  id: ID!
  name: String!
  edition: String!
  baseUrl: String!
  authType: String!
  username: String
  credentialEnv: String!
  credentialConfigured: Boolean!
  enabled: Boolean!
  createdAt: String!
  updatedAt: String!
}
type JiraProjectMapping {
  id: ID!
  connectionId: ID!
  repositoryId: ID!
  jiraProjectKey: String!
  acceptanceCriteriaFields: [String!]!
  issueKeyPattern: String!
  createdAt: String!
  updatedAt: String!
}
type JiraConnectionTest {
  ok: Boolean!
  message: String!
  serverTitle: String
  serverVersion: String
  authenticatedUser: String
}
type OnboardRepositoryPayload {
  repository: Repository!
  job: IndexingJob!
}
type OnboardRepositoriesPayload {
  repositories: [Repository!]!
  job: IndexingJob!
}
type GraphSnapshot {
  available: Boolean!
  source: String!
  generatedAt: String
  repositoryIds: [ID!]!
  statistics: GraphStatistics
}
type ScanResult {
  scanRunId: ID!
  indexingMode: String!
  repositoryIds: [String!]!
  graphPath: String!
  statistics: GraphStatistics!
  knowledgeUris: [String!]!
  embeddedChunkCount: Int!
  agentInvocations: [AgentInvocation!]!
  evidenceWarnings: [String!]!
  neo4jSnapshotId: String
}
type GraphNode {
  id: ID!
  type: String!
  name: String!
  qualifiedName: String!
  repositoryId: String
  filePath: String
  startLine: Int
  endLine: Int
  metadataJson: String!
}
type RelationshipEvidence {
  filePath: String!
  startLine: Int!
  startColumn: Int!
  endLine: Int
  endColumn: Int
  analyzer: String!
  confidence: String!
  resolution: String!
  excerpt: String
}
type GraphRelationship {
  id: ID!
  type: String!
  sourceId: ID!
  targetId: ID!
  repositoryId: String
  evidence: RelationshipEvidence!
  metadataJson: String!
}
type DeploymentResult {
  reportId: ID!
  recommendation: String!
  riskScore: Int!
  reasoning: [String!]!
  changedFilePaths: [String!]!
  affectedServiceIds: [ID!]!
  affectedEndpointIds: [ID!]!
  markdownReportPath: String!
  jsonReportPath: String!
}
input ScanInput {
  persistNeo4j: Boolean = false
  indexEmbeddings: Boolean = false
  persistMetadata: Boolean = false
}
input DeploymentInput {
  repositoryId: String!
  baseRevision: String
  headRevision: String
  staged: Boolean = false
  intent: String
  useAi: Boolean = true
  useNeo4jImpact: Boolean = false
  refreshIndex: Boolean = true
  persistNeo4j: Boolean = false
  indexEmbeddings: Boolean = false
  persistMetadata: Boolean = false
}
input OnboardRepositoryInput {
  gitlabUrl: String!
  repositoryId: ID
  name: String
  description: String
  ref: String
}
input OnboardRepositoriesInput {
  repositories: [OnboardRepositoryInput!]!
}
input ReindexInput {
  repositoryId: ID
}
input CancelIndexingJobInput {
  jobId: ID!
}
input ArchitectureChatMessageInput {
  role: String!
  content: String!
}
input ArchitectureChatInput {
  question: String!
  repositoryId: ID
  history: [ArchitectureChatMessageInput!] = []
}
input JiraConnectionInput {
  id: ID!
  name: String!
  edition: String!
  baseUrl: String!
  authType: String!
  username: String
  credentialEnv: String!
  enabled: Boolean = true
}
input JiraConnectionTestInput {
  connectionId: ID!
}
input DeleteJiraConnectionInput {
  connectionId: ID!
}
input JiraProjectMappingInput {
  id: ID
  connectionId: ID!
  repositoryId: ID!
  jiraProjectKey: String!
  acceptanceCriteriaFields: [String!]! = []
  issueKeyPattern: String!
}
input DeleteJiraProjectMappingInput {
  mappingId: ID!
}
type Query {
  health: Health!
  agents: [AgentDescriptor!]!
  repositories(includeDisabled: Boolean = false): [Repository!]!
  repository(id: ID!): Repository
  indexingJob(id: ID!): IndexingJob
  indexingJobs(status: String, limit: Int = 50): [IndexingJob!]!
  indexingJobEvents(jobId: ID!, limit: Int = 200): [IndexingJobEvent!]!
  graphSnapshot: GraphSnapshot!
  latestScan: ScanResult
  graphNodes(repositoryId: String, type: String, limit: Int = 100): [GraphNode!]!
  graphRelationships(repositoryId: String, type: String, limit: Int = 100): [GraphRelationship!]!
  jiraConnections(includeDisabled: Boolean = false): [JiraConnection!]!
  jiraProjectMappings(repositoryId: ID): [JiraProjectMapping!]!
}
type Mutation {
  onboardRepository(input: OnboardRepositoryInput!): OnboardRepositoryPayload!
  onboardRepositories(input: OnboardRepositoriesInput!): OnboardRepositoriesPayload!
  reindex(input: ReindexInput!): IndexingJob!
  cancelIndexingJob(input: CancelIndexingJobInput!): IndexingJob!
  askArchitecture(input: ArchitectureChatInput!): ArchitectureChatAnswer!
  scan(input: ScanInput!): ScanResult!
  analyzeDeployment(input: DeploymentInput!): DeploymentResult!
  saveJiraConnection(input: JiraConnectionInput!): JiraConnection!
  testJiraConnection(input: JiraConnectionTestInput!): JiraConnectionTest!
  deleteJiraConnection(input: DeleteJiraConnectionInput!): Boolean!
  saveJiraProjectMapping(input: JiraProjectMappingInput!): JiraProjectMapping!
  deleteJiraProjectMapping(input: DeleteJiraProjectMappingInput!): Boolean!
}
"""


def create_schema(service: PlatformService) -> GraphQLSchema:
    """Build a transport-only schema whose resolvers delegate to PlatformService."""

    schema = build_schema(TYPE_DEFINITIONS)
    query = schema.get_type("Query")
    mutation = schema.get_type("Mutation")
    assert query is not None and mutation is not None
    query.fields["health"].resolve = lambda _root, _info: service.health()  # type: ignore[attr-defined]
    query.fields["agents"].resolve = lambda _root, _info: [  # type: ignore[attr-defined]
        {
            "role": item.role.value,
            "name": item.name,
            "responsibility": item.responsibility,
        }
        for item in service.agents()
    ]
    query.fields["latestScan"].resolve = lambda _root, _info: _scan(service.latest_scan())  # type: ignore[attr-defined]
    query.fields["repositories"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, includeDisabled=False: [
            _repository(item) for item in service.repositories(include_disabled=includeDisabled)
        ]
    )
    query.fields["repository"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, id: _optional_repository(service.repository(id))
    )
    query.fields["indexingJob"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, id: _optional_job(service.indexing_job(id))
    )
    query.fields["indexingJobs"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, status=None, limit=50: [
            _job(item) for item in service.indexing_jobs(status=status, limit=limit)
        ]
    )
    query.fields["indexingJobEvents"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, jobId, limit=200: [
            _job_event(item) for item in service.indexing_job_events(jobId, limit=limit)
        ]
    )
    query.fields["graphSnapshot"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info: _graph_snapshot(service.graph_snapshot())
    )
    query.fields["graphNodes"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, repositoryId=None, type=None, limit=100: [
            _node(item) for item in service.graph_nodes(repositoryId, type, limit)
        ]
    )
    query.fields["graphRelationships"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, repositoryId=None, type=None, limit=100: [
            _relationship(item) for item in service.graph_relationships(repositoryId, type, limit)
        ]
    )
    query.fields["jiraConnections"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, includeDisabled=False: [
            _jira_connection(item, service.jira_credential_configured(item))
            for item in service.jira_connections(include_disabled=includeDisabled)
        ]
    )
    query.fields["jiraProjectMappings"].resolve = (  # type: ignore[attr-defined]
        lambda _root, _info, repositoryId=None: [
            _jira_mapping(item)
            for item in service.jira_project_mappings(repository_id=repositoryId)
        ]
    )

    async def scan(_root: object, _info: object, input: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            service.scan,
            persist_neo4j=input.get("persistNeo4j", False),
            index_embeddings=input.get("indexEmbeddings", False),
            persist_metadata=input.get("persistMetadata", False),
        )
        mapped = _scan(result)
        assert mapped is not None
        return mapped

    async def onboard(_root: object, _info: object, input: dict[str, Any]) -> dict[str, Any]:
        repository, job = await asyncio.to_thread(
            service.onboard_repository,
            input["gitlabUrl"],
            repository_id=input.get("repositoryId"),
            name=input.get("name"),
            description=input.get("description"),
            ref=input.get("ref"),
        )
        return {"repository": _repository(repository), "job": _job(job)}

    async def onboard_many(_root: object, _info: object, input: dict[str, Any]) -> dict[str, Any]:
        repositories, job = await asyncio.to_thread(
            service.onboard_repositories,
            tuple(input["repositories"]),
        )
        return {
            "repositories": [_repository(repository) for repository in repositories],
            "job": _job(job),
        }

    async def reindex(_root: object, _info: object, input: dict[str, Any]) -> dict[str, Any]:
        job = await asyncio.to_thread(service.reindex, input.get("repositoryId"))
        return _job(job)

    async def cancel_indexing_job(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> dict[str, Any]:
        job = await asyncio.to_thread(service.cancel_indexing_job, input["jobId"])
        return _job(job)

    async def ask_architecture(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            history = tuple(ArchitectureChatMessage.model_validate(item) for item in input.get("history", []))
        except ValueError as error:
            raise UserInputError("architecture chat history is invalid") from error
        answer = await asyncio.to_thread(
            service.ask_architecture,
            input["question"],
            repository_id=input.get("repositoryId"),
            history=history,
        )
        return _architecture_chat_answer(answer)

    async def analyze(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            service.analyze_deployment,
            input["repositoryId"],
            base_revision=input.get("baseRevision"),
            head_revision=input.get("headRevision"),
            staged=input.get("staged", False),
            intent=input.get("intent"),
            use_ai=input.get("useAi", True),
            use_neo4j_impact=input.get("useNeo4jImpact", False),
            refresh_index=input.get("refreshIndex", True),
            persist_neo4j=input.get("persistNeo4j", False),
            index_embeddings=input.get("indexEmbeddings", False),
            persist_metadata=input.get("persistMetadata", False),
        )
        return _deployment(result)

    async def save_jira_connection(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> dict[str, Any]:
        connection = await asyncio.to_thread(service.save_jira_connection, input)
        return _jira_connection(connection, service.jira_credential_configured(connection))

    async def test_jira_connection(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            service.test_jira_connection,
            input["connectionId"],
        )
        return _jira_connection_test(result)

    async def delete_jira_connection(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> bool:
        return await asyncio.to_thread(
            service.delete_jira_connection,
            input["connectionId"],
        )

    async def save_jira_mapping(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> dict[str, Any]:
        return _jira_mapping(
            await asyncio.to_thread(service.save_jira_project_mapping, input)
        )

    async def delete_jira_mapping(
        _root: object,
        _info: object,
        input: dict[str, Any],
    ) -> bool:
        return await asyncio.to_thread(
            service.delete_jira_project_mapping,
            input["mappingId"],
        )

    mutation.fields["onboardRepository"].resolve = onboard  # type: ignore[attr-defined]
    mutation.fields["onboardRepositories"].resolve = onboard_many  # type: ignore[attr-defined]
    mutation.fields["reindex"].resolve = reindex  # type: ignore[attr-defined]
    mutation.fields["cancelIndexingJob"].resolve = cancel_indexing_job  # type: ignore[attr-defined]
    mutation.fields["askArchitecture"].resolve = ask_architecture  # type: ignore[attr-defined]
    mutation.fields["scan"].resolve = scan  # type: ignore[attr-defined]
    mutation.fields["analyzeDeployment"].resolve = analyze  # type: ignore[attr-defined]
    mutation.fields["saveJiraConnection"].resolve = save_jira_connection  # type: ignore[attr-defined]
    mutation.fields["testJiraConnection"].resolve = test_jira_connection  # type: ignore[attr-defined]
    mutation.fields["deleteJiraConnection"].resolve = delete_jira_connection  # type: ignore[attr-defined]
    mutation.fields["saveJiraProjectMapping"].resolve = save_jira_mapping  # type: ignore[attr-defined]
    mutation.fields["deleteJiraProjectMapping"].resolve = delete_jira_mapping  # type: ignore[attr-defined]
    return schema


def _scan(result: ScanResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "scanRunId": result.scan_run_id,
        "indexingMode": result.indexing_mode,
        "repositoryIds": result.repository_ids,
        "graphPath": result.graph_path,
        "statistics": {
            "nodeCount": result.statistics.node_count,
            "relationshipCount": result.statistics.relationship_count,
            "nodesByType": [
                {"key": key, "value": value} for key, value in sorted(result.statistics.nodes_by_type.items())
            ],
            "relationshipsByType": [
                {"key": key, "value": value}
                for key, value in sorted(result.statistics.relationships_by_type.items())
            ],
        },
        "knowledgeUris": result.knowledge_uris,
        "embeddedChunkCount": result.embedded_chunk_count,
        "agentInvocations": [
            {
                "role": item.role.value,
                "invocationId": item.invocation_id,
                "modelId": item.model_id,
                "repositoryId": item.repository_id,
                "batchId": item.batch_id,
                "inputTokens": item.input_tokens,
                "outputTokens": item.output_tokens,
                "totalTokens": item.total_tokens,
                "latencyMs": item.latency_ms,
            }
            for item in result.agent_invocations
        ],
        "evidenceWarnings": result.evidence_warnings,
        "neo4jSnapshotId": result.neo4j_snapshot_id,
    }


def _repository(record: RepositoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "provider": record.provider.value,
        "status": record.status.value,
        "gitlabUrl": record.canonical_url,
        "ref": record.canonical_ref,
        "exactCommit": record.exact_commit,
        "knowledgeBaseUri": record.knowledge_base_uri,
        "graphNodeCount": record.graph_node_count,
        "graphRelationshipCount": record.graph_relationship_count,
        "lastScanRunId": record.last_scan_run_id,
        "createdAt": record.created_at.isoformat(),
        "updatedAt": record.updated_at.isoformat(),
        "indexedAt": record.indexed_at.isoformat() if record.indexed_at else None,
    }


def _optional_repository(record: RepositoryRecord | None) -> dict[str, Any] | None:
    return _repository(record) if record is not None else None


def _job(job: IndexingJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "requestedRepositoryId": job.requested_repository_id,
        "status": job.status.value,
        "stage": job.stage.value,
        "scanRunId": job.scan_run_id,
        "errorCode": job.error_code,
        "createdAt": job.created_at.isoformat(),
        "updatedAt": job.updated_at.isoformat(),
        "startedAt": job.started_at.isoformat() if job.started_at else None,
        "completedAt": job.completed_at.isoformat() if job.completed_at else None,
        "cancellationRequestedAt": (
            job.cancellation_requested_at.isoformat() if job.cancellation_requested_at else None
        ),
        "canCancel": job.can_cancel,
    }


def _optional_job(job: IndexingJob | None) -> dict[str, Any] | None:
    return _job(job) if job is not None else None


def _job_event(event: IndexingJobEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "jobId": event.job_id,
        "level": event.level.value,
        "stage": event.stage.value,
        "code": event.code,
        "message": event.message,
        "errorType": event.error_type,
        "createdAt": event.created_at.isoformat(),
    }


def _architecture_chat_answer(answer: ArchitectureChatAnswer) -> dict[str, Any]:
    return {
        "answerMarkdown": answer.answer_markdown,
        "sources": [
            {
                "id": source.id,
                "title": source.title,
                "repositoryId": source.repository_id,
                "section": source.section,
                "sourceUri": source.source_uri,
                "evidenceKind": source.evidence_kind,
            }
            for source in answer.sources
        ],
        "limitations": answer.limitations,
        "invocation": _agent_invocation(answer.invocation),
    }


def _jira_connection(
    connection: JiraConnection,
    credential_configured: bool,
) -> dict[str, Any]:
    return {
        "id": connection.id,
        "name": connection.name,
        "edition": connection.edition.value,
        "baseUrl": connection.base_url,
        "authType": connection.auth_type.value,
        "username": connection.username,
        "credentialEnv": connection.credential_env,
        "credentialConfigured": credential_configured,
        "enabled": connection.enabled,
        "createdAt": connection.created_at.isoformat(),
        "updatedAt": connection.updated_at.isoformat(),
    }


def _jira_mapping(mapping: JiraProjectMapping) -> dict[str, Any]:
    return {
        "id": mapping.id,
        "connectionId": mapping.connection_id,
        "repositoryId": mapping.repository_id,
        "jiraProjectKey": mapping.jira_project_key,
        "acceptanceCriteriaFields": mapping.acceptance_criteria_fields,
        "issueKeyPattern": mapping.issue_key_pattern,
        "createdAt": mapping.created_at.isoformat(),
        "updatedAt": mapping.updated_at.isoformat(),
    }


def _jira_connection_test(result: JiraConnectionTestResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "message": result.message,
        "serverTitle": result.server_title,
        "serverVersion": result.server_version,
        "authenticatedUser": result.authenticated_user,
    }


def _agent_invocation(invocation: AgentInvocation) -> dict[str, Any]:
    return {
        "role": invocation.role.value,
        "invocationId": invocation.invocation_id,
        "modelId": invocation.model_id,
        "repositoryId": invocation.repository_id,
        "batchId": invocation.batch_id,
        "inputTokens": invocation.input_tokens,
        "outputTokens": invocation.output_tokens,
        "totalTokens": invocation.total_tokens,
        "latencyMs": invocation.latency_ms,
    }


def _graph_snapshot(snapshot: dict[str, object]) -> dict[str, Any]:
    statistics = snapshot.get("statistics")
    if statistics is not None and not isinstance(statistics, GraphStatistics):
        raise TypeError("graph snapshot statistics are invalid")
    return {
        "available": snapshot["available"],
        "source": snapshot["source"],
        "generatedAt": snapshot["generatedAt"],
        "repositoryIds": snapshot["repositoryIds"],
        "statistics": (
            {
                "nodeCount": statistics.node_count,
                "relationshipCount": statistics.relationship_count,
                "nodesByType": [
                    {"key": key, "value": value} for key, value in sorted(statistics.nodes_by_type.items())
                ],
                "relationshipsByType": [
                    {"key": key, "value": value}
                    for key, value in sorted(statistics.relationships_by_type.items())
                ],
            }
            if statistics is not None
            else None
        ),
    }


def _node(node: GraphNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type.value,
        "name": node.name,
        "qualifiedName": node.qualified_name,
        "repositoryId": node.repository_id,
        "filePath": node.file_path,
        "startLine": node.start_line,
        "endLine": node.end_line,
        "metadataJson": json.dumps(node.metadata, separators=(",", ":"), default=str),
    }


def _relationship(relationship: GraphRelationship) -> dict[str, Any]:
    evidence = relationship.evidence
    return {
        "id": relationship.id,
        "type": relationship.type.value,
        "sourceId": relationship.source_id,
        "targetId": relationship.target_id,
        "repositoryId": relationship.repository_id,
        "evidence": {
            "filePath": evidence.file_path,
            "startLine": evidence.start_line,
            "startColumn": evidence.start_column,
            "endLine": evidence.end_line,
            "endColumn": evidence.end_column,
            "analyzer": evidence.analyzer,
            "confidence": evidence.confidence,
            "resolution": evidence.resolution,
            "excerpt": evidence.excerpt,
        },
        "metadataJson": json.dumps(relationship.metadata, separators=(",", ":"), default=str),
    }


def _deployment(result: DeploymentResult) -> dict[str, Any]:
    report = result.report
    return {
        "reportId": report.id,
        "recommendation": report.recommendation,
        "riskScore": report.risk_score,
        "reasoning": report.reasoning,
        "changedFilePaths": [file.path for file in report.change.changed_files],
        "affectedServiceIds": [node.id for node in report.impact.affected_services],
        "affectedEndpointIds": [node.id for node in report.impact.affected_endpoints],
        "markdownReportPath": result.markdown_report_path,
        "jsonReportPath": result.json_report_path,
    }
