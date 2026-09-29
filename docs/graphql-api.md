# GraphQL API

Endpoint: `POST /graphql`. When `graphql.api_key_env` is configured, pass `x-api-key`.
Requests must use `Content-Type: application/json`.

## Browser application

The same service hosts the operations console at `GET /`. `POST /auth/login` exchanges the configured
API key for a signed, eight-hour `HttpOnly`, `SameSite=Strict` browser session. The returned CSRF token
must accompany browser GraphQL POSTs in `x-csrf-token`; direct automation continues to use
`x-api-key`.

Knowledge documents are not made public and no AWS credential or S3 presigned URL is exposed to the
browser. Authenticated clients read validated JSON documents through:

- `GET /api/knowledge/central`
- `GET /api/knowledge/repositories/{repository_id}`
- `GET /api/knowledge/human/central`
- `GET /api/knowledge/human/repositories/{repository_id}`

The `/human/` routes are the default console view and hide internal graph IDs. The original evidence
routes remain available for traceability. Reader editions are deterministic transformations of the
same evidence documents; no second model rewrites their claims. The console renders returned
Markdown using DOM text nodes rather than inserting untrusted HTML.

## Repository onboarding

```graphql
mutation AddRepository {
  onboardRepository(input: {
    gitlabUrl: "https://gitlab.company.com/beelinks/ticket-service.git"
    repositoryId: "ticket-service"
    name: "Beelinks Ticket Service"
    ref: "main"
  }) {
    repository { id provider status gitlabUrl ref }
    job { id status stage createdAt }
  }
}
```

The mutation validates and durably registers the repository but does not clone or call Bedrock in
the request. It returns a queued job.

For 1–50 repositories, use the atomic bulk mutation. Every URL, ref, derived or supplied repository
ID, and collision with the existing catalogue is preflighted before any record is written. Repository
records and the queued job are committed in one file write or PostgreSQL transaction. A failure
rejects the complete batch; a successful batch queues one portfolio job rather than one job per
repository. Canonically equivalent GitLab URLs and duplicate derived IDs are rejected with their
input positions.

```graphql
mutation AddRepositories {
  onboardRepositories(input: {
    repositories: [
      {
        gitlabUrl: "https://gitlab.company.com/beelinks/ticket-service.git"
        ref: "main"
      }
      {
        gitlabUrl: "https://gitlab.company.com/beelinks/reporting-service.git"
        repositoryId: "reporting-service"
        ref: "develop"
      }
    ]
  }) {
    repositories { id provider status gitlabUrl ref }
    job { id requestedRepositoryId status stage createdAt }
  }
}
```

A newly created bulk job has a null `requestedRepositoryId` because it represents one commit-aware
portfolio refresh. If any portfolio job is already queued but not yet claimed, new registrations
attach to it; in that case the existing job can retain an attribution ID from an earlier single-repository
request. A running job is never reused because it may already have materialized its repository set;
the bulk request queues a later job instead.

Poll either single or bulk onboarding jobs with:

```graphql
query Job($id: ID!) {
  indexingJob(id: $id) {
    id status stage scanRunId errorCode startedAt completedAt
    cancellationRequestedAt canCancel
  }
}
```

`SUCCEEDED` means the required repository updates, affected central knowledge, and unified Neo4j
graph were published. Unchanged commits reuse their published artifacts. `errorCode` and the durable
event timeline are sanitized. Read the timeline for queued, active, completed, cancelled, or failed
jobs with:

```graphql
query JobLog($id: ID!) {
  indexingJobEvents(jobId: $id, limit: 200) {
    level stage code message errorType createdAt
  }
}
```

Raw tracebacks remain in protected worker logs; credentials and internal storage errors are never
returned by GraphQL.

Cancel a queued job, or request cooperative cancellation of materialization/scanning, with:

```graphql
mutation CancelJob($id: ID!) {
  cancelIndexingJob(input: {jobId: $id}) {
    id status stage cancellationRequestedAt canCancel completedAt
  }
}
```

Queued work becomes `CANCELLED` atomically and cannot be claimed. Running work remains `RUNNING`
with `cancellationRequestedAt` until its current Git/Bedrock step returns; the worker then restores
repository catalogue state and records `JOB_CANCELLED`. Cancellation is rejected once publication
starts so a caller cannot interrupt Neo4j, metadata, and manifest publication midway.

## Architecture chat

```graphql
mutation AskArchitecture {
  askArchitecture(input: {
    question: "How do the email and socket services communicate?"
    history: []
  }) {
    answerMarkdown
    sources { id title repositoryId section evidenceKind }
    limitations
    invocation { modelId inputTokens outputTokens latencyMs }
  }
}
```

Set `repositoryId` to restrict retrieval to that repository plus the central architecture. When it
is omitted, retrieval considers every published repository reader edition plus the central reader
edition. It also searches a bounded one-hop neighborhood in the exact Neo4j snapshot named by the
manifest. Conversation history is supplied by the caller, bounded by configuration, and kept only in
browser memory by the bundled UI. GPT-OSS receives selected reader excerpts (`S` labels) and
human-readable graph facts (`G` labels); `evidenceKind` identifies their provenance. The server
rejects citations to any source label that was not supplied in that request and requires a graph
citation whenever graph evidence is available.

## Repository catalogue and persisted graph

```graphql
query PlatformState {
  repositories {
    id name provider status exactCommit knowledgeBaseUri
    graphNodeCount graphRelationshipCount indexedAt
  }
  graphSnapshot {
    available source generatedAt repositoryIds
    statistics { nodeCount relationshipCount }
  }
}
```

Use `reindex(input: {})` to resolve the portfolio commits and queue an update without adding a
repository. Repositories whose resolved commits still match the published manifest and graph are
reused.

## Jira configuration

Jira GraphQL operations persist credential-free metadata only. Set `JIRA_ALLOWED_HOSTS` and the
credential environment variable on the server before saving/testing a connection; never send a
token through GraphQL. Jira Cloud requires `api_token` plus an email username. Jira Data Center
requires `personal_access_token` and a null/omitted username.

Create or update a Jira Cloud connection:

```graphql
mutation SaveJiraConnection {
  saveJiraConnection(input: {
    id: "engineering-jira"
    name: "Engineering Jira"
    edition: "cloud"
    baseUrl: "https://company.atlassian.net"
    authType: "api_token"
    username: "developer@company.com"
    credentialEnv: "JIRA_CLOUD_API_TOKEN"
    enabled: true
  }) {
    id name edition baseUrl authType username credentialEnv
    credentialConfigured enabled updatedAt
  }
}
```

`credentialConfigured` reports only whether that server environment variable is non-empty. Test the
saved connection with a read-only `serverInfo`/`myself` probe:

```graphql
mutation TestJiraConnection {
  testJiraConnection(input: {connectionId: "engineering-jira"}) {
    ok message serverTitle serverVersion authenticatedUser
  }
}
```

Map an indexed repository to one project:

```graphql
mutation SaveJiraMapping {
  saveJiraProjectMapping(input: {
    connectionId: "engineering-jira"
    repositoryId: "email-service"
    jiraProjectKey: "EMAIL"
    acceptanceCriteriaFields: ["description", "customfield_10042"]
    issueKeyPattern: "^[A-Z][A-Z0-9_]*-[0-9]+$"
  }) {
    id connectionId repositoryId jiraProjectKey
    acceptanceCriteriaFields issueKeyPattern updatedAt
  }
}
```

Read all connection metadata (including disabled connections) and mappings:

```graphql
query JiraConfiguration {
  jiraConnections(includeDisabled: true) {
    id name edition baseUrl credentialEnv credentialConfigured enabled
  }
  jiraProjectMappings {
    id connectionId repositoryId jiraProjectKey
    acceptanceCriteriaFields issueKeyPattern
  }
}
```

`deleteJiraProjectMapping(input: {mappingId: "..."})` deletes one mapping.
`deleteJiraConnection(input: {connectionId: "..."})` deletes a connection and its mappings. Both
return `Boolean!`. Browser calls use the same authenticated session and CSRF rules as other GraphQL
writes; automation calls use `x-api-key`.

## Scan

```graphql
mutation Scan {
  scan(input: {
    persistNeo4j: true
    persistMetadata: true
    indexEmbeddings: false
  }) {
    scanRunId
    repositoryIds
    statistics { nodeCount relationshipCount }
    knowledgeUris
    evidenceWarnings
  }
}
```

## Deployment analysis

```graphql
mutation Analyze {
  analyzeDeployment(input: {
    repositoryId: "email-service"
    baseRevision: "origin/main"
    headRevision: "HEAD"
    intent: "Add bounded email delivery retries"
    refreshIndex: true
    useAi: true
  }) {
    recommendation
    riskScore
    changedFilePaths
    affectedServiceIds
    affectedEndpointIds
    reasoning
    markdownReportPath
  }
}
```

## Graph exploration

```graphql
query Explore {
  graphNodes(repositoryId: "socket-service", type: "SocketEvent", limit: 100) {
    id name filePath startLine metadataJson
  }
  graphRelationships(repositoryId: "socket-service", type: "EMITS", limit: 100) {
    id sourceId targetId evidence { filePath startLine excerpt }
  }
}
```

Graph exploration hydrates from the latest validated local snapshot and falls back to Neo4j after a
restart. The GraphQL API is the transport; Neo4j is the graph database, PostgreSQL stores the
repository/job audit state, and S3 stores knowledge documents.

