# PostgreSQL schema

The default schema is `code_intelligence`. `code-intel migrate` creates it idempotently;
[001_initial.sql](../migrations/001_initial.sql) and
[002_repository_registry.sql](../migrations/002_repository_registry.sql), and
[003_indexing_job_events.sql](../migrations/003_indexing_job_events.sql), and
[004_indexing_job_cancellation.sql](../migrations/004_indexing_job_cancellation.sql), and
[005_jira_configuration.sql](../migrations/005_jira_configuration.sql) are mounted by Docker
Compose.

The private Codex-facing self-index uses the same table model in the separate
`code_intelligence_internal` schema. No cross-schema foreign keys, views, embedding rows, scan
audits, or repository registrations are created. Sharing the PostgreSQL server does not merge the
two publication domains.

| Table | Purpose | Key fields |
|---|---|---|
| `scan_runs` | One auditable indexing run | repository IDs, graph path/statistics, warnings, Neo4j snapshot |
| `agent_invocations` | Bedrock cost/latency audit | role, model, repository/batch, token counts, latency |
| `knowledge_documents` | S3/local object catalogue | kind, repository, URI, SHA-256, size |
| `embeddings` | pgvector chunks | source, content, metadata, model ID, `vector(1024)` |
| `deployment_reports` | Immutable gate outcome | repository, PASS/BLOCK, score, full JSON report |
| `repositories` | Durable local/GitLab catalogue | provider, credential-free URL, managed path, ref, exact commit, status, KB/graph metadata |
| `indexing_jobs` | Durable portfolio work queue | requested repository, status/stage, scan ID, safe error code, cancellation request, timestamps, lease |
| `indexing_job_events` | Sanitized durable job timeline | severity, stage, event code/message, safe error type, timestamp |
| `jira_connections` | Credential-free Jira instance configuration | edition, HTTPS base URL, auth type, optional Cloud email, credential environment-variable name, enabled flag |
| `jira_project_mappings` | Repository-to-Jira-project configuration | connection/repository IDs, project key, acceptance-criteria fields, issue-key pattern |

The embeddings primary key is deterministic from document content and chunk position. Re-indexing updates identical chunks rather than duplicating them. The HNSW cosine index supports future retrieval.

Only one `indexing_jobs` row can be actively processed by the worker policy at a time. PostgreSQL
claims work with row locking, expires abandoned leases, and keeps caller-visible errors to constrained
codes. GitLab credentials are never stored. `repositories.exact_commit` accepts only full SHA-1 or
SHA-256 object IDs.

`indexing_jobs.status` includes `queued`, `running`, `succeeded`, `failed`, and `cancelled`.
`cancellation_requested_at` separates a cooperative stop request from its acknowledgement: queued
jobs transition directly to `cancelled`, while active materialization/scanning jobs remain leased
until the worker reaches a safe boundary. Publication-stage jobs reject cancellation.

Job events are append-only and ordered by an identity sequence. New jobs record queue, materialize,
scan, publish, success, and failure events. Migration backfill adds a clearly labelled legacy event
for failures created before this table existed; expired worker leases also receive their own event.

Jira secret values are not stored. `jira_connections.credential_env` is only the name of an
environment variable resolved by the application at request time. Connection deletion cascades to
its mappings; repository deletion also removes its mappings.

