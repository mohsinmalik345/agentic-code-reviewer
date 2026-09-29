# Roadmap

## GitLab CI/CD

Repository registration, secure managed HTTPS checkout, exact commit capture, durable jobs, and
commit-aware repository-level refreshes are implemented in the MVP. Refreshes reuse unchanged graph,
knowledge, and embedding partitions while publishing one complete portfolio snapshot. The next phase
is push/change analysis:

1. Add GitLab webhook ingestion with constant-time secret verification and delivery-ID idempotency.
2. Persist before/after commit SHAs and map each push to the already registered repository.
3. Implement a GitLab compare provider behind the existing deterministic Git diff abstraction.
4. Queue targeted deployment analysis after a successful graph refresh, using the immutable SHAs.
5. Build a pipeline job that submits merge-request intent, base SHA, and head SHA to GraphQL.
6. Publish the PASS/BLOCK report as a merge-request note and required commit status.
7. Store graph snapshots by commit SHA so analysis never mixes revisions.
8. Extend repository-level reuse to file/function fragments keyed by content SHA-256 and changed dependency cones.
9. Add tenant/repository authorization and per-project Bedrock budgets.

## Jira

Implemented foundation:

- Secure, read-only Jira Cloud REST v3 and Data Center REST v2 clients
- Exact-host allowlisting, verified TLS, bounded responses, and environment-backed credentials
- PostgreSQL/local connection metadata plus repository-to-project mappings
- Frontend and GraphQL management with credential-presence status and connection testing
- Normalized single-issue retrieval in the internal application service

Next phases:

1. Generalize Jira behind a `WorkItemProvider` abstraction.
2. Discover and retrieve only explicitly linked issue keys from Git changes/merge requests.
3. Normalize acceptance criteria into a separately cited intent document.
4. Embed Jira documents with tenant/project ACL metadata.
5. Include acceptance-criteria coverage in the deployment agent schema.
6. Write report links back only after explicit workflow authorization.

## Platform hardening

- Evaluation datasets for extraction recall, evidence rejection, and deployment-gate precision
- Prompt/model version registry and reproducible replay
- OpenTelemetry traces and Bedrock cost budgets
- Dead-letter handling for throttling and object-store failures
- Retrieval-augmented knowledge contexts for repositories beyond one model context
- Additional language extractors while retaining evidence validation
- Alternate vector stores behind the existing port

