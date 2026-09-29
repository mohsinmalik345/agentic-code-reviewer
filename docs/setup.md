# Setup and local execution

## Requirements

- Python 3.12
- `uv` (recommended) or pip
- AWS credentials allowed to invoke GLM-5, GPT-OSS 120B, and Titan Embeddings in the configured Region
- Optional: Docker, Neo4j, PostgreSQL with pgvector, and S3

## Install

```powershell
cd ai-code-intelligence-python
uv python install 3.12
uv sync --extra dev
Copy-Item .env.example .env
```

Edit `.env` after copying it. The application loads this project-local file without overriding
values already present in the process environment. Keep long-lived AWS access keys out of it;
prefer `AWS_PROFILE`, IAM Identity Center, or a workload role.

Minimum application values:

| Variable | Value |
|---|---|
| `AWS_REGION` | `us-west-2` (the verified Region of `beelinks-code-knowledgebase`) |
| `AWS_PROFILE` | Your configured AWS CLI/SSO profile for local execution |
| `S3_BUCKET` | `beelinks-code-knowledgebase` |
| `NEO4J_PASSWORD` | The password configured for Neo4j |
| `POSTGRES_URL` | PostgreSQL connection string used by migrations/metadata |
| `CODE_INTEL_API_KEY` | A long random API key for GraphQL |
| `GITLAB_ALLOWED_HOSTS` | Comma-separated exact hostnames, for example `gitlab.company.com` |
| `GITLAB_TOKEN` | Optional read-only token for private repositories; use `read_repository` scope |
| `JIRA_ALLOWED_HOSTS` | Comma-separated exact Jira hostnames, for example `company.atlassian.net,jira.company.com` |
| `JIRA_CLOUD_API_TOKEN` | Jira Cloud API token; leave empty when Cloud is not used |
| `JIRA_DATA_CENTER_PAT` | Jira Data Center personal access token; leave empty when Data Center is not used |
| `BEDROCK_CHAT_MODEL_ID` | `openai.gpt-oss-120b-1:0` (optional override) |
| `BEDROCK_CHAT_MAX_OUTPUT_TOKENS` | `16000` (optional chat response/reasoning budget) |

Do not put a token in a GitLab URL. Public repositories can leave `GITLAB_TOKEN` empty.

## Validate repository configuration

```powershell
uv run code-intel validate-config --config config/default.yaml
```

Configured paths resolve relative to the project root. Scans exclude `node_modules`, `dist`, `build`, `coverage`, and `.git`. Symlinks are not followed.

## Scan and build local graph/knowledge

```powershell
uv run code-intel scan
```

Optional persistence:

```powershell
$env:NEO4J_PASSWORD = "a-strong-password"
$env:POSTGRES_URL = "postgresql://codeintel:password@localhost:5432/codeintel"
uv run code-intel migrate
uv run code-intel scan --persist-neo4j --persist-metadata --index-embeddings
```

The default configuration now uses S3 and the supplied `S3_BUCKET`. Use a separate development
configuration with `knowledge.driver: local` only when S3 publication is intentionally disabled.

## Analyze a deployment

Working-tree diff:

```powershell
uv run code-intel analyze email-service --intent "Add delivery retry handling"
```

Two revisions:

```powershell
uv run code-intel analyze email-service --base origin/main --head HEAD --intent "Add delivery retry handling"
```

Staged changes:

```powershell
uv run code-intel analyze socket-service --staged --intent "Change visitor presence event"
```

Use `--use-neo4j-impact` to persist the fresh snapshot and load the bounded neighborhood from Neo4j before applying domain traversal rules.

## Web console and GraphQL

```powershell
$env:CODE_INTEL_API_KEY = "replace-with-a-long-random-value"
uv run code-intel serve
```

Open `http://127.0.0.1:4000` and sign in with `CODE_INTEL_API_KEY`. The web console supports
repository onboarding, full or repository-attributed reindex jobs, live worker status, graph
statistics, durable job-event and failure logs, reader/evidence document viewing, and grounded
architecture chat. Its **Jira integration** section manages credential-free connection metadata and
repository mappings. The API key is exchanged for a signed browser session and is not persisted in
browser storage. Chat history is also memory-only and is cleared on sign-out or page reload.

POST automation requests to `http://127.0.0.1:4000/graphql` with header `x-api-key`. The CLI-style
API-key flow remains available independently of browser sessions.

Registering a repository only queues work, so the HTTP request does not wait for Bedrock:

```graphql
mutation AddRepository {
  onboardRepository(input: {
    gitlabUrl: "https://gitlab.company.com/beelinks/ticket-service.git"
    ref: "main"
  }) {
    repository { id status }
    job { id status stage }
  }
}
```

With `config/default.yaml`, `serve` runs one embedded local worker. For an explicit worker process,
or whenever `repository_registry.driver` is `postgres`, run:

```powershell
uv run code-intel worker
```

You can also register through the CLI:

```powershell
uv run code-intel onboard "https://gitlab.company.com/beelinks/ticket-service.git" --ref main
```

## Jira configuration

Jira support is read-only. It currently provides secure connection configuration, connection tests,
and repository-to-project mappings; automatic Git-change verification against Jira acceptance
criteria remains a roadmap item.

1. Add each exact Jira hostname and the required secret to `.env`:

   ```dotenv
   JIRA_ALLOWED_HOSTS=company.atlassian.net,jira.company.com
   JIRA_CLOUD_API_TOKEN=replace-with-a-cloud-api-token
   JIRA_DATA_CENTER_PAT=replace-with-a-data-center-personal-access-token
   ```

2. Recreate the application containers so Docker receives the changed environment:

   ```powershell
   docker compose up -d --build app worker
   ```

3. Sign in at `http://127.0.0.1:4000`, open **Jira integration**, and save a connection:

   | Jira edition | Base URL | Authentication | Username | Credential environment variable |
   |---|---|---|---|---|
   | Cloud | `https://company.atlassian.net` | API token | Atlassian account email | `JIRA_CLOUD_API_TOKEN` |
   | Data Center | `https://jira.company.com` | Personal access token | Leave empty | `JIRA_DATA_CENTER_PAT` |

   Enter only the environment-variable **name** in the form, never the token value. The base URL
   must use HTTPS, contain no credentials/query/fragment, and match `JIRA_ALLOWED_HOSTS` exactly.

4. Select **Test** on the saved connection. A successful result shows bounded server and authenticated-user
   metadata without exposing the credential.

5. Map an indexed repository to an uppercase Jira project key. Optional acceptance-criteria field
   IDs (for example `description,customfield_10042`) and the issue-key regular expression are stored
   with the mapping for the future change-verification workflow.

With the PostgreSQL repository registry used by Docker, migration
`005_jira_configuration.sql` persists connections and mappings. Local mode uses
`artifacts/jira-configuration.json`. Both stores contain the credential environment-variable name,
not its value. Deleting a connection also deletes its mappings.

### Jira troubleshooting

| Symptom | Check |
|---|---|
| `Jira host is not allowlisted` | Add only the URL hostname (no scheme/path) to `JIRA_ALLOWED_HOSTS`, then restart/recreate the app. |
| `Jira credential is not configured` | Set the exact variable named by the connection and restart/recreate the app. A blank value is treated as missing. |
| Cloud connection validation fails | Use edition `cloud`, auth type `api_token`, and the Atlassian account email associated with the API token. |
| Data Center connection validation fails | Use edition `data_center`, auth type `personal_access_token`, and leave username empty. |
| Connection test returns unauthorized/forbidden | Verify the token is active and has read access to the intended Jira projects. |
| Network, timeout, or TLS failure | Confirm the Jira host is reachable from the `app` container and presents a certificate trusted by the container. Redirects and plain HTTP are rejected. |
| Jira tables are missing | Run `docker compose run --rm migrate` (or `uv run code-intel migrate`) and restart the app. |

## Adopt an existing graph and knowledge base (no Bedrock)

Use `adopt-existing` when a validated local graph and its generated Markdown already exist. This is
not a reindex: it does not scan source, invoke GLM-5, invoke Titan, or change either Beelinks
repository. The source directory must contain exactly:

```text
legacy/
├── graph/latest.json
└── knowledge/
    ├── central.md
    └── repositories/
        ├── email-service.md
        └── socket-service.md
```

Stop the portfolio worker so it cannot claim or publish another snapshot concurrently, then mount
the host artifacts read-only. Run the read-only preflight first:

```powershell
docker compose stop worker

docker compose run --rm `
  -v "${PWD}\artifacts:/legacy:ro" `
  app code-intel adopt-existing `
  --config /app/config/container.yaml `
  --graph /legacy/graph/latest.json `
  --knowledge-directory /legacy/knowledge `
  --validate-only
```

For the existing Beelinks snapshot the expected preflight result is `graph_action: "preserve"`.
That means the supplied 2,418-node/7,304-relationship graph matches Neo4j and its existing snapshot
ID will be retained. If preflight reports a mismatch it fails closed; do not add
`--replace-neo4j` until the difference has been independently reviewed.

Publish the validated snapshot by repeating the command without `--validate-only`:

```powershell
docker compose run --rm `
  -v "${PWD}\artifacts:/legacy:ro" `
  app code-intel adopt-existing `
  --config /app/config/container.yaml `
  --graph /legacy/graph/latest.json `
  --knowledge-directory /legacy/knowledge

docker compose start worker
```

The workflow validates exact enabled-repository coverage, graph statistics, one Repository node per
repository, and every `[node:<id>]` / `[relationship:<id>]` Markdown citation. It copies the graph
to the application artifact volume, writes deterministic versioned S3 documents, records an
`adopted-existing` scan plus six knowledge-document rows in PostgreSQL (three evidence documents and
three deterministic reader editions), atomically updates the
repository catalogue, and writes `knowledge/latest.json` last. Local repositories intentionally
retain a null commit SHA because the legacy Git commit cannot be established independently.

Retries are idempotent: the scan and document IDs derive from graph, Markdown, repository, and
commit content. A failure before the final manifest leaves only unreachable versioned objects or an
inert scan audit row. Catalog, local graph, Neo4j, and the previous manifest are restored when a
later publication step fails.

If an evidence manifest was published before reader editions were introduced, stop the worker and
upgrade that exact manifest without rescanning or changing the registered portfolio:

```powershell
docker compose stop worker
docker compose run --rm app code-intel backfill-readers --config /app/config/container.yaml
docker compose start worker
```

The command rejects an active queue, verifies the latest PostgreSQL scan matches the S3 manifest,
derives and catalogues reader documents, and publishes schema-version 2 `latest.json` last. It is
idempotent and does not invoke GLM-5, GPT-OSS, Titan, Git, or Neo4j.

## Docker Compose

```powershell
docker compose up --build
```

Both existing Beelinks repositories remain mounted read-only. Compose starts PostgreSQL/pgvector,
applies migrations, starts Neo4j, the GraphQL API, and a durable PostgreSQL-backed worker. Dynamic
GitLab clones and graph artifacts use named volumes. Supply real secrets through environment
variables or a Compose secret mechanism before non-local use.

Use HTTPS at the reverse proxy for non-local deployments. The application marks browser cookies
`Secure` when the trusted request scheme is HTTPS and sends a restrictive content-security policy.

The private platform-source index is opt-in and does not start with the public portfolio. See
[Private Codex index](internal-index.md) for its one-time no-model separation and normal startup
commands.

On every successful onboarding job the worker:

1. Refreshes all registered GitLab checkouts and resolves exact commit SHAs.
2. Scans the complete enabled Beelinks repository portfolio without running repository code.
3. Rebuilds cross-service graph relationships.
4. Writes evidence and human-readable reader editions for each repository and the central knowledge
   base to `s3://beelinks-code-knowledgebase/knowledge/`.
5. Atomically replaces the Neo4j graph snapshot and records scan/job metadata.
6. Publishes `s3://beelinks-code-knowledgebase/knowledge/latest.json` last.

## Verification

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```

