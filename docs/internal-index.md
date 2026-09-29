# Private Codex Index

The `ai-code-intelligence` repository is operational tooling, not a Beelinks CRM microservice. Its
graph and knowledge base therefore run in a separate publication domain used for internal code
reference. It must never appear in the Beelinks central knowledge base, public Neo4j graph, or
active S3 manifest.

## Isolation boundary

| Resource | Beelinks portfolio | Private Codex index |
|---|---|---|
| GraphQL/API | `http://127.0.0.1:4000` | `http://127.0.0.1:4100` |
| Neo4j Browser | `http://127.0.0.1:7474` | `http://127.0.0.1:7475` |
| Neo4j Bolt | `localhost:7687` | `localhost:7688` |
| PostgreSQL schema | `code_intelligence` | `code_intelligence_internal` |
| Knowledge | private S3 `knowledge/` | local `internal-artifacts` volume |
| Graph/artifacts | `app-artifacts` volume | `internal-artifacts` volume |
| Repository source | managed Beelinks checkouts | read-only `/app/self-source` mount |

Both internal ports bind to `127.0.0.1`; they are not exposed on the LAN. Set distinct
`NEO4J_INTERNAL_PASSWORD` and `CODE_INTEL_INTERNAL_API_KEY` values in `.env` when desired. If they
are omitted, Docker Compose inherits the main Neo4j password and API key so the internal API never
starts with an empty default credential.

## One-time separation without GLM re-scanning

The separation command reuses the active graph and repository evidence. It first publishes the
private copy, verifies that pruning the self slice exactly matches a chosen historical public scan,
then atomically advances public Neo4j/catalog/manifest state. No GLM-5 or GPT-OSS invocation occurs.
Only the restored central document is re-embedded with Titan after the public manifest commits;
other Beelinks vectors and all source analysis are reused.

Stop the public worker, start the private Neo4j service, build the current image, and run:

```powershell
docker compose stop worker
docker compose --profile codex-internal up -d --build neo4j-internal

docker compose --profile codex-internal run --rm internal-migrate `
  code-intel separate-internal-repository `
  --config /app/config/container.yaml `
  --internal-config /app/config/internal.yaml `
  --repository-id ai-code-intelligence `
  --restore-scan-run-id 2b980868-25e2-458a-b6ed-894d218501ce `
  --staging-directory /app/internal-artifacts/import
```

That restore scan ID is the audited three-repository Beelinks publication immediately before the
self-index was added. The command refuses to proceed if its repository IDs or graph statistics no
longer match the exact pruned graph. The public manifest is written last. If an earlier public
commit step fails, Neo4j, the local graph, repository metadata, document aliases, scan audit, and
prior manifest are restored. The already-created private copy is safe to reuse on retry.

After a successful split, start both normal runtimes:

```powershell
docker compose up -d --build app worker
docker compose --profile codex-internal up -d --build internal-app internal-worker
```

The internal worker is only needed to refresh this private local-source index later. Starting or
using it cannot enqueue work in the Beelinks schema.

## Querying from Codex or automation

Use the internal API key (or the inherited main key) only against port 4100:

```powershell
$headers = @{ "x-api-key" = $env:CODE_INTEL_INTERNAL_API_KEY }
$body = @{
  query = @"
    query InternalIndex {
      graphSnapshot {
        repositoryIds
        statistics { nodeCount relationshipCount }
      }
      graphNodes(repositoryId: "ai-code-intelligence", limit: 100) {
        id type name qualifiedName filePath startLine endLine
      }
    }
"@
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:4100/graphql `
  -Headers $headers `
  -ContentType application/json `
  -Body $body
```

The authenticated knowledge endpoint is:

```text
GET http://127.0.0.1:4100/api/knowledge/human/repositories/ai-code-intelligence
```

For direct graph inspection, open the private Neo4j Browser on port 7475 and run:

```cypher
MATCH (n:CodeIntelligence {repositoryId: 'ai-code-intelligence'})
RETURN n.type AS type, count(*) AS count
ORDER BY count DESC;
```

The public verification query on port 7474 should return zero:

```cypher
MATCH (n:CodeIntelligence {repositoryId: 'ai-code-intelligence'})
RETURN count(n) AS internalNodesInPublicGraph;
```
