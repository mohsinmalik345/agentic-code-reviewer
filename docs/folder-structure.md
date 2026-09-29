# Folder structure

```text
ai-code-intelligence-python/
├── config/
│   ├── default.yaml
│   └── container.yaml
├── docs/
│   ├── architecture.md
│   ├── aws-setup.md
│   ├── database-schema.md
│   ├── example-output.md
│   ├── folder-structure.md
│   ├── graphql-api.md
│   ├── neo4j-model.md
│   ├── roadmap.md
│   └── setup.md
├── migrations/
│   ├── 001_initial.sql
│   ├── 002_repository_registry.sql
│   └── 003_indexing_job_events.sql
├── src/ai_code_intelligence/
│   ├── agents/          # Bedrock client, specialist team, supervisor, validators
│   ├── analysis/        # Deterministic impact and deployment reasoning
│   ├── api/             # GraphQL, signed browser sessions, private knowledge endpoints
│   │   └── static/      # Dependency-free operations console and safe Markdown viewer
│   ├── application/     # Indexing, onboarding, no-Bedrock snapshot adoption, composition
│   ├── domain/          # Immutable graph, repository, and job models
│   ├── embeddings/      # Titan, chunking, source selection, vector-store port
│   ├── git/             # Local diff plus safe managed GitLab checkout
│   ├── graph/           # Graph builder, JSON snapshot, Neo4j adapter
│   ├── knowledge/       # Knowledge agents, reader editions, grounded chat, S3/local stores
│   ├── persistence/     # PostgreSQL/file catalog, durable jobs, audit metadata
│   ├── reports/         # Deployment gate and Markdown/JSON reports
│   ├── scanner/         # Read-only repository inventory
│   ├── utils/           # Stable IDs and secret redaction
│   ├── cli.py           # Typer entry point (`code-intel`)
│   ├── config.py        # Strict YAML configuration models
│   ├── logging.py       # JSON-per-line structured logging
│   └── __main__.py      # `python -m ai_code_intelligence`
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

Controllers/resolvers do not contain business logic. The `application` package coordinates ports; infrastructure adapters stay in their owning packages and are selected only by the composition root.

