CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS code_intelligence;

CREATE TABLE IF NOT EXISTS code_intelligence.scan_runs (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    indexing_mode text NOT NULL DEFAULT 'bedrock-agents',
    repository_ids jsonb NOT NULL,
    graph_path text NOT NULL,
    graph_statistics jsonb NOT NULL,
    evidence_warnings jsonb NOT NULL,
    neo4j_snapshot_id text
);

CREATE TABLE IF NOT EXISTS code_intelligence.agent_invocations (
    invocation_id text PRIMARY KEY,
    scan_run_id text REFERENCES code_intelligence.scan_runs(id) ON DELETE CASCADE,
    role text NOT NULL,
    model_id text NOT NULL,
    repository_id text,
    batch_id text,
    input_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    total_tokens integer NOT NULL,
    latency_ms double precision NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_intelligence.knowledge_documents (
    document_id text PRIMARY KEY,
    scan_run_id text REFERENCES code_intelligence.scan_runs(id) ON DELETE CASCADE,
    repository_id text,
    kind text NOT NULL,
    uri text NOT NULL,
    content_sha256 text NOT NULL,
    size_bytes bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS code_intelligence.embeddings (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL,
    repository_id text,
    kind text NOT NULL,
    source_uri text NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding_model_id text NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS embeddings_repository_idx
    ON code_intelligence.embeddings(repository_id);
CREATE INDEX IF NOT EXISTS embeddings_cosine_idx
    ON code_intelligence.embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS code_intelligence.deployment_reports (
    report_id text PRIMARY KEY,
    repository_id text NOT NULL,
    recommendation text NOT NULL CHECK (recommendation IN ('PASS', 'BLOCK')),
    risk_score integer NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    markdown_path text NOT NULL,
    json_path text NOT NULL,
    report jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

