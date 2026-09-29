CREATE TABLE IF NOT EXISTS code_intelligence.repositories (
    id text PRIMARY KEY CHECK (id ~ '^[a-z0-9][a-z0-9-]*$'),
    provider text NOT NULL CHECK (provider IN ('local', 'gitlab')),
    name text NOT NULL,
    description text,
    local_path text NOT NULL,
    canonical_url text UNIQUE,
    canonical_ref text,
    exact_commit text CHECK (exact_commit IS NULL OR exact_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'),
    status text NOT NULL CHECK (status IN ('registered', 'ready', 'indexing', 'indexed', 'failed', 'disabled')),
    knowledge_base_uri text,
    graph_node_count integer NOT NULL DEFAULT 0 CHECK (graph_node_count >= 0),
    graph_relationship_count integer NOT NULL DEFAULT 0 CHECK (graph_relationship_count >= 0),
    last_scan_run_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    indexed_at timestamptz,
    CHECK ((provider = 'gitlab' AND canonical_url IS NOT NULL) OR
           (provider = 'local' AND canonical_url IS NULL))
);

CREATE TABLE IF NOT EXISTS code_intelligence.indexing_jobs (
    id text PRIMARY KEY,
    requested_repository_id text REFERENCES code_intelligence.repositories(id),
    status text NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    stage text NOT NULL CHECK (stage IN ('queued', 'materializing', 'scanning', 'publishing', 'complete')),
    scan_run_id text,
    error_code text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    lease_expires_at timestamptz
);

CREATE INDEX IF NOT EXISTS indexing_jobs_queue_idx
    ON code_intelligence.indexing_jobs(status, created_at);
