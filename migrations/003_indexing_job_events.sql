CREATE TABLE IF NOT EXISTS code_intelligence.indexing_job_events (
    id text PRIMARY KEY,
    sequence bigint GENERATED ALWAYS AS IDENTITY,
    job_id text NOT NULL REFERENCES code_intelligence.indexing_jobs(id) ON DELETE CASCADE,
    level text NOT NULL CHECK (level IN ('info', 'warning', 'error')),
    stage text NOT NULL CHECK (
        stage IN ('queued', 'materializing', 'scanning', 'publishing', 'complete')
    ),
    code text NOT NULL CHECK (code ~ '^[A-Z][A-Z0-9_]{0,127}$'),
    message text NOT NULL CHECK (length(message) BETWEEN 1 AND 2000),
    error_type text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS indexing_job_events_job_idx
    ON code_intelligence.indexing_job_events(job_id, sequence);

INSERT INTO code_intelligence.indexing_job_events
    (id, job_id, level, stage, code, message, error_type, created_at)
SELECT
    'event:legacy:' || md5(id || ':' || COALESCE(error_code, 'FAILED')),
    id,
    'error',
    stage,
    COALESCE(error_code, 'INDEXING_FAILED'),
    'This job failed before durable event logging was enabled; '
        || 'consult its error code and archived worker logs.',
    'LegacyFailure',
    COALESCE(completed_at, updated_at)
FROM code_intelligence.indexing_jobs
WHERE status = 'failed'
ON CONFLICT (id) DO NOTHING;
