ALTER TABLE code_intelligence.indexing_jobs
    ADD COLUMN IF NOT EXISTS cancellation_requested_at timestamptz;

ALTER TABLE code_intelligence.indexing_jobs
    DROP CONSTRAINT IF EXISTS indexing_jobs_status_check;

ALTER TABLE code_intelligence.indexing_jobs
    ADD CONSTRAINT indexing_jobs_status_check
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled'));
