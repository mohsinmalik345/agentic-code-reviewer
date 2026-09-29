CREATE TABLE IF NOT EXISTS code_intelligence.jira_connections (
    id text PRIMARY KEY
        CHECK (id ~ '^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$'),
    name text NOT NULL CHECK (length(btrim(name)) BETWEEN 1 AND 200),
    edition text NOT NULL CHECK (edition IN ('cloud', 'data_center')),
    base_url text NOT NULL UNIQUE CHECK (length(base_url) BETWEEN 1 AND 2048),
    auth_type text NOT NULL CHECK (auth_type IN ('api_token', 'personal_access_token')),
    username text CHECK (
        username IS NULL OR length(btrim(username)) BETWEEN 1 AND 320
    ),
    credential_env text NOT NULL
        CHECK (credential_env ~ '^[A-Za-z_][A-Za-z0-9_]{0,127}$'),
    enabled boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (updated_at >= created_at),
    CHECK (
        (edition = 'cloud' AND auth_type = 'api_token' AND username IS NOT NULL)
        OR
        (edition = 'data_center' AND auth_type = 'personal_access_token'
            AND username IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS jira_connections_enabled_idx
    ON code_intelligence.jira_connections(enabled, id);

CREATE TABLE IF NOT EXISTS code_intelligence.jira_project_mappings (
    id text PRIMARY KEY
        CHECK (id ~ '^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$'),
    connection_id text NOT NULL
        REFERENCES code_intelligence.jira_connections(id) ON DELETE CASCADE,
    repository_id text NOT NULL
        REFERENCES code_intelligence.repositories(id) ON DELETE CASCADE,
    jira_project_key text NOT NULL
        CHECK (jira_project_key ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    acceptance_criteria_fields text[] NOT NULL DEFAULT '{}'::text[]
        CHECK (cardinality(acceptance_criteria_fields) <= 32),
    issue_key_pattern text NOT NULL CHECK (
        length(issue_key_pattern) BETWEEN 1 AND 1000
    ),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (connection_id, repository_id),
    CHECK (updated_at >= created_at)
);

CREATE INDEX IF NOT EXISTS jira_project_mappings_repository_idx
    ON code_intelligence.jira_project_mappings(repository_id);

CREATE INDEX IF NOT EXISTS jira_project_mappings_project_idx
    ON code_intelligence.jira_project_mappings(connection_id, jira_project_key);
