from __future__ import annotations

import json
from typing import Any

from psycopg import sql
from psycopg_pool import ConnectionPool

from ai_code_intelligence.application.models import ScanResult
from ai_code_intelligence.domain.analysis import DeploymentReport
from ai_code_intelligence.domain.knowledge import (
    KnowledgeDocument,
    KnowledgeDocumentRecord,
    StoredDocument,
)
from ai_code_intelligence.domain.models import AgentInvocation, AgentRole, GraphStatistics
from ai_code_intelligence.utils.redaction import redact_json_strings, redact_sensitive_text


class PostgresMetadataStore:
    """Persists scan audit trails, token usage, knowledge objects, and deployment gates."""

    def __init__(self, connection_string: str, schema_name: str) -> None:
        self._schema = schema_name
        self._pool = ConnectionPool(conninfo=connection_string, min_size=1, max_size=5, open=True)

    def close(self) -> None:
        self._pool.close()

    def initialize(self) -> None:
        statements: list[str | sql.Composed] = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.scan_runs (
                  id text PRIMARY KEY,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  indexing_mode text NOT NULL,
                  repository_ids jsonb NOT NULL,
                  graph_path text NOT NULL,
                  graph_statistics jsonb NOT NULL,
                  evidence_warnings jsonb NOT NULL,
                  neo4j_snapshot_id text
                )
                """
            ).format(sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.agent_invocations (
                  invocation_id text PRIMARY KEY,
                  scan_run_id text REFERENCES {}.scan_runs(id) ON DELETE CASCADE,
                  role text NOT NULL,
                  model_id text NOT NULL,
                  repository_id text,
                  batch_id text,
                  input_tokens integer NOT NULL,
                  output_tokens integer NOT NULL,
                  total_tokens integer NOT NULL,
                  latency_ms double precision NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.knowledge_documents (
                  document_id text PRIMARY KEY,
                  scan_run_id text REFERENCES {}.scan_runs(id) ON DELETE CASCADE,
                  repository_id text,
                  kind text NOT NULL,
                  uri text NOT NULL,
                  content_sha256 text NOT NULL,
                  size_bytes bigint NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.embeddings (
                  chunk_id text PRIMARY KEY,
                  document_id text NOT NULL,
                  repository_id text,
                  kind text NOT NULL,
                  source_uri text NOT NULL,
                  chunk_index integer NOT NULL,
                  content text NOT NULL,
                  metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                  embedding_model_id text NOT NULL,
                  embedding vector(1024) NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.deployment_reports (
                  report_id text PRIMARY KEY,
                  repository_id text NOT NULL,
                  recommendation text NOT NULL CHECK (recommendation IN ('PASS', 'BLOCK')),
                  risk_score integer NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
                  markdown_path text NOT NULL,
                  json_path text NOT NULL,
                  report jsonb NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.repositories (
                  id text PRIMARY KEY CHECK (id ~ '^[a-z0-9][a-z0-9-]*$'),
                  provider text NOT NULL CHECK (provider IN ('local', 'gitlab')),
                  name text NOT NULL,
                  description text,
                  local_path text NOT NULL,
                  canonical_url text UNIQUE,
                  canonical_ref text,
                  exact_commit text CHECK (
                    exact_commit IS NULL OR exact_commit ~ '^([0-9a-f]{{40}}|[0-9a-f]{{64}})$'
                  ),
                  status text NOT NULL CHECK (
                    status IN ('registered', 'ready', 'indexing', 'indexed', 'failed', 'disabled')
                  ),
                  knowledge_base_uri text,
                  graph_node_count integer NOT NULL DEFAULT 0 CHECK (graph_node_count >= 0),
                  graph_relationship_count integer NOT NULL DEFAULT 0
                    CHECK (graph_relationship_count >= 0),
                  last_scan_run_id text,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  updated_at timestamptz NOT NULL DEFAULT now(),
                  indexed_at timestamptz,
                  CHECK ((provider = 'gitlab' AND canonical_url IS NOT NULL) OR
                         (provider = 'local' AND canonical_url IS NULL))
                )
                """
            ).format(sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.indexing_jobs (
                  id text PRIMARY KEY,
                  requested_repository_id text REFERENCES {}.repositories(id),
                  status text NOT NULL CHECK (
                    status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
                  ),
                  stage text NOT NULL CHECK (
                    stage IN ('queued', 'materializing', 'scanning', 'publishing', 'complete')
                  ),
                  scan_run_id text,
                  error_code text,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  updated_at timestamptz NOT NULL DEFAULT now(),
                  started_at timestamptz,
                  completed_at timestamptz,
                  cancellation_requested_at timestamptz,
                  lease_expires_at timestamptz
                )
                """
            ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
            sql.SQL("ALTER TABLE {}.indexing_jobs ADD COLUMN IF NOT EXISTS cancellation_requested_at timestamptz").format(
                sql.Identifier(self._schema)
            ),
            sql.SQL("ALTER TABLE {}.indexing_jobs DROP CONSTRAINT IF EXISTS indexing_jobs_status_check").format(
                sql.Identifier(self._schema)
            ),
            sql.SQL(
                "ALTER TABLE {}.indexing_jobs ADD CONSTRAINT indexing_jobs_status_check "
                "CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled'))"
            ).format(sql.Identifier(self._schema)),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.indexing_job_events (
                  id text PRIMARY KEY,
                  sequence bigint GENERATED ALWAYS AS IDENTITY,
                  job_id text NOT NULL REFERENCES {}.indexing_jobs(id) ON DELETE CASCADE,
                  level text NOT NULL CHECK (level IN ('info', 'warning', 'error')),
                  stage text NOT NULL CHECK (
                    stage IN ('queued', 'materializing', 'scanning', 'publishing', 'complete')
                  ),
                  code text NOT NULL CHECK (code ~ '^[A-Z][A-Z0-9_]{{0,127}}$'),
                  message text NOT NULL CHECK (length(message) BETWEEN 1 AND 2000),
                  error_type text,
                  created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.embeddings(repository_id)").format(
                sql.Identifier("embeddings_repository_idx"),
                sql.Identifier(self._schema),
            ),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {}.embeddings USING hnsw (embedding vector_cosine_ops)"
            ).format(
                sql.Identifier("embeddings_cosine_idx"),
                sql.Identifier(self._schema),
            ),
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.indexing_jobs(status, created_at)").format(
                sql.Identifier("indexing_jobs_queue_idx"),
                sql.Identifier(self._schema),
            ),
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.indexing_job_events(job_id, sequence)").format(
                sql.Identifier("indexing_job_events_job_idx"),
                sql.Identifier(self._schema),
            ),
            sql.SQL(
                """
                INSERT INTO {}.indexing_job_events
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
                FROM {}.indexing_jobs
                WHERE status = 'failed'
                ON CONFLICT (id) DO NOTHING
                """
            ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
        ]
        with self._pool.connection() as connection, connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
            connection.commit()

    def record_scan(
        self,
        result: ScanResult,
        invocations: tuple[AgentInvocation, ...],
        documents: tuple[KnowledgeDocument, ...],
        stored: tuple[StoredDocument, ...],
    ) -> None:
        document_by_id = {document.id: document for document in documents}
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.scan_runs
                      (id, indexing_mode, repository_ids, graph_path, graph_statistics,
                       evidence_warnings, neo4j_snapshot_id)
                    VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE SET
                      graph_path = EXCLUDED.graph_path,
                      graph_statistics = EXCLUDED.graph_statistics,
                      evidence_warnings = EXCLUDED.evidence_warnings,
                      neo4j_snapshot_id = EXCLUDED.neo4j_snapshot_id
                    """
                ).format(sql.Identifier(self._schema)),
                (
                    result.scan_run_id,
                    result.indexing_mode,
                    json.dumps(result.repository_ids),
                    result.graph_path,
                    result.statistics.model_dump_json(),
                    json.dumps([redact_sensitive_text(value) for value in result.evidence_warnings]),
                    result.neo4j_snapshot_id,
                ),
            )
            invocation_statement = sql.SQL(
                """
                INSERT INTO {}.agent_invocations
                  (invocation_id, scan_run_id, role, model_id, repository_id, batch_id,
                   input_tokens, output_tokens, total_tokens, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (invocation_id) DO NOTHING
                """
            ).format(sql.Identifier(self._schema))
            cursor.executemany(
                invocation_statement,
                [
                    (
                        item.invocation_id,
                        result.scan_run_id,
                        item.role.value,
                        item.model_id,
                        item.repository_id,
                        item.batch_id,
                        item.input_tokens,
                        item.output_tokens,
                        item.total_tokens,
                        item.latency_ms,
                    )
                    for item in invocations
                ],
            )
            document_statement = sql.SQL(
                """
                INSERT INTO {}.knowledge_documents
                  (document_id, scan_run_id, repository_id, kind, uri, content_sha256, size_bytes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                  uri = EXCLUDED.uri,
                  content_sha256 = EXCLUDED.content_sha256,
                  size_bytes = EXCLUDED.size_bytes
                """
            ).format(sql.Identifier(self._schema))
            cursor.executemany(
                document_statement,
                [
                    (
                        item.document_id,
                        result.scan_run_id,
                        document_by_id[item.document_id].repository_id,
                        document_by_id[item.document_id].kind,
                        item.uri,
                        item.content_sha256,
                        item.size_bytes,
                    )
                    for item in stored
                ],
            )
            connection.commit()

    def latest_scan(self) -> ScanResult | None:
        """Load the newest durable scan audit without invoking an analysis model."""

        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT id, indexing_mode, repository_ids, graph_path, graph_statistics,
                           evidence_warnings, neo4j_snapshot_id
                    FROM {}.scan_runs
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                ).format(sql.Identifier(self._schema))
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._scan_result(cursor, row)

    def get_scan(self, scan_run_id: str) -> ScanResult | None:
        """Load one exact historical scan for deterministic publication recovery."""

        if not scan_run_id.strip():
            raise ValueError("scan_run_id cannot be empty")
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT id, indexing_mode, repository_ids, graph_path, graph_statistics,
                           evidence_warnings, neo4j_snapshot_id
                    FROM {}.scan_runs
                    WHERE id = %s
                    """
                ).format(sql.Identifier(self._schema)),
                (scan_run_id,),
            )
            row = cursor.fetchone()
            return self._scan_result(cursor, row) if row is not None else None

    def knowledge_documents_for_scan(
        self,
        scan_run_id: str,
    ) -> tuple[KnowledgeDocumentRecord, ...]:
        """Return the complete, ordered knowledge-object audit for one scan."""

        if not scan_run_id.strip():
            raise ValueError("scan_run_id cannot be empty")
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT document_id, scan_run_id, repository_id, kind, uri,
                           content_sha256, size_bytes
                    FROM {}.knowledge_documents
                    WHERE scan_run_id = %s
                    ORDER BY kind, repository_id NULLS LAST, document_id
                    """
                ).format(sql.Identifier(self._schema)),
                (scan_run_id,),
            )
            return tuple(
                KnowledgeDocumentRecord(
                    document_id=row[0],
                    scan_run_id=row[1],
                    repository_id=row[2],
                    kind=row[3],
                    uri=row[4],
                    content_sha256=row[5],
                    size_bytes=row[6],
                )
                for row in cursor.fetchall()
            )

    def delete_scan(self, scan_run_id: str) -> bool:
        """Delete one caller-owned recovery audit when publication rollback succeeds."""

        if not scan_run_id.strip():
            raise ValueError("scan_run_id cannot be empty")
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {}.scan_runs WHERE id = %s").format(
                    sql.Identifier(self._schema)
                ),
                (scan_run_id,),
            )
            deleted = cursor.rowcount > 0
            connection.commit()
            return deleted

    def _scan_result(self, cursor: Any, row: Any) -> ScanResult:
        """Hydrate a scan result while reusing the caller's consistent DB snapshot."""

        (
            scan_run_id,
            indexing_mode,
            repository_ids,
            graph_path,
            graph_statistics,
            evidence_warnings,
            neo4j_snapshot_id,
        ) = row
        cursor.execute(
            sql.SQL(
                """
                SELECT uri
                FROM {}.knowledge_documents
                WHERE scan_run_id = %s
                  AND kind IN ('repository', 'central')
                ORDER BY CASE WHEN kind = 'repository' THEN 0 ELSE 1 END,
                         repository_id NULLS LAST, document_id
                """
            ).format(sql.Identifier(self._schema)),
            (scan_run_id,),
        )
        knowledge_uris = tuple(str(item[0]) for item in cursor.fetchall())

        cursor.execute(
            sql.SQL(
                """
                SELECT role, invocation_id, model_id, repository_id, batch_id,
                       input_tokens, output_tokens, total_tokens, latency_ms
                FROM {}.agent_invocations
                WHERE scan_run_id = %s
                ORDER BY created_at, invocation_id
                """
            ).format(sql.Identifier(self._schema)),
            (scan_run_id,),
        )
        invocations = tuple(
                AgentInvocation(
                    role=AgentRole(item[0]),
                    invocation_id=item[1],
                    model_id=item[2],
                    repository_id=item[3],
                    batch_id=item[4],
                    input_tokens=item[5],
                    output_tokens=item[6],
                    total_tokens=item[7],
                    latency_ms=item[8],
                )
            for item in cursor.fetchall()
        )

        cursor.execute(
            sql.SQL(
                """
                SELECT count(*)
                FROM {}.embeddings
                WHERE document_id IN (
                  SELECT document_id
                  FROM {}.knowledge_documents
                  WHERE scan_run_id = %s
                    AND kind IN ('repository', 'central')
                )
                """
            ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
            (scan_run_id,),
        )
        embedded_row = cursor.fetchone()
        embedded_chunk_count = int(embedded_row[0]) if embedded_row is not None else 0

        return ScanResult(
            scan_run_id=scan_run_id,
            indexing_mode=indexing_mode,
            repository_ids=tuple(repository_ids),
            graph_path=graph_path,
            statistics=GraphStatistics.model_validate(graph_statistics),
            knowledge_uris=knowledge_uris,
            embedded_chunk_count=embedded_chunk_count,
            agent_invocations=invocations,
            evidence_warnings=tuple(evidence_warnings),
            neo4j_snapshot_id=neo4j_snapshot_id,
        )

    def record_report(self, report: DeploymentReport, markdown_path: str, json_path: str) -> None:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.deployment_reports
                      (report_id, repository_id, recommendation, risk_score,
                       markdown_path, json_path, report)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (report_id) DO UPDATE SET
                      recommendation = EXCLUDED.recommendation,
                      risk_score = EXCLUDED.risk_score,
                      markdown_path = EXCLUDED.markdown_path,
                      json_path = EXCLUDED.json_path,
                      report = EXCLUDED.report
                    """
                ).format(sql.Identifier(self._schema)),
                (
                    report.id,
                    report.change.repository_id,
                    report.recommendation,
                    report.risk_score,
                    markdown_path,
                    json_path,
                    json.dumps(redact_json_strings(report.model_dump(mode="json"))),
                ),
            )
            connection.commit()
