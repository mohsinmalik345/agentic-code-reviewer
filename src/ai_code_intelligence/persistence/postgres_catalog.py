from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import uuid4

from psycopg import errors, sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ai_code_intelligence.domain.models import RepositoryDefinition
from ai_code_intelligence.domain.repositories import (
    IndexingJob,
    IndexingJobEvent,
    IndexingJobStatus,
    IndexingStage,
    JobEventLevel,
    RepositoryProvider,
    RepositoryRecord,
    RepositoryStatus,
)
from ai_code_intelligence.persistence.catalog import (
    IndexingJobEventStore,
    IndexingJobStore,
    RepositoryCatalog,
)

_STAGE_ORDER = {
    IndexingStage.QUEUED: 0,
    IndexingStage.MATERIALIZING: 1,
    IndexingStage.SCANNING: 2,
    IndexingStage.PUBLISHING: 3,
    IndexingStage.COMPLETE: 4,
}


class PostgresRepositoryCatalog(RepositoryCatalog, IndexingJobStore, IndexingJobEventStore):
    """PostgreSQL repository registry and single-portfolio durable job queue."""

    def __init__(
        self,
        connection_string: str,
        schema_name: str,
        lease_seconds: int,
    ) -> None:
        self._schema = schema_name
        self._lease_seconds = lease_seconds
        self._pool = ConnectionPool(
            conninfo=connection_string,
            min_size=1,
            max_size=5,
            open=True,
        )

    def close(self) -> None:
        """Close all pooled PostgreSQL connections."""

        self._pool.close()

    def seed_local_repositories(
        self,
        definitions: Iterable[RepositoryDefinition],
    ) -> tuple[RepositoryRecord, ...]:
        """Idempotently register YAML repositories as local bootstrap records."""

        records: list[RepositoryRecord] = []
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            for definition in definitions:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.repositories
                          (id, provider, name, description, local_path, status)
                        VALUES (%s, 'local', %s, %s, %s, 'ready')
                        ON CONFLICT (id) DO UPDATE SET
                          name = EXCLUDED.name,
                          description = EXCLUDED.description,
                          local_path = EXCLUDED.local_path,
                          updated_at = CASE
                            WHEN {}.repositories.provider = 'local' THEN now()
                            ELSE {}.repositories.updated_at
                          END
                        WHERE {}.repositories.provider = 'local'
                        RETURNING *
                        """
                    ).format(
                        sql.Identifier(self._schema),
                        sql.Identifier(self._schema),
                        sql.Identifier(self._schema),
                        sql.Identifier(self._schema),
                    ),
                    (definition.id, definition.name, definition.description, definition.path),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(
                        f"configured local repository {definition.id} conflicts with a GitLab registration"
                    )
                records.append(_repository_from_row(row))
            connection.commit()
        return tuple(records)

    def upsert_repository(self, record: RepositoryRecord) -> RepositoryRecord:
        """Insert or update mutable repository state while preserving source identity."""

        return self.upsert_repositories((record,))[0]

    def upsert_repositories(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]:
        """Atomically insert or update a nonempty set of repository registrations."""

        requested = tuple(
            RepositoryRecord.model_validate(record.model_dump(mode="python")) for record in records
        )
        if not requested:
            raise ValueError("repository registration requires at least one record")
        if len({record.id for record in requested}) != len(requested):
            raise ValueError("repository registration requires unique repository IDs")
        canonical_urls = tuple(
            record.canonical_url for record in requested if record.canonical_url is not None
        )
        if len(set(canonical_urls)) != len(canonical_urls):
            raise ValueError("repository registration requires unique GitLab URLs")

        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock("
                    "hashtext('code_intelligence.repository_registration'))"
                )
                cursor.execute(
                    sql.SQL("SELECT * FROM {}.repositories FOR UPDATE").format(
                        sql.Identifier(self._schema)
                    )
                )
                existing_by_id = {
                    existing.id: existing
                    for row in cursor.fetchall()
                    for existing in (_repository_from_row(row),)
                }
                requested_ids = {record.id for record in requested}
                occupied_urls = {
                    existing.canonical_url: existing.id
                    for existing in existing_by_id.values()
                    if existing.id not in requested_ids and existing.canonical_url is not None
                }
                for record in requested:
                    existing = existing_by_id.get(record.id)
                    if existing is not None:
                        if existing.provider is not record.provider:
                            raise ValueError(f"repository {record.id} provider cannot be changed")
                        if existing.canonical_url != record.canonical_url:
                            raise ValueError(
                                f"repository {record.id} canonical_url cannot be changed"
                            )
                    duplicate_id = (
                        occupied_urls.get(record.canonical_url)
                        if record.canonical_url is not None
                        else None
                    )
                    if duplicate_id is not None:
                        raise ValueError(
                            f"GitLab repository URL is already registered as {duplicate_id}"
                        )

                stored: list[RepositoryRecord] = []
                for record in requested:
                    cursor.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.repositories
                              (id, provider, name, description, local_path, canonical_url,
                               canonical_ref, exact_commit, status, knowledge_base_uri,
                               graph_node_count, graph_relationship_count, last_scan_run_id,
                               created_at, updated_at, indexed_at)
                            VALUES
                              (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                              name = EXCLUDED.name,
                              description = EXCLUDED.description,
                              local_path = EXCLUDED.local_path,
                              canonical_ref = EXCLUDED.canonical_ref,
                              exact_commit = EXCLUDED.exact_commit,
                              status = EXCLUDED.status,
                              knowledge_base_uri = EXCLUDED.knowledge_base_uri,
                              graph_node_count = EXCLUDED.graph_node_count,
                              graph_relationship_count = EXCLUDED.graph_relationship_count,
                              last_scan_run_id = EXCLUDED.last_scan_run_id,
                              updated_at = now(),
                              indexed_at = EXCLUDED.indexed_at
                            RETURNING *
                            """
                        ).format(sql.Identifier(self._schema)),
                        (
                            record.id,
                            record.provider.value,
                            record.name,
                            record.description,
                            record.local_path,
                            record.canonical_url,
                            record.canonical_ref,
                            record.exact_commit,
                            record.status.value,
                            record.knowledge_base_uri,
                            record.graph_node_count,
                            record.graph_relationship_count,
                            record.last_scan_run_id,
                            record.created_at,
                            record.updated_at,
                            record.indexed_at,
                        ),
                    )
                    row = cursor.fetchone()
                    assert row is not None
                    stored.append(_repository_from_row(row))
                connection.commit()
                return tuple(stored)
            except errors.UniqueViolation as error:
                connection.rollback()
                raise ValueError("GitLab repository URL is already registered") from error
            except Exception:
                connection.rollback()
                raise

    def register_repositories_and_ensure_queued_job(
        self,
        records: Iterable[RepositoryRecord],
        *,
        requested_repository_id: str | None = None,
    ) -> tuple[tuple[RepositoryRecord, ...], IndexingJob, bool]:
        """Register operator fields and ensure one queued job in one SQL transaction."""

        requested = _validated_repository_batch(records)
        requested_ids = {record.id for record in requested}
        if requested_repository_id is not None and requested_repository_id not in requested_ids:
            raise ValueError("requested repository must belong to the registration batch")
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock("
                    "hashtext('code_intelligence.repository_registration'))"
                )
                stored: list[RepositoryRecord] = []
                for record in requested:
                    cursor.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.repositories
                              (id, provider, name, description, local_path, canonical_url,
                               canonical_ref, exact_commit, status, knowledge_base_uri,
                               graph_node_count, graph_relationship_count, last_scan_run_id,
                               created_at, updated_at, indexed_at)
                            VALUES
                              (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                              name = EXCLUDED.name,
                              description = EXCLUDED.description,
                              canonical_ref = EXCLUDED.canonical_ref,
                              updated_at = now()
                            WHERE repositories.provider = EXCLUDED.provider
                              AND repositories.canonical_url IS NOT DISTINCT FROM EXCLUDED.canonical_url
                            RETURNING *
                            """
                        ).format(sql.Identifier(self._schema)),
                        (
                            record.id,
                            record.provider.value,
                            record.name,
                            record.description,
                            record.local_path,
                            record.canonical_url,
                            record.canonical_ref,
                            record.exact_commit,
                            record.status.value,
                            record.knowledge_base_uri,
                            record.graph_node_count,
                            record.graph_relationship_count,
                            record.last_scan_run_id,
                            record.created_at,
                            record.updated_at,
                            record.indexed_at,
                        ),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise ValueError(
                            f"repository {record.id} source identity cannot be changed"
                        )
                    stored.append(_repository_from_row(row))

                cursor.execute(
                    sql.SQL(
                        """
                        SELECT * FROM {}.indexing_jobs
                        WHERE status = %s
                        ORDER BY created_at, id
                        LIMIT 1
                        FOR UPDATE
                        """
                    ).format(sql.Identifier(self._schema)),
                    (IndexingJobStatus.QUEUED.value,),
                )
                queued = cursor.fetchone()
                job_created = queued is None
                if queued is None:
                    new_job = IndexingJob.create(requested_repository_id)
                    cursor.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.indexing_jobs
                              (id, requested_repository_id, status, stage, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING *
                            """
                        ).format(sql.Identifier(self._schema)),
                        (
                            new_job.id,
                            new_job.requested_repository_id,
                            new_job.status.value,
                            new_job.stage.value,
                            new_job.created_at,
                            new_job.updated_at,
                        ),
                    )
                    queued = cursor.fetchone()
                    assert queued is not None
                connection.commit()
                return tuple(stored), _job_from_row(queued), job_created
            except errors.UniqueViolation as error:
                connection.rollback()
                raise ValueError("GitLab repository URL is already registered") from error
            except Exception:
                connection.rollback()
                raise

    def get_repository(self, repository_id: str) -> RepositoryRecord | None:
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.repositories WHERE id = %s").format(sql.Identifier(self._schema)),
                (repository_id,),
            )
            row = cursor.fetchone()
        return _repository_from_row(row) if row is not None else None

    def list_repositories(self, *, include_disabled: bool = True) -> tuple[RepositoryRecord, ...]:
        where = sql.SQL("") if include_disabled else sql.SQL(" WHERE status <> 'disabled'")
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.repositories").format(sql.Identifier(self._schema))
                + where
                + sql.SQL(" ORDER BY id")
            )
            rows = cursor.fetchall()
        return tuple(_repository_from_row(row) for row in rows)

    def publish_index_metadata(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]:
        """Publish all portfolio index pointers in one PostgreSQL transaction."""

        requested = tuple(records)
        if not requested or len({record.id for record in requested}) != len(requested):
            raise ValueError("index metadata requires nonempty, unique repository records")
        validated = tuple(
            RepositoryRecord.model_validate(record.model_dump(mode="python")) for record in requested
        )
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    sql.SQL("SELECT * FROM {}.repositories WHERE id = ANY(%s) FOR UPDATE").format(
                        sql.Identifier(self._schema)
                    ),
                    ([record.id for record in validated],),
                )
                existing_by_id = {
                    record.id: record for row in cursor.fetchall() for record in (_repository_from_row(row),)
                }
                missing = sorted({record.id for record in validated} - existing_by_id.keys())
                if missing:
                    raise ValueError(f"unknown repository ids: {missing}")

                published: list[RepositoryRecord] = []
                for record in validated:
                    existing = existing_by_id[record.id]
                    if (
                        existing.provider is not record.provider
                        or existing.canonical_url != record.canonical_url
                    ):
                        raise ValueError("repository source identity cannot be changed")
                    if _index_metadata_tuple(existing) == _index_metadata_tuple(record):
                        published.append(existing)
                        continue
                    cursor.execute(
                        sql.SQL(
                            """
                            UPDATE {}.repositories SET
                              status = %s,
                              knowledge_base_uri = %s,
                              graph_node_count = %s,
                              graph_relationship_count = %s,
                              last_scan_run_id = %s,
                              indexed_at = %s,
                              updated_at = now()
                            WHERE id = %s
                            RETURNING *
                            """
                        ).format(sql.Identifier(self._schema)),
                        (
                            record.status.value,
                            record.knowledge_base_uri,
                            record.graph_node_count,
                            record.graph_relationship_count,
                            record.last_scan_run_id,
                            record.indexed_at,
                            record.id,
                        ),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError(f"repository disappeared during publication: {record.id}")
                    published.append(_repository_from_row(row))
                connection.commit()
                return tuple(published)
            except Exception:
                connection.rollback()
                raise

    def migrate_local_sources_to_gitlab(
        self,
        records: Iterable[RepositoryRecord],
    ) -> tuple[RepositoryRecord, ...]:
        """Atomically convert idle local registrations into GitLab registrations."""

        requested = tuple(
            RepositoryRecord.model_validate(record.model_dump(mode="python")) for record in records
        )
        if not requested or len({record.id for record in requested}) != len(requested):
            raise ValueError("source migration requires nonempty, unique repository records")
        if len({record.canonical_url for record in requested}) != len(requested):
            raise ValueError("source migration requires unique GitLab URLs")
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock("
                    "hashtext('code_intelligence.portfolio_index_claim'))"
                )
                cursor.execute(
                    sql.SQL(
                        "SELECT EXISTS(SELECT 1 FROM {}.indexing_jobs "
                        "WHERE status IN ('queued', 'running')) AS active"
                    ).format(sql.Identifier(self._schema))
                )
                active = cursor.fetchone()
                if active is not None and bool(active["active"]):
                    raise ValueError(
                        "repository sources cannot be migrated while an indexing job is active"
                    )
                cursor.execute(
                    sql.SQL("SELECT * FROM {}.repositories FOR UPDATE").format(
                        sql.Identifier(self._schema)
                    )
                )
                existing_by_id = {
                    record.id: record
                    for row in cursor.fetchall()
                    for record in (_repository_from_row(row),)
                }
                migrated_ids = {record.id for record in requested}
                occupied_urls = {
                    record.canonical_url
                    for record in existing_by_id.values()
                    if record.id not in migrated_ids and record.canonical_url is not None
                }
                if occupied_urls.intersection(record.canonical_url for record in requested):
                    raise ValueError("GitLab repository URL is already registered")

                migrated: list[RepositoryRecord] = []
                for record in requested:
                    existing = existing_by_id.get(record.id)
                    if existing is None:
                        raise ValueError(f"unknown repository id: {record.id}")
                    if existing.provider is not RepositoryProvider.LOCAL:
                        raise ValueError(f"repository {record.id} is not a local registration")
                    if record.provider is not RepositoryProvider.GITLAB or record.canonical_url is None:
                        raise ValueError("source migration target must be a GitLab registration")
                    cursor.execute(
                        sql.SQL(
                            """
                            UPDATE {}.repositories SET
                              provider = 'gitlab', local_path = %s, canonical_url = %s,
                              canonical_ref = %s, exact_commit = %s, status = 'ready',
                              updated_at = now()
                            WHERE id = %s AND provider = 'local'
                            RETURNING *
                            """
                        ).format(sql.Identifier(self._schema)),
                        (
                            record.local_path,
                            record.canonical_url,
                            record.canonical_ref,
                            record.exact_commit,
                            record.id,
                        ),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError(f"repository source changed during migration: {record.id}")
                    migrated.append(_repository_from_row(row))
                connection.commit()
                return tuple(migrated)
            except Exception:
                connection.rollback()
                raise

    def enqueue_job(self, job: IndexingJob) -> IndexingJob:
        """Persist one queued full-portfolio indexing request."""

        if job.status is not IndexingJobStatus.QUEUED:
            raise ValueError("only queued jobs can be enqueued")
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.indexing_jobs
                          (id, requested_repository_id, status, stage, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING *
                        """
                    ).format(sql.Identifier(self._schema)),
                    (
                        job.id,
                        job.requested_repository_id,
                        job.status.value,
                        job.stage.value,
                        job.created_at,
                        job.updated_at,
                    ),
                )
                row = cursor.fetchone()
                assert row is not None
                connection.commit()
                return _job_from_row(row)
            except errors.ForeignKeyViolation as error:
                connection.rollback()
                raise ValueError(f"unknown repository id: {job.requested_repository_id}") from error

    def get_job(self, job_id: str) -> IndexingJob | None:
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.indexing_jobs WHERE id = %s").format(sql.Identifier(self._schema)),
                (job_id,),
            )
            row = cursor.fetchone()
        return _job_from_row(row) if row is not None else None

    def list_jobs(self, *, status: IndexingJobStatus | None = None) -> tuple[IndexingJob, ...]:
        where = sql.SQL("") if status is None else sql.SQL(" WHERE status = %s")
        parameters: tuple[str, ...] = () if status is None else (status.value,)
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.indexing_jobs").format(sql.Identifier(self._schema))
                + where
                + sql.SQL(" ORDER BY created_at, id"),
                parameters,
            )
            rows = cursor.fetchall()
        return tuple(_job_from_row(row) for row in rows)

    def claim_next_job(self) -> IndexingJob | None:
        """Lease the oldest job only when no other portfolio refresh is active."""

        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT pg_try_advisory_xact_lock("
                "hashtext('code_intelligence.portfolio_index_claim')) AS acquired"
            )
            lock = cursor.fetchone()
            if lock is None or not bool(lock["acquired"]):
                connection.commit()
                return None
            cursor.execute(
                sql.SQL(
                    """
                    UPDATE {}.indexing_jobs
                    SET status = CASE
                          WHEN cancellation_requested_at IS NULL THEN 'failed'
                          ELSE 'cancelled'
                        END,
                        error_code = CASE
                          WHEN cancellation_requested_at IS NULL THEN 'WORKER_LEASE_EXPIRED'
                          ELSE NULL
                        END,
                        completed_at = now(), updated_at = now(), lease_expires_at = NULL
                    WHERE status = 'running' AND lease_expires_at < now()
                    RETURNING id, stage, status, completed_at
                    """
                ).format(sql.Identifier(self._schema))
            )
            expired_jobs = cursor.fetchall()
            for expired in expired_jobs:
                cancelled = expired["status"] == IndexingJobStatus.CANCELLED.value
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.indexing_job_events
                          (id, job_id, level, stage, code, message, error_type, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """
                    ).format(sql.Identifier(self._schema)),
                    (
                        f"event:{uuid4()}",
                        expired["id"],
                        JobEventLevel.WARNING.value if cancelled else JobEventLevel.ERROR.value,
                        expired["stage"],
                        "JOB_CANCELLED" if cancelled else "WORKER_LEASE_EXPIRED",
                        (
                            "The worker stopped after cancellation was requested."
                            if cancelled
                            else (
                                "The worker stopped renewing this job before it completed. "
                                "Retrying will resume successful Bedrock checkpoints."
                            )
                        ),
                        None if cancelled else "WorkerLeaseExpired",
                        expired["completed_at"],
                    ),
                )
            cursor.execute(
                sql.SQL(
                    """
                    SELECT EXISTS(
                      SELECT 1 FROM {}.indexing_jobs WHERE status = 'running'
                    ) AS active
                    """
                ).format(sql.Identifier(self._schema))
            )
            active = cursor.fetchone()
            if active is not None and bool(active["active"]):
                connection.commit()
                return None
            cursor.execute(
                sql.SQL(
                    """
                    WITH candidate AS (
                      SELECT id FROM {}.indexing_jobs
                      WHERE status = 'queued'
                      ORDER BY created_at, id
                      FOR UPDATE SKIP LOCKED
                      LIMIT 1
                    )
                    UPDATE {}.indexing_jobs AS job
                    SET status = 'running', stage = 'materializing',
                        started_at = now(), updated_at = now(),
                        lease_expires_at = now() + (%s * interval '1 second')
                    FROM candidate
                    WHERE job.id = candidate.id
                    RETURNING job.*
                    """
                ).format(sql.Identifier(self._schema), sql.Identifier(self._schema)),
                (self._lease_seconds,),
            )
            row = cursor.fetchone()
            connection.commit()
        return _job_from_row(row) if row is not None else None

    def request_job_cancellation(self, job_id: str) -> IndexingJob:
        """Atomically cancel queued work or request cancellation of leased work."""

        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.indexing_jobs WHERE id = %s FOR UPDATE").format(
                    sql.Identifier(self._schema)
                ),
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(job_id)
            existing = _job_from_row(row)
            if existing.status is IndexingJobStatus.CANCELLED:
                connection.commit()
                return existing
            if existing.status in {IndexingJobStatus.SUCCEEDED, IndexingJobStatus.FAILED}:
                connection.rollback()
                raise ValueError("completed indexing jobs cannot be cancelled")
            if existing.stage is IndexingStage.PUBLISHING:
                connection.rollback()
                raise ValueError("an indexing job cannot be cancelled after publication starts")
            if existing.cancellation_requested_at is not None:
                connection.commit()
                return existing
            cursor.execute(
                sql.SQL(
                    """
                    UPDATE {}.indexing_jobs
                    SET cancellation_requested_at = now(), updated_at = now(),
                        status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                        completed_at = CASE WHEN status = 'queued' THEN now() ELSE completed_at END,
                        lease_expires_at = CASE
                          WHEN status = 'queued' THEN NULL
                          ELSE lease_expires_at
                        END
                    WHERE id = %s
                    RETURNING *
                    """
                ).format(sql.Identifier(self._schema)),
                (job_id,),
            )
            updated = cursor.fetchone()
            assert updated is not None
            connection.commit()
        return _job_from_row(updated)

    def update_job(
        self,
        job_id: str,
        *,
        status: IndexingJobStatus,
        stage: IndexingStage,
        scan_run_id: str | None = None,
        error_code: str | None = None,
    ) -> IndexingJob:
        """Advance a leased job through the validated monotonic state machine."""

        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.indexing_jobs WHERE id = %s FOR UPDATE").format(
                    sql.Identifier(self._schema)
                ),
                (job_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(job_id)
            existing = _job_from_row(row)
            _validate_transition(existing, status, stage, error_code)
            terminal = status in {
                IndexingJobStatus.SUCCEEDED,
                IndexingJobStatus.FAILED,
                IndexingJobStatus.CANCELLED,
            }
            cursor.execute(
                sql.SQL(
                    """
                    UPDATE {}.indexing_jobs
                    SET status = %s, stage = %s,
                        scan_run_id = COALESCE(%s, scan_run_id),
                        error_code = %s, updated_at = now(),
                        completed_at = CASE WHEN %s THEN now() ELSE NULL END,
                        lease_expires_at = CASE
                          WHEN %s THEN NULL
                          ELSE now() + (%s * interval '1 second')
                        END
                    WHERE id = %s
                    RETURNING *
                    """
                ).format(sql.Identifier(self._schema)),
                (
                    status.value,
                    stage.value,
                    scan_run_id,
                    error_code,
                    terminal,
                    terminal,
                    self._lease_seconds,
                    job_id,
                ),
            )
            updated = cursor.fetchone()
            assert updated is not None
            connection.commit()
        return _job_from_row(updated)

    def append_job_event(self, event: IndexingJobEvent) -> IndexingJobEvent:
        """Append a sanitized event to the durable job timeline."""

        event = IndexingJobEvent.model_validate(event.model_dump(mode="python"))
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.indexing_job_events
                          (id, job_id, level, stage, code, message, error_type, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
                        """
                    ).format(sql.Identifier(self._schema)),
                    (
                        event.id,
                        event.job_id,
                        event.level.value,
                        event.stage.value,
                        event.code,
                        event.message,
                        event.error_type,
                        event.created_at,
                    ),
                )
                row = cursor.fetchone()
                assert row is not None
                connection.commit()
                return _event_from_row(row)
            except errors.ForeignKeyViolation as error:
                connection.rollback()
                raise KeyError(event.job_id) from error

    def list_job_events(
        self,
        job_id: str,
        *,
        limit: int = 200,
    ) -> tuple[IndexingJobEvent, ...]:
        """Return a bounded job timeline in chronological order."""

        if limit < 1 or limit > 1_000:
            raise ValueError("job event limit must be between 1 and 1000")
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT *
                    FROM {}.indexing_job_events
                    WHERE job_id = %s
                    ORDER BY sequence DESC
                    LIMIT %s
                    """
                ).format(sql.Identifier(self._schema)),
                (job_id, limit),
            )
            rows = cursor.fetchall()
        return tuple(_event_from_row(row) for row in reversed(rows))


def _repository_from_row(row: dict[str, object]) -> RepositoryRecord:
    return RepositoryRecord(
        id=str(row["id"]),
        provider=RepositoryProvider(str(row["provider"])),
        name=str(row["name"]),
        description=_optional_string(row.get("description")),
        local_path=str(row["local_path"]),
        canonical_url=_optional_string(row.get("canonical_url")),
        canonical_ref=_optional_string(row.get("canonical_ref")),
        exact_commit=_optional_string(row.get("exact_commit")),
        status=RepositoryStatus(str(row["status"])),
        knowledge_base_uri=_optional_string(row.get("knowledge_base_uri")),
        graph_node_count=int(str(row.get("graph_node_count", 0))),
        graph_relationship_count=int(str(row.get("graph_relationship_count", 0))),
        last_scan_run_id=_optional_string(row.get("last_scan_run_id")),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        indexed_at=_optional_datetime(row.get("indexed_at")),
    )


def _validated_repository_batch(
    records: Iterable[RepositoryRecord],
) -> tuple[RepositoryRecord, ...]:
    requested = tuple(
        RepositoryRecord.model_validate(record.model_dump(mode="python")) for record in records
    )
    if not requested:
        raise ValueError("repository registration requires at least one record")
    if len({record.id for record in requested}) != len(requested):
        raise ValueError("repository registration requires unique repository IDs")
    canonical_urls = tuple(
        record.canonical_url for record in requested if record.canonical_url is not None
    )
    if len(set(canonical_urls)) != len(canonical_urls):
        raise ValueError("repository registration requires unique GitLab URLs")
    return requested


def _job_from_row(row: dict[str, object]) -> IndexingJob:
    return IndexingJob(
        id=str(row["id"]),
        requested_repository_id=_optional_string(row.get("requested_repository_id")),
        status=IndexingJobStatus(str(row["status"])),
        stage=IndexingStage(str(row["stage"])),
        scan_run_id=_optional_string(row.get("scan_run_id")),
        error_code=_optional_string(row.get("error_code")),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        started_at=_optional_datetime(row.get("started_at")),
        completed_at=_optional_datetime(row.get("completed_at")),
        cancellation_requested_at=_optional_datetime(row.get("cancellation_requested_at")),
    )


def _event_from_row(row: dict[str, object]) -> IndexingJobEvent:
    return IndexingJobEvent(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        level=JobEventLevel(str(row["level"])),
        stage=IndexingStage(str(row["stage"])),
        code=str(row["code"]),
        message=str(row["message"]),
        error_type=_optional_string(row.get("error_type")),
        created_at=_datetime(row["created_at"]),
    )


def _index_metadata_tuple(record: RepositoryRecord) -> tuple[object, ...]:
    return (
        record.status,
        record.knowledge_base_uri,
        record.graph_node_count,
        record.graph_relationship_count,
        record.last_scan_run_id,
        record.indexed_at,
    )


def _validate_transition(
    existing: IndexingJob,
    status: IndexingJobStatus,
    stage: IndexingStage,
    error_code: str | None,
) -> None:
    if existing.status in {
        IndexingJobStatus.SUCCEEDED,
        IndexingJobStatus.FAILED,
        IndexingJobStatus.CANCELLED,
    }:
        raise ValueError("terminal indexing jobs cannot be updated")
    if existing.status is IndexingJobStatus.QUEUED:
        raise ValueError("an indexing job must be claimed before it can be updated")
    if status is IndexingJobStatus.QUEUED:
        raise ValueError("a running indexing job cannot return to queued")
    if _STAGE_ORDER[stage] < _STAGE_ORDER[existing.stage]:
        raise ValueError("indexing stage cannot move backwards")
    if status is IndexingJobStatus.RUNNING and stage not in {
        IndexingStage.MATERIALIZING,
        IndexingStage.SCANNING,
        IndexingStage.PUBLISHING,
    }:
        raise ValueError("running jobs require an active indexing stage")
    if status is IndexingJobStatus.SUCCEEDED and stage is not IndexingStage.COMPLETE:
        raise ValueError("succeeded jobs must use the complete stage")
    if status is IndexingJobStatus.FAILED and stage is IndexingStage.COMPLETE:
        raise ValueError("failed jobs must retain the stage that failed")
    if status is IndexingJobStatus.CANCELLED and stage not in {
        IndexingStage.MATERIALIZING,
        IndexingStage.SCANNING,
    }:
        raise ValueError("cancelled active jobs must stop before publication")
    if status is IndexingJobStatus.CANCELLED and existing.cancellation_requested_at is None:
        raise ValueError("cancellation must be requested before a running job is cancelled")
    if (
        existing.cancellation_requested_at is not None
        and status not in {IndexingJobStatus.RUNNING, IndexingJobStatus.CANCELLED}
    ):
        raise ValueError("job cancellation has been requested")
    if stage is IndexingStage.PUBLISHING and existing.cancellation_requested_at is not None:
        raise ValueError("job cancellation has been requested")
    if status is IndexingJobStatus.FAILED and error_code is None:
        raise ValueError("failed jobs require an error code")
    if status is not IndexingJobStatus.FAILED and error_code is not None:
        raise ValueError("only failed jobs may have an error code")


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("PostgreSQL returned a non-datetime timestamp")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _optional_datetime(value: object) -> datetime | None:
    return _datetime(value) if value is not None else None
