from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock, RLock
from time import sleep
from typing import Any, Literal, Protocol
from uuid import uuid4

from psycopg import errors, sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_code_intelligence.jira.models import JiraConnection, JiraProjectMapping

_SCHEMA_VERSION = 1
_PATH_LOCKS: dict[Path, RLock] = {}
_PATH_LOCKS_GUARD = Lock()


class JiraConfigurationStore(Protocol):
    """Persistence port for Jira connections and repository-to-project mappings."""

    def initialize(self) -> None: ...

    def upsert_connection(self, connection: JiraConnection) -> JiraConnection: ...

    def get_connection(self, connection_id: str) -> JiraConnection | None: ...

    def list_connections(self, *, include_disabled: bool = True) -> tuple[JiraConnection, ...]: ...

    def delete_connection(self, connection_id: str) -> bool: ...

    def upsert_mapping(self, mapping: JiraProjectMapping) -> JiraProjectMapping: ...

    def get_mapping(self, mapping_id: str) -> JiraProjectMapping | None: ...

    def list_mappings(
        self,
        *,
        connection_id: str | None = None,
        repository_id: str | None = None,
    ) -> tuple[JiraProjectMapping, ...]: ...

    def delete_mapping(self, mapping_id: str) -> bool: ...


class _JiraConfigurationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    connections: dict[str, JiraConnection] = Field(default_factory=dict)
    mappings: dict[str, JiraProjectMapping] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_relations_and_keys(self) -> _JiraConfigurationState:
        if any(key != connection.id for key, connection in self.connections.items()):
            raise ValueError("Jira connection keys must match connection IDs")
        if any(key != mapping.id for key, mapping in self.mappings.items()):
            raise ValueError("Jira mapping keys must match mapping IDs")
        if any(mapping.connection_id not in self.connections for mapping in self.mappings.values()):
            raise ValueError("Jira mappings must reference a configured connection")

        base_urls = [connection.base_url for connection in self.connections.values()]
        if len(base_urls) != len(set(base_urls)):
            raise ValueError("Jira connection base URLs must be unique")
        pairs = [(mapping.connection_id, mapping.repository_id) for mapping in self.mappings.values()]
        if len(pairs) != len(set(pairs)):
            raise ValueError("a repository can have only one mapping per Jira connection")
        return self


class FileJiraConfigurationStore(JiraConfigurationStore):
    """Atomic, thread-safe JSON storage for Jira configuration metadata."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._lock = _path_lock(self._path)
        self.initialize()

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        """Create an empty schema-v1 document or validate the existing document."""

        with self._lock:
            if self._path.exists():
                self._read()
            else:
                self._write(_JiraConfigurationState())

    def upsert_connection(self, connection: JiraConnection) -> JiraConnection:
        requested = _validated_connection(connection)
        with self._lock:
            state = self._read()
            duplicate = next(
                (
                    item
                    for item in state.connections.values()
                    if item.id != requested.id and item.base_url == requested.base_url
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"Jira base URL is already configured as connection {duplicate.id}")

            existing = state.connections.get(requested.id)
            if existing is None:
                stored = requested
            else:
                stored = requested.model_copy(
                    update={
                        "created_at": existing.created_at,
                        "updated_at": existing.updated_at,
                    }
                )
                stored = _validated_connection(stored)
                if _connection_payload(stored) == _connection_payload(existing):
                    return existing
                stored = stored.model_copy(update={"updated_at": _next_updated_at(existing.updated_at)})
                stored = _validated_connection(stored)

            self._write(state.model_copy(update={"connections": {**state.connections, stored.id: stored}}))
            return stored

    def get_connection(self, connection_id: str) -> JiraConnection | None:
        with self._lock:
            return self._read().connections.get(connection_id)

    def list_connections(self, *, include_disabled: bool = True) -> tuple[JiraConnection, ...]:
        with self._lock:
            connections = self._read().connections.values()
            return tuple(
                sorted(
                    (connection for connection in connections if include_disabled or connection.enabled),
                    key=lambda connection: connection.id,
                )
            )

    def delete_connection(self, connection_id: str) -> bool:
        with self._lock:
            state = self._read()
            if connection_id not in state.connections:
                return False
            connections = dict(state.connections)
            del connections[connection_id]
            mappings = {
                mapping_id: mapping
                for mapping_id, mapping in state.mappings.items()
                if mapping.connection_id != connection_id
            }
            self._write(state.model_copy(update={"connections": connections, "mappings": mappings}))
            return True

    def upsert_mapping(self, mapping: JiraProjectMapping) -> JiraProjectMapping:
        requested = _validated_mapping(mapping)
        with self._lock:
            state = self._read()
            if requested.connection_id not in state.connections:
                raise ValueError(f"unknown Jira connection: {requested.connection_id}")
            duplicate = next(
                (
                    item
                    for item in state.mappings.values()
                    if item.id != requested.id
                    and item.connection_id == requested.connection_id
                    and item.repository_id == requested.repository_id
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"repository is already mapped for this Jira connection as {duplicate.id}")

            existing = state.mappings.get(requested.id)
            if existing is None:
                stored = requested
            else:
                if (
                    existing.connection_id != requested.connection_id
                    or existing.repository_id != requested.repository_id
                ):
                    raise ValueError("a Jira mapping cannot be reassigned")
                stored = requested.model_copy(
                    update={
                        "created_at": existing.created_at,
                        "updated_at": existing.updated_at,
                    }
                )
                stored = _validated_mapping(stored)
                if _mapping_payload(stored) == _mapping_payload(existing):
                    return existing
                stored = stored.model_copy(update={"updated_at": _next_updated_at(existing.updated_at)})
                stored = _validated_mapping(stored)

            self._write(state.model_copy(update={"mappings": {**state.mappings, stored.id: stored}}))
            return stored

    def get_mapping(self, mapping_id: str) -> JiraProjectMapping | None:
        with self._lock:
            return self._read().mappings.get(mapping_id)

    def list_mappings(
        self,
        *,
        connection_id: str | None = None,
        repository_id: str | None = None,
    ) -> tuple[JiraProjectMapping, ...]:
        with self._lock:
            mappings = self._read().mappings.values()
            return tuple(
                sorted(
                    (
                        mapping
                        for mapping in mappings
                        if (connection_id is None or mapping.connection_id == connection_id)
                        and (repository_id is None or mapping.repository_id == repository_id)
                    ),
                    key=lambda mapping: mapping.id,
                )
            )

    def delete_mapping(self, mapping_id: str) -> bool:
        with self._lock:
            state = self._read()
            if mapping_id not in state.mappings:
                return False
            mappings = dict(state.mappings)
            del mappings[mapping_id]
            self._write(state.model_copy(update={"mappings": mappings}))
            return True

    def _read(self) -> _JiraConfigurationState:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Jira configuration root must be a JSON object")
            if raw.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported Jira configuration schema version: {raw.get('schema_version')!r}"
                )
            return _JiraConfigurationState.model_validate(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Jira configuration is not valid JSON: {self._path}") from error
        except OSError as error:
            raise RuntimeError(f"unable to read Jira configuration: {self._path}") from error

    def _write(self, state: _JiraConfigurationState) -> None:
        validated = _JiraConfigurationState.model_validate(state.model_dump(mode="python"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid4()}.tmp")
        payload = json.dumps(
            validated.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_retry(temporary, self._path)
        except OSError as error:
            raise RuntimeError(f"unable to write Jira configuration: {self._path}") from error
        finally:
            temporary.unlink(missing_ok=True)


class PostgresJiraConfigurationStore(JiraConfigurationStore):
    """PostgreSQL-backed Jira configuration with transactional upserts."""

    def __init__(self, connection_string: str, schema_name: str) -> None:
        self._schema = schema_name
        self._pool = ConnectionPool(
            conninfo=connection_string,
            min_size=1,
            max_size=5,
            open=True,
        )

    def close(self) -> None:
        self._pool.close()

    def initialize(self) -> None:
        """Idempotently create Jira configuration tables for existing deployments."""

        schema = sql.Identifier(self._schema)
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.jira_connections (
                        id text PRIMARY KEY
                            CHECK (id ~ '^[A-Za-z0-9][A-Za-z0-9:._-]{{0,127}}$'),
                        name text NOT NULL CHECK (length(btrim(name)) BETWEEN 1 AND 200),
                        edition text NOT NULL CHECK (edition IN ('cloud', 'data_center')),
                        base_url text NOT NULL UNIQUE
                            CHECK (length(base_url) BETWEEN 1 AND 2048),
                        auth_type text NOT NULL
                            CHECK (auth_type IN ('api_token', 'personal_access_token')),
                        username text CHECK (
                            username IS NULL OR length(btrim(username)) BETWEEN 1 AND 320
                        ),
                        credential_env text NOT NULL CHECK (
                            credential_env ~ '^[A-Za-z_][A-Za-z0-9_]{{0,127}}$'
                        ),
                        enabled boolean NOT NULL DEFAULT true,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        CHECK (updated_at >= created_at),
                        CHECK (
                            (edition = 'cloud' AND auth_type = 'api_token'
                                AND username IS NOT NULL)
                            OR
                            (edition = 'data_center'
                                AND auth_type = 'personal_access_token'
                                AND username IS NULL)
                        )
                    )
                    """
                ).format(schema)
            )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {} ON {}.jira_connections(enabled, id)
                    """
                ).format(
                    sql.Identifier("jira_connections_enabled_idx"),
                    schema,
                )
            )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.jira_project_mappings (
                        id text PRIMARY KEY
                            CHECK (id ~ '^[A-Za-z0-9][A-Za-z0-9:._-]{{0,127}}$'),
                        connection_id text NOT NULL
                            REFERENCES {}.jira_connections(id) ON DELETE CASCADE,
                        repository_id text NOT NULL
                            REFERENCES {}.repositories(id) ON DELETE CASCADE,
                        jira_project_key text NOT NULL
                            CHECK (jira_project_key ~ '^[A-Z][A-Z0-9_]{{0,63}}$'),
                        acceptance_criteria_fields text[] NOT NULL DEFAULT '{{}}'::text[]
                            CHECK (cardinality(acceptance_criteria_fields) <= 32),
                        issue_key_pattern text NOT NULL CHECK (
                            length(issue_key_pattern) BETWEEN 1 AND 1000
                        ),
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        UNIQUE (connection_id, repository_id),
                        CHECK (updated_at >= created_at)
                    )
                    """
                ).format(schema, schema, schema)
            )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {}
                    ON {}.jira_project_mappings(repository_id)
                    """
                ).format(
                    sql.Identifier("jira_project_mappings_repository_idx"),
                    schema,
                )
            )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {}
                    ON {}.jira_project_mappings(connection_id, jira_project_key)
                    """
                ).format(
                    sql.Identifier("jira_project_mappings_project_idx"),
                    schema,
                )
            )
            connection.commit()

    def upsert_connection(self, connection: JiraConnection) -> JiraConnection:
        requested = _validated_connection(connection)
        with self._pool.connection() as db_connection, db_connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.jira_connections AS current
                          (id, name, edition, base_url, auth_type, username,
                           credential_env, enabled, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                          name = EXCLUDED.name,
                          edition = EXCLUDED.edition,
                          base_url = EXCLUDED.base_url,
                          auth_type = EXCLUDED.auth_type,
                          username = EXCLUDED.username,
                          credential_env = EXCLUDED.credential_env,
                          enabled = EXCLUDED.enabled,
                          updated_at = CASE
                            WHEN ROW(
                              current.name, current.edition, current.base_url,
                              current.auth_type, current.username,
                              current.credential_env, current.enabled
                            ) IS DISTINCT FROM ROW(
                              EXCLUDED.name, EXCLUDED.edition, EXCLUDED.base_url,
                              EXCLUDED.auth_type, EXCLUDED.username,
                              EXCLUDED.credential_env, EXCLUDED.enabled
                            )
                            THEN GREATEST(
                              now(), current.updated_at + interval '1 microsecond'
                            )
                            ELSE current.updated_at
                          END
                        RETURNING *
                        """
                    ).format(sql.Identifier(self._schema)),
                    (
                        requested.id,
                        requested.name,
                        _enum_value(requested.edition),
                        str(requested.base_url),
                        _enum_value(requested.auth_type),
                        requested.username,
                        requested.credential_env,
                        requested.enabled,
                        requested.created_at,
                        requested.updated_at,
                    ),
                )
                row = cursor.fetchone()
                assert row is not None
                db_connection.commit()
                return _connection_from_row(row)
            except errors.UniqueViolation as error:
                db_connection.rollback()
                raise ValueError("Jira base URL is already configured") from error
            except Exception:
                db_connection.rollback()
                raise

    def get_connection(self, connection_id: str) -> JiraConnection | None:
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.jira_connections WHERE id = %s").format(
                    sql.Identifier(self._schema)
                ),
                (connection_id,),
            )
            row = cursor.fetchone()
        return _connection_from_row(row) if row is not None else None

    def list_connections(self, *, include_disabled: bool = True) -> tuple[JiraConnection, ...]:
        where = sql.SQL("") if include_disabled else sql.SQL(" WHERE enabled")
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.jira_connections").format(sql.Identifier(self._schema))
                + where
                + sql.SQL(" ORDER BY id")
            )
            rows = cursor.fetchall()
        return tuple(_connection_from_row(row) for row in rows)

    def delete_connection(self, connection_id: str) -> bool:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {}.jira_connections WHERE id = %s RETURNING id").format(
                    sql.Identifier(self._schema)
                ),
                (connection_id,),
            )
            deleted = cursor.fetchone() is not None
            connection.commit()
        return deleted

    def upsert_mapping(self, mapping: JiraProjectMapping) -> JiraProjectMapping:
        requested = _validated_mapping(mapping)
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            try:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.jira_project_mappings AS current
                          (id, connection_id, repository_id, jira_project_key,
                           acceptance_criteria_fields, issue_key_pattern,
                           created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                          jira_project_key = EXCLUDED.jira_project_key,
                          acceptance_criteria_fields = EXCLUDED.acceptance_criteria_fields,
                          issue_key_pattern = EXCLUDED.issue_key_pattern,
                          updated_at = CASE
                            WHEN ROW(
                              current.jira_project_key,
                              current.acceptance_criteria_fields,
                              current.issue_key_pattern
                            ) IS DISTINCT FROM ROW(
                              EXCLUDED.jira_project_key,
                              EXCLUDED.acceptance_criteria_fields,
                              EXCLUDED.issue_key_pattern
                            )
                            THEN GREATEST(
                              now(), current.updated_at + interval '1 microsecond'
                            )
                            ELSE current.updated_at
                          END
                        WHERE current.connection_id = EXCLUDED.connection_id
                          AND current.repository_id = EXCLUDED.repository_id
                        RETURNING *
                        """
                    ).format(sql.Identifier(self._schema)),
                    (
                        requested.id,
                        requested.connection_id,
                        requested.repository_id,
                        requested.jira_project_key,
                        list(requested.acceptance_criteria_fields),
                        requested.issue_key_pattern,
                        requested.created_at,
                        requested.updated_at,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("a Jira mapping cannot be reassigned")
                connection.commit()
                return _mapping_from_row(row)
            except errors.ForeignKeyViolation as error:
                connection.rollback()
                raise ValueError("Jira mapping references an unknown connection or repository") from error
            except errors.UniqueViolation as error:
                connection.rollback()
                raise ValueError("repository is already mapped for this Jira connection") from error
            except Exception:
                connection.rollback()
                raise

    def get_mapping(self, mapping_id: str) -> JiraProjectMapping | None:
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.jira_project_mappings WHERE id = %s").format(
                    sql.Identifier(self._schema)
                ),
                (mapping_id,),
            )
            row = cursor.fetchone()
        return _mapping_from_row(row) if row is not None else None

    def list_mappings(
        self,
        *,
        connection_id: str | None = None,
        repository_id: str | None = None,
    ) -> tuple[JiraProjectMapping, ...]:
        clauses: list[sql.Composable] = []
        parameters: list[str] = []
        if connection_id is not None:
            clauses.append(sql.SQL("connection_id = %s"))
            parameters.append(connection_id)
        if repository_id is not None:
            clauses.append(sql.SQL("repository_id = %s"))
            parameters.append(repository_id)
        where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")
        with self._pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                sql.SQL("SELECT * FROM {}.jira_project_mappings").format(sql.Identifier(self._schema))
                + where
                + sql.SQL(" ORDER BY id"),
                tuple(parameters),
            )
            rows = cursor.fetchall()
        return tuple(_mapping_from_row(row) for row in rows)

    def delete_mapping(self, mapping_id: str) -> bool:
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {}.jira_project_mappings WHERE id = %s RETURNING id").format(
                    sql.Identifier(self._schema)
                ),
                (mapping_id,),
            )
            deleted = cursor.fetchone() is not None
            connection.commit()
        return deleted


def _path_lock(path: Path) -> RLock:
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(path, RLock())


def _replace_with_retry(source: Path, target: Path) -> None:
    """Handle short-lived Windows/OneDrive destination locks without losing atomicity."""

    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            sleep(0.02 * (2**attempt))


def _next_updated_at(previous: datetime) -> datetime:
    now = datetime.now(UTC)
    return max(now, previous + timedelta(microseconds=1))


def _validated_connection(connection: JiraConnection) -> JiraConnection:
    return JiraConnection.model_validate(connection.model_dump(mode="python"))


def _validated_mapping(mapping: JiraProjectMapping) -> JiraProjectMapping:
    return JiraProjectMapping.model_validate(mapping.model_dump(mode="python"))


def _connection_payload(connection: JiraConnection) -> dict[str, object]:
    return connection.model_dump(mode="python", exclude={"created_at", "updated_at"})


def _mapping_payload(mapping: JiraProjectMapping) -> dict[str, object]:
    return mapping.model_dump(mode="python", exclude={"created_at", "updated_at"})


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _connection_from_row(row: dict[str, Any]) -> JiraConnection:
    return JiraConnection.model_validate(row)


def _mapping_from_row(row: dict[str, Any]) -> JiraProjectMapping:
    payload = dict(row)
    payload["acceptance_criteria_fields"] = tuple(payload["acceptance_criteria_fields"] or ())
    return JiraProjectMapping.model_validate(payload)
