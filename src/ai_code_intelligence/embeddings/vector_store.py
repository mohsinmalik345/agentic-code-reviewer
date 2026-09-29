from __future__ import annotations

import json
from typing import Protocol

from psycopg import sql
from psycopg_pool import ConnectionPool

from ai_code_intelligence.embeddings.models import EmbeddedChunk


class VectorStore(Protocol):
    """Replaceable vector-store port."""

    def replace(self, values: tuple[EmbeddedChunk, ...]) -> None: ...

    def replace_scopes(
        self,
        values: tuple[EmbeddedChunk, ...],
        repository_ids: frozenset[str],
    ) -> None: ...

    def replace_delta(
        self,
        values: tuple[EmbeddedChunk, ...],
        *,
        full_repository_ids: frozenset[str],
        selective_repository_ids: frozenset[str],
        stale_source_uris: frozenset[str],
        replace_central: bool,
    ) -> None: ...


class PostgresVectorStore:
    """Stores embeddings in PostgreSQL/pgvector with idempotent chunk keys."""

    def __init__(self, connection_string: str, schema_name: str) -> None:
        self._schema = schema_name
        self._pool = ConnectionPool(conninfo=connection_string, min_size=1, max_size=5, open=True)

    def close(self) -> None:
        self._pool.close()

    def replace(self, values: tuple[EmbeddedChunk, ...]) -> None:
        """Atomically publish exactly the chunks in the latest portfolio snapshot."""

        self._replace(values, None)

    def replace_scopes(
        self,
        values: tuple[EmbeddedChunk, ...],
        repository_ids: frozenset[str],
    ) -> None:
        """Atomically replace changed repository scopes and the global central scope."""

        if not repository_ids:
            raise ValueError("scoped embedding replacement requires repository IDs")
        unexpected = {
            item.chunk.repository_id
            for item in values
            if item.chunk.repository_id is not None
            and item.chunk.repository_id not in repository_ids
        }
        if unexpected:
            raise ValueError(
                "scoped embeddings contain unchanged repositories: " + ", ".join(sorted(unexpected))
            )
        self._replace(values, repository_ids)

    def replace_delta(
        self,
        values: tuple[EmbeddedChunk, ...],
        *,
        full_repository_ids: frozenset[str],
        selective_repository_ids: frozenset[str],
        stale_source_uris: frozenset[str],
        replace_central: bool,
    ) -> None:
        """Atomically replace full repositories and selected source/document scopes."""

        overlap = full_repository_ids.intersection(selective_repository_ids)
        if overlap:
            raise ValueError("full and selective embedding repository scopes must be disjoint")
        allowed_ids = full_repository_ids | selective_repository_ids
        unexpected = {
            item.chunk.repository_id
            for item in values
            if item.chunk.repository_id is not None
            and item.chunk.repository_id not in allowed_ids
        }
        if unexpected:
            raise ValueError(
                "delta embeddings contain out-of-scope repositories: "
                + ", ".join(sorted(unexpected))
            )
        self._replace_delta(
            values,
            full_repository_ids,
            selective_repository_ids,
            stale_source_uris,
            replace_central,
        )

    def _replace(
        self,
        values: tuple[EmbeddedChunk, ...],
        repository_ids: frozenset[str] | None,
    ) -> None:

        statement = sql.SQL(
            """
            INSERT INTO {}.embeddings
              (chunk_id, document_id, repository_id, kind, source_uri, chunk_index,
               content, metadata, embedding_model_id, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
              document_id = EXCLUDED.document_id,
              repository_id = EXCLUDED.repository_id,
              kind = EXCLUDED.kind,
              source_uri = EXCLUDED.source_uri,
              chunk_index = EXCLUDED.chunk_index,
              content = EXCLUDED.content,
              metadata = EXCLUDED.metadata,
              embedding_model_id = EXCLUDED.embedding_model_id,
              embedding = EXCLUDED.embedding,
              updated_at = now()
            """
        ).format(sql.Identifier(self._schema))
        rows = [
            (
                item.chunk.id,
                item.chunk.document_id,
                item.chunk.repository_id,
                item.chunk.kind,
                item.chunk.source_uri,
                item.chunk.chunk_index,
                item.chunk.content,
                json.dumps(item.chunk.metadata, separators=(",", ":")),
                item.model_id,
                "[" + ",".join(str(value) for value in item.vector) + "]",
            )
            for item in values
        ]
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE active_embedding_chunks (chunk_id text PRIMARY KEY) ON COMMIT DROP"
            )
            if rows:
                cursor.executemany(statement, rows)
                cursor.executemany(
                    "INSERT INTO active_embedding_chunks (chunk_id) VALUES (%s)",
                    [(item.chunk.id,) for item in values],
                )
            if repository_ids is None:
                cursor.execute(
                    sql.SQL(
                        "DELETE FROM {}.embeddings AS stored "
                        "WHERE NOT EXISTS ("
                        "SELECT 1 FROM active_embedding_chunks AS active "
                        "WHERE active.chunk_id = stored.chunk_id)"
                    ).format(sql.Identifier(self._schema))
                )
            else:
                cursor.execute(
                    sql.SQL(
                        "DELETE FROM {}.embeddings AS stored "
                        "WHERE (stored.repository_id IS NULL "
                        "OR stored.repository_id = ANY(%s::text[])) "
                        "AND NOT EXISTS ("
                        "SELECT 1 FROM active_embedding_chunks AS active "
                        "WHERE active.chunk_id = stored.chunk_id)"
                    ).format(sql.Identifier(self._schema)),
                    (sorted(repository_ids),),
                )
            connection.commit()

    def _replace_delta(
        self,
        values: tuple[EmbeddedChunk, ...],
        full_repository_ids: frozenset[str],
        selective_repository_ids: frozenset[str],
        stale_source_uris: frozenset[str],
        replace_central: bool,
    ) -> None:
        statement = sql.SQL(
            """
            INSERT INTO {}.embeddings
              (chunk_id, document_id, repository_id, kind, source_uri, chunk_index,
               content, metadata, embedding_model_id, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
              document_id = EXCLUDED.document_id,
              repository_id = EXCLUDED.repository_id,
              kind = EXCLUDED.kind,
              source_uri = EXCLUDED.source_uri,
              chunk_index = EXCLUDED.chunk_index,
              content = EXCLUDED.content,
              metadata = EXCLUDED.metadata,
              embedding_model_id = EXCLUDED.embedding_model_id,
              embedding = EXCLUDED.embedding,
              updated_at = now()
            """
        ).format(sql.Identifier(self._schema))
        rows = [
            (
                item.chunk.id,
                item.chunk.document_id,
                item.chunk.repository_id,
                item.chunk.kind,
                item.chunk.source_uri,
                item.chunk.chunk_index,
                item.chunk.content,
                json.dumps(item.chunk.metadata, separators=(",", ":")),
                item.model_id,
                "[" + ",".join(str(value) for value in item.vector) + "]",
            )
            for item in values
        ]
        with self._pool.connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE active_embedding_chunks (chunk_id text PRIMARY KEY) ON COMMIT DROP"
            )
            if rows:
                cursor.executemany(statement, rows)
                cursor.executemany(
                    "INSERT INTO active_embedding_chunks (chunk_id) VALUES (%s)",
                    [(item.chunk.id,) for item in values],
                )
            if full_repository_ids:
                cursor.execute(
                    sql.SQL(
                        "DELETE FROM {}.embeddings AS stored "
                        "WHERE stored.repository_id = ANY(%s::text[]) "
                        "AND NOT EXISTS (SELECT 1 FROM active_embedding_chunks AS active "
                        "WHERE active.chunk_id = stored.chunk_id)"
                    ).format(sql.Identifier(self._schema)),
                    (sorted(full_repository_ids),),
                )
            if selective_repository_ids:
                cursor.execute(
                    sql.SQL(
                        "DELETE FROM {}.embeddings AS stored "
                        "WHERE stored.repository_id = ANY(%s::text[]) "
                        "AND (stored.kind LIKE 'knowledge-%%' "
                        "OR stored.source_uri = ANY(%s::text[])) "
                        "AND NOT EXISTS (SELECT 1 FROM active_embedding_chunks AS active "
                        "WHERE active.chunk_id = stored.chunk_id)"
                    ).format(sql.Identifier(self._schema)),
                    (sorted(selective_repository_ids), sorted(stale_source_uris)),
                )
            if replace_central:
                cursor.execute(
                    sql.SQL(
                        "DELETE FROM {}.embeddings AS stored "
                        "WHERE stored.repository_id IS NULL "
                        "AND NOT EXISTS (SELECT 1 FROM active_embedding_chunks AS active "
                        "WHERE active.chunk_id = stored.chunk_id)"
                    ).format(sql.Identifier(self._schema))
                )
            connection.commit()
