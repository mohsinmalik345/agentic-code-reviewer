from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ai_code_intelligence.domain.knowledge import KnowledgeDocument
from ai_code_intelligence.embeddings.chunker import DocumentChunker
from ai_code_intelligence.embeddings.models import EmbeddedChunk, EmbeddingChunk, EmbeddingDocument
from ai_code_intelligence.embeddings.provider import EmbeddingProvider
from ai_code_intelligence.embeddings.vector_store import VectorStore
from ai_code_intelligence.utils.ids import stable_id

_LOGGER = logging.getLogger(__name__)


class EmbeddingIndexer:
    """Chunks, embeds, and atomically upserts selected documents through abstractions."""

    def __init__(
        self,
        chunker: DocumentChunker,
        provider: EmbeddingProvider,
        store: VectorStore,
        concurrency: int = 3,
    ) -> None:
        self._chunker = chunker
        self._provider = provider
        self._store = store
        self._concurrency = concurrency

    def index(self, documents: tuple[EmbeddingDocument, ...]) -> int:
        """Replace the complete embedding snapshot."""

        embedded = self._embed(documents)
        self._store.replace(embedded)
        return len(embedded)

    def index_scoped(
        self,
        documents: tuple[EmbeddingDocument, ...],
        repository_ids: frozenset[str],
    ) -> int:
        """Replace changed repository and central chunks while retaining other repositories."""

        if not repository_ids:
            raise ValueError("scoped embedding replacement requires repository IDs")
        embedded = self._embed(documents)
        self._store.replace_scopes(embedded, repository_ids)
        return len(embedded)

    def index_delta(
        self,
        documents: tuple[EmbeddingDocument, ...],
        *,
        full_repository_ids: frozenset[str],
        selective_repository_ids: frozenset[str],
        stale_source_uris: frozenset[str],
        replace_central: bool,
    ) -> int:
        """Patch selected source/document scopes while retaining all other chunks."""

        embedded = self._embed(documents)
        self._store.replace_delta(
            embedded,
            full_repository_ids=full_repository_ids,
            selective_repository_ids=selective_repository_ids,
            stale_source_uris=stale_source_uris,
            replace_central=replace_central,
        )
        return len(embedded)

    def _embed(self, documents: tuple[EmbeddingDocument, ...]) -> tuple[EmbeddedChunk, ...]:
        chunks = tuple(chunk for document in documents for chunk in self._chunker.chunk(document))

        def embed(chunk: EmbeddingChunk) -> EmbeddedChunk:
            try:
                return EmbeddedChunk(
                    chunk=chunk,
                    vector=self._provider.embed(chunk.content),
                    model_id=self._provider.model_id,
                )
            except Exception:
                _LOGGER.exception(
                    "Embedding chunk failed",
                    extra={
                        "fields": {
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "repository_id": chunk.repository_id,
                            "source_uri": chunk.source_uri,
                            "character_count": len(chunk.content),
                            "utf8_byte_count": len(chunk.content.encode("utf-8")),
                            "embedding_model_id": self._provider.model_id,
                        }
                    },
                )
                raise

        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            embedded = tuple(executor.map(embed, chunks))
        return embedded


class KnowledgeEmbeddingScopeRefresher:
    """Replaces central vectors while removing retired repository scopes atomically."""

    def __init__(
        self,
        chunker: DocumentChunker,
        provider_factory: Callable[[], EmbeddingProvider],
        store_factory: Callable[[], VectorStore],
        concurrency: int = 3,
    ) -> None:
        self._chunker = chunker
        self._provider_factory = provider_factory
        self._store_factory = store_factory
        self._concurrency = concurrency

    def replace_central_and_remove(
        self,
        central: KnowledgeDocument,
        repository_ids: frozenset[str],
    ) -> int:
        """Embed one restored central document and retire the supplied repository scopes."""

        if central.kind != "central" or central.repository_id is not None:
            raise ValueError("embedding scope refresh requires a central evidence document")
        if not repository_ids:
            raise ValueError("embedding scope refresh requires retired repository IDs")
        document = EmbeddingDocument(
            id=stable_id("embedding-document", central.id),
            kind="knowledge-central",
            source_uri=f"knowledge://{central.id}",
            content=central.markdown,
            metadata={"title": central.title},
        )
        store = self._store_factory()
        try:
            indexer = EmbeddingIndexer(
                self._chunker,
                self._provider_factory(),
                store,
                self._concurrency,
            )
            return indexer.index_scoped((document,), repository_ids)
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
