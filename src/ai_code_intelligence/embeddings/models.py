from __future__ import annotations

from pydantic import Field

from ai_code_intelligence.domain.models import FrozenModel


class EmbeddingDocument(FrozenModel):
    """Selected source document before deterministic character chunking."""

    id: str
    kind: str
    source_uri: str
    content: str
    repository_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddingChunk(FrozenModel):
    """Deterministic retrievable text chunk with source metadata."""

    id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    kind: str
    source_uri: str
    content: str
    repository_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddedChunk(FrozenModel):
    """Text chunk paired with its provider model and vector."""

    chunk: EmbeddingChunk
    vector: tuple[float, ...]
    model_id: str
