from __future__ import annotations

from ai_code_intelligence.embeddings.models import EmbeddingChunk, EmbeddingDocument
from ai_code_intelligence.utils.ids import stable_id


class DocumentChunker:
    """Produces deterministic, provider-bounded chunks without dropping content."""

    def __init__(
        self,
        size: int,
        overlap: int,
        *,
        max_utf8_bytes: int | None = None,
    ) -> None:
        if size < 1 or overlap < 0 or overlap >= size:
            raise ValueError("chunk size must be positive and overlap must be smaller")
        if max_utf8_bytes is not None and max_utf8_bytes < 1:
            raise ValueError("maximum UTF-8 bytes must be positive")
        if max_utf8_bytes is not None and overlap >= max_utf8_bytes:
            raise ValueError("chunk overlap must be smaller than maximum UTF-8 bytes")
        self._size = size
        self._overlap = overlap
        self._max_utf8_bytes = max_utf8_bytes

    def chunk(self, document: EmbeddingDocument) -> tuple[EmbeddingChunk, ...]:
        chunks: list[EmbeddingChunk] = []
        start = 0
        while start < len(document.content):
            tentative_end = min(len(document.content), start + self._size)
            if self._max_utf8_bytes is not None:
                tentative_end = _utf8_bounded_end(
                    document.content,
                    start,
                    tentative_end,
                    self._max_utf8_bytes,
                )
            end = tentative_end
            if tentative_end < len(document.content):
                midpoint = start + max(1, (tentative_end - start) // 2)
                newline = document.content.rfind("\n", midpoint, tentative_end)
                if newline > start:
                    end = newline + 1
            content = document.content[start:end]
            chunks.append(
                EmbeddingChunk(
                    id=stable_id("embedding-chunk", document.id, len(chunks), content),
                    document_id=document.id,
                    chunk_index=len(chunks),
                    kind=document.kind,
                    source_uri=document.source_uri,
                    content=content,
                    repository_id=document.repository_id,
                    metadata=document.metadata,
                )
            )
            if end >= len(document.content):
                break
            start = max(start + 1, end - self._overlap)
        return tuple(chunks)


def _utf8_bounded_end(text: str, start: int, end: int, maximum: int) -> int:
    """Return the largest character boundary within the configured byte budget."""

    candidate = text[start:end]
    if len(candidate.encode("utf-8")) <= maximum:
        return end
    low = 0
    high = len(candidate)
    while low < high:
        middle = (low + high + 1) // 2
        if len(candidate[:middle].encode("utf-8")) <= maximum:
            low = middle
        else:
            high = middle - 1
    if low == 0:
        raise ValueError("one character exceeds the embedding UTF-8 byte limit")
    return start + low
