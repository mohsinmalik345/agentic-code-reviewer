from __future__ import annotations

from typing import Protocol

from ai_code_intelligence.domain.models import FrozenModel


class GraphEvidence(FrozenModel):
    """One human-readable fact retrieved from the published Neo4j snapshot."""

    title: str
    repository_id: str | None = None
    section: str
    source_uri: str
    content: str


class GraphEvidenceRetriever(Protocol):
    """Retrieves bounded graph facts relevant to an architecture question."""

    def retrieve(
        self,
        question: str,
        *,
        repository_id: str | None,
        snapshot_id: str,
    ) -> tuple[GraphEvidence, ...]: ...


class EmptyGraphEvidenceRetriever:
    """No-op graph retriever used when graph-grounded chat is disabled."""

    def retrieve(
        self,
        question: str,
        *,
        repository_id: str | None,
        snapshot_id: str,
    ) -> tuple[GraphEvidence, ...]:
        del question, repository_id, snapshot_id
        return ()
