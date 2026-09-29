from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, Literal, Protocol

from ai_code_intelligence.application.models import ScanResult
from ai_code_intelligence.domain.knowledge import (
    KnowledgeDocument,
    KnowledgeManifest,
    StoredDocument,
)
from ai_code_intelligence.domain.models import AgentInvocation, FrozenModel
from ai_code_intelligence.domain.repositories import IndexingJob, IndexingJobStatus
from ai_code_intelligence.knowledge.human import human_readable_document
from ai_code_intelligence.knowledge.store import DocumentStore, ManifestPublisher
from ai_code_intelligence.utils.ids import stable_id


class ReaderBackfillStore(DocumentStore, ManifestPublisher, Protocol):
    """Document store capable of atomically advancing the latest manifest."""


class ReaderBackfillMetadataStore(Protocol):
    """Minimal scan-audit port used to catalogue newly derived reader documents."""

    def latest_scan(self) -> ScanResult | None: ...

    def record_scan(
        self,
        result: ScanResult,
        invocations: tuple[AgentInvocation, ...],
        documents: tuple[KnowledgeDocument, ...],
        stored: tuple[StoredDocument, ...],
    ) -> None: ...

    def close(self) -> None: ...


class ReaderBackfillJobStore(Protocol):
    """Read-only queue view used to reject a publication race."""

    def list_jobs(self, *, status: IndexingJobStatus | None = None) -> tuple[IndexingJob, ...]: ...


class ReaderEditionBackfillResult(FrozenModel):
    """Result of an idempotent evidence-to-reader manifest upgrade."""

    scan_run_id: str
    reader_knowledge_uris: tuple[str, ...]
    manifest_uri: str | None
    already_current: bool


class ReaderEditionBackfillService:
    """Publish reader editions for an existing complete evidence manifest without model calls."""

    def __init__(
        self,
        documents: ReaderBackfillStore,
        jobs: ReaderBackfillJobStore,
        metadata_factory: Callable[[], ReaderBackfillMetadataStore],
    ) -> None:
        self._documents = documents
        self._jobs = jobs
        self._metadata_factory = metadata_factory

    def run(self) -> ReaderEditionBackfillResult:
        """Derive, catalogue, and publish reader pointers with the manifest written last."""

        active = self._jobs.list_jobs(status=IndexingJobStatus.QUEUED) + self._jobs.list_jobs(
            status=IndexingJobStatus.RUNNING
        )
        if active:
            raise RuntimeError("reader-edition backfill requires an idle indexing queue")
        manifest = self._documents.latest_manifest()
        if manifest is None:
            raise RuntimeError("knowledge base has not been published yet")
        if manifest.schema_version == 2:
            reader_uris = _manifest_reader_uris(manifest)
            for uri in reader_uris:
                self._documents.get(uri)
            return ReaderEditionBackfillResult(
                scan_run_id=manifest.scan_run_id,
                reader_knowledge_uris=reader_uris,
                manifest_uri=None,
                already_current=True,
            )

        evidence: list[KnowledgeDocument] = []
        for repository in manifest.repositories:
            evidence.append(
                _evidence_document(
                    markdown=self._documents.get(repository.knowledge_uri),
                    kind="repository",
                    title=f"{repository.id} repository knowledge",
                    repository_id=repository.id,
                    source_uri=repository.knowledge_uri,
                    generated_at=manifest.generated_at,
                )
            )
        evidence.append(
            _evidence_document(
                markdown=self._documents.get(manifest.central_knowledge_uri),
                kind="central",
                title="Central Knowledge Base",
                repository_id=None,
                source_uri=manifest.central_knowledge_uri,
                generated_at=manifest.generated_at,
            )
        )
        readers = tuple(human_readable_document(document) for document in evidence)
        stored = tuple(self._documents.put(document) for document in readers)

        metadata = self._metadata_factory()
        try:
            scan = metadata.latest_scan()
            if scan is None or scan.scan_run_id != manifest.scan_run_id:
                raise RuntimeError("latest PostgreSQL scan does not match the knowledge manifest")
            metadata.record_scan(scan, (), readers, stored)
        finally:
            metadata.close()

        repository_stored = stored[:-1]
        central_stored = stored[-1]
        payload: Mapping[str, Any] = {
            "schemaVersion": 2,
            "scanRunId": manifest.scan_run_id,
            "generatedAt": manifest.generated_at.isoformat(),
            "repositories": [
                {
                    "id": repository.id,
                    "commitSha": repository.commit_sha,
                    "knowledgeUri": repository.knowledge_uri,
                    "humanKnowledgeUri": human.uri,
                }
                for repository, human in zip(
                    manifest.repositories,
                    repository_stored,
                    strict=True,
                )
            ],
            "centralKnowledgeUri": manifest.central_knowledge_uri,
            "humanCentralKnowledgeUri": central_stored.uri,
            "graph": {
                "neo4jSnapshotId": manifest.graph.neo4j_snapshot_id,
                "statistics": manifest.graph.statistics.model_dump(mode="json"),
            },
        }
        manifest_uri = self._documents.publish_manifest(payload)
        return ReaderEditionBackfillResult(
            scan_run_id=manifest.scan_run_id,
            reader_knowledge_uris=tuple(item.uri for item in stored),
            manifest_uri=manifest_uri,
            already_current=False,
        )


def _evidence_document(
    *,
    markdown: str,
    kind: Literal["repository", "central"],
    title: str,
    repository_id: str | None,
    source_uri: str,
    generated_at: datetime,
) -> KnowledgeDocument:
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    return KnowledgeDocument.model_validate(
        {
            "id": stable_id("knowledge", "reader-backfill", source_uri, digest),
            "kind": kind,
            "title": title,
            "markdown": markdown,
            "repository_id": repository_id,
            "generated_at": generated_at,
        }
    )


def _manifest_reader_uris(manifest: KnowledgeManifest) -> tuple[str, ...]:
    values = (
        *(repository.human_knowledge_uri for repository in manifest.repositories),
        manifest.human_central_knowledge_uri,
    )
    if any(uri is None for uri in values):
        raise RuntimeError("reader-edition manifest is incomplete")
    return tuple(uri for uri in values if uri is not None)
