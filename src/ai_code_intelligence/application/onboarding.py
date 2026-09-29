from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from ai_code_intelligence.application.indexing import IndexingArtifacts, IndexingPipeline
from ai_code_intelligence.domain.change import RepositoryChangeSet
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
from ai_code_intelligence.git.gitlab_checkout import (
    GitCheckout,
    validate_git_ref,
    validate_gitlab_url,
)
from ai_code_intelligence.knowledge.store import ManifestPublisher
from ai_code_intelligence.persistence.catalog import (
    IndexingJobEventStore,
    IndexingJobStore,
    RepositoryCatalog,
)
from ai_code_intelligence.utils.redaction import redact_sensitive_text

_LOGGER = logging.getLogger(__name__)
_SLUG_CHARACTER = re.compile(r"[^a-z0-9]+")
_MAX_BULK_REPOSITORIES = 50


class CheckoutManager(Protocol):
    """Materializes one registered remote repository at an exact commit."""

    def checkout(self, repository_id: str, url: str, ref: str | None = None) -> GitCheckout: ...

    def changes_since(
        self,
        repository_id: str,
        checkout: GitCheckout,
        last_indexed_commit: str | None,
    ) -> RepositoryChangeSet: ...


class OnboardingResult:
    """Repository registration paired with its durable portfolio-indexing job."""

    def __init__(self, repository: RepositoryRecord, job: IndexingJob) -> None:
        self.repository = repository
        self.job = job


@dataclass(frozen=True, slots=True)
class GitLabRepositoryRegistration:
    """Validated-at-the-boundary request to register one GitLab repository."""

    url: str
    repository_id: str | None = None
    name: str | None = None
    description: str | None = None
    ref: str | None = None


class BulkOnboardingResult:
    """Atomic repository registrations paired with one portfolio-indexing job."""

    def __init__(self, repositories: tuple[RepositoryRecord, ...], job: IndexingJob) -> None:
        self.repositories = repositories
        self.job = job


class GitLabSourceMigration:
    """Requested GitLab source for one existing local repository identity."""

    def __init__(self, repository_id: str, url: str, ref: str | None = None) -> None:
        self.repository_id = repository_id
        self.url = url
        self.ref = ref


class SourceMigrationResult:
    """Atomically migrated repositories and their single queued portfolio refresh."""

    def __init__(self, repositories: tuple[RepositoryRecord, ...], job: IndexingJob) -> None:
        self.repositories = repositories
        self.job = job


class JobCancellationRequested(RuntimeError):
    """Internal cooperative-control signal raised only at safe indexing boundaries."""


class RepositoryOnboardingService:
    """Registers GitLab repositories and runs whole-portfolio publication jobs."""

    def __init__(
        self,
        *,
        catalog: RepositoryCatalog,
        jobs: IndexingJobStore,
        events: IndexingJobEventStore,
        checkout: CheckoutManager,
        indexing: IndexingPipeline,
        manifest_publisher: ManifestPublisher,
        checkout_root: str | Path,
        allowed_gitlab_hosts: tuple[str, ...],
        index_embeddings: bool,
        heartbeat_seconds: float = 60,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self._catalog = catalog
        self._jobs = jobs
        self._events = events
        self._checkout = checkout
        self._indexing = indexing
        self._manifest_publisher = manifest_publisher
        self._checkout_root = Path(checkout_root).expanduser().resolve()
        self._allowed_hosts = allowed_gitlab_hosts
        self._index_embeddings = index_embeddings
        self._heartbeat_seconds = heartbeat_seconds

    def register_gitlab(
        self,
        url: str,
        *,
        repository_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        ref: str | None = None,
    ) -> OnboardingResult:
        """Validate, durably register, and enqueue a GitLab repository without blocking."""

        records = self._prepare_gitlab_registrations(
            (
                GitLabRepositoryRegistration(
                    url=url,
                    repository_id=repository_id,
                    name=name,
                    description=description,
                    ref=ref,
                ),
            )
        )
        stored_records, job, job_created = self._register_and_ensure_queued_job(
            records,
            requested_repository_id=records[0].id,
        )
        stored = stored_records[0]
        if job_created:
            self._record_event(
                job,
                level=JobEventLevel.INFO,
                code="JOB_QUEUED",
                message=f"Queued portfolio indexing for repository {stored.id}.",
            )
        return OnboardingResult(stored, job)

    def register_gitlab_repositories(
        self,
        registrations: tuple[GitLabRepositoryRegistration, ...],
    ) -> BulkOnboardingResult:
        """Atomically register 1-50 GitLab sources and queue one future portfolio refresh."""

        records = self._prepare_gitlab_registrations(registrations)
        stored, job, job_created = self._register_and_ensure_queued_job(records)
        if job_created:
            self._record_event(
                job,
                level=JobEventLevel.INFO,
                code="JOB_QUEUED",
                message=_batch_event_message(
                    "Queued one portfolio indexing job",
                    stored,
                ),
            )
        else:
            self._record_event(
                job,
                level=JobEventLevel.INFO,
                code="REPOSITORIES_ATTACHED",
                message=_batch_event_message(
                    "Attached registrations to the queued portfolio job",
                    stored,
                ),
            )
        return BulkOnboardingResult(stored, job)

    def enqueue_reindex(self, repository_id: str | None = None) -> IndexingJob:
        """Queue a commit-aware portfolio or repository-scoped refresh."""

        if repository_id is not None and self._catalog.get_repository(repository_id) is None:
            raise ValueError(f"unknown repository id: {repository_id}")
        existing = self._existing_active_job(repository_id)
        if existing is not None:
            return existing
        job = self._jobs.enqueue_job(IndexingJob.create(repository_id))
        scope = f"repository {repository_id}" if repository_id else "the full portfolio"
        self._record_event(
            job,
            level=JobEventLevel.INFO,
            code="JOB_QUEUED",
            message=f"Queued reindexing for {scope}.",
        )
        return job

    def cancel_job(self, job_id: str) -> IndexingJob:
        """Cancel queued work immediately or request a safe stop from the active worker."""

        existing = self._jobs.get_job(job_id)
        if existing is None:
            raise ValueError("indexing job was not found")
        updated = self._jobs.request_job_cancellation(job_id)
        if existing.cancellation_requested_at is not None:
            return updated
        if updated.status is IndexingJobStatus.CANCELLED:
            self._record_event(
                updated,
                level=JobEventLevel.WARNING,
                code="JOB_CANCELLED",
                message="The queued job was cancelled before a worker claimed it.",
            )
        else:
            self._record_event(
                updated,
                level=JobEventLevel.WARNING,
                code="CANCELLATION_REQUESTED",
                message=(
                    "Cancellation requested. The worker will stop at the next safe boundary; "
                    "an active Bedrock invocation cannot be interrupted."
                ),
            )
        return updated

    def migrate_local_sources_to_gitlab(
        self,
        sources: tuple[GitLabSourceMigration, ...],
    ) -> SourceMigrationResult:
        """Preflight and atomically migrate local sources, then queue one portfolio refresh."""

        if not sources or len({source.repository_id for source in sources}) != len(sources):
            raise ValueError("source migration requires nonempty, unique repository IDs")
        active = next(
            (
                job
                for job in self._jobs.list_jobs()
                if job.status in {IndexingJobStatus.QUEUED, IndexingJobStatus.RUNNING}
            ),
            None,
        )
        if active is not None:
            raise ValueError(f"source migration cannot run while indexing job {active.id} is active")

        validated: list[tuple[RepositoryRecord, str, str | None]] = []
        already_migrated: list[RepositoryRecord] = []
        for source in sources:
            existing = self._catalog.get_repository(source.repository_id)
            if existing is None:
                raise ValueError(f"unknown repository id: {source.repository_id}")
            canonical_url = validate_gitlab_url(source.url, self._allowed_hosts)
            canonical_ref = validate_git_ref(source.ref)
            if existing.provider is RepositoryProvider.GITLAB:
                if (
                    existing.canonical_url != canonical_url
                    or existing.canonical_ref != canonical_ref
                ):
                    raise ValueError(
                        f"repository {source.repository_id} is registered to another GitLab source"
                    )
                already_migrated.append(existing)
                continue
            validated.append((existing, canonical_url, canonical_ref))

        prepared: list[RepositoryRecord] = []
        for existing, canonical_url, canonical_ref in validated:
            checkout = self._checkout.checkout(existing.id, canonical_url, canonical_ref)
            prepared.append(
                RepositoryRecord.model_validate(
                    existing.model_copy(
                        update={
                            "provider": RepositoryProvider.GITLAB,
                            "local_path": str(checkout.path),
                            "canonical_url": canonical_url,
                            "canonical_ref": canonical_ref,
                            "exact_commit": checkout.commit_sha,
                            "status": RepositoryStatus.READY,
                        }
                    ).model_dump(mode="python")
                )
            )

        migrated = self._catalog.migrate_local_sources_to_gitlab(prepared) if prepared else ()
        by_id = {record.id: record for record in (*already_migrated, *migrated)}
        ordered = tuple(by_id[source.repository_id] for source in sources)
        job = self.enqueue_reindex()
        self._record_event(
            job,
            level=JobEventLevel.INFO,
            code="SOURCES_MIGRATED",
            message=(
                "Migrated local repository sources to GitLab and queued one portfolio refresh: "
                + ", ".join(record.id for record in ordered)
                + "."
            ),
        )
        return SourceMigrationResult(ordered, job)

    def process_next_job(self) -> IndexingJob | None:
        """Claim and execute at most one durable portfolio or repository-scoped job."""

        claimed_job = self._jobs.claim_next_job()
        if claimed_job is None:
            return None
        job: IndexingJob = claimed_job
        self._record_event(
            job,
            level=JobEventLevel.INFO,
            code="JOB_STARTED",
            message="The worker claimed this job and started repository materialization.",
        )
        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._heartbeat_job,
            args=(job.id, heartbeat_stop),
            name=f"index-job-heartbeat-{job.id}",
            daemon=True,
        )
        heartbeat.start()
        stage = IndexingStage.MATERIALIZING
        original_records: tuple[RepositoryRecord, ...] = ()
        scoped_original_records: tuple[RepositoryRecord, ...] = ()
        try:
            self._raise_if_cancelled(job.id)
            original_records = self._catalog.list_repositories(include_disabled=False)
            scoped_original_records = _records_in_job_scope(job, original_records)
            reusable_repository_ids, published_repository_ids = self._repository_reuse_state(
                original_records
            )
            context_original_records = _records_in_index_context(
                job,
                original_records,
                reusable_repository_ids,
                published_repository_ids,
            )
            self._record_event(
                job,
                level=JobEventLevel.INFO,
                code="MATERIALIZATION_STARTED",
                message=(
                    f"Materializing {len(scoped_original_records)} "
                    + ("requested repository." if job.requested_repository_id else "enabled repositories.")
                ),
            )
            scoped_materialized_values: list[RepositoryRecord] = []
            for record in scoped_original_records:
                self._raise_if_cancelled(job.id)
                scoped_materialized_values.append(self._materialize(record))
            scoped_materialized = tuple(scoped_materialized_values)
            scoped_materialized_by_id = {record.id: record for record in scoped_materialized}
            materialized = tuple(
                scoped_materialized_by_id.get(record.id, record)
                for record in context_original_records
            )
            self._raise_if_cancelled(job.id)
            if not materialized:
                raise RuntimeError("no enabled repositories are registered")
            original_by_id = {record.id: record for record in original_records}
            changed_repository_ids = frozenset(
                record.id
                for record in scoped_materialized
                if record.id not in reusable_repository_ids
                or original_by_id[record.id].exact_commit != record.exact_commit
            )
            if not changed_repository_ids:
                return self._complete_no_changes(
                    job,
                    original_records,
                    scoped_materialized,
                    message=(
                        "All resolved Git commits match the published graph and knowledge manifest; "
                        "no repositories were reindexed."
                    ),
                )

            repository_change_sets: dict[str, RepositoryChangeSet] = {}
            change_resolver = getattr(self._checkout, "changes_since", None)
            if callable(change_resolver):
                for record in scoped_materialized:
                    original = original_by_id[record.id]
                    if (
                        record.id not in changed_repository_ids
                        or record.id not in reusable_repository_ids
                        or record.provider is not RepositoryProvider.GITLAB
                        or record.exact_commit is None
                        or original.exact_commit is None
                    ):
                        continue
                    try:
                        repository_change_sets[record.id] = change_resolver(
                            record.id,
                            GitCheckout(Path(record.local_path), record.exact_commit),
                            original.exact_commit,
                        )
                    except Exception as error:
                        _LOGGER.warning(
                            "Exact incremental change resolution failed; repository will be rebuilt",
                            exc_info=True,
                            extra={"fields": {"repository_id": record.id}},
                        )
                        self._record_event(
                            job,
                            level=JobEventLevel.WARNING,
                            code="INCREMENTAL_FALLBACK",
                            message=(
                                f"Could not prove an exact Git change set for {record.id}; "
                                "using a safe full repository rebuild. "
                                f"Reason: {type(error).__name__}."
                            ),
                        )

            tree_unchanged_ids = {
                repository_id
                for repository_id, change_set in repository_change_sets.items()
                if change_set.is_unchanged
            }
            changed_repository_ids = changed_repository_ids.difference(tree_unchanged_ids)
            for repository_id in tree_unchanged_ids:
                repository_change_sets.pop(repository_id, None)
            if not changed_repository_ids:
                return self._complete_no_changes(
                    job,
                    original_records,
                    scoped_materialized,
                    message=(
                        "Resolved Git commits changed, but their tracked source trees are identical; "
                        "the published graph, knowledge, and embeddings were reused."
                    ),
                )

            scoped_ids = frozenset(record.id for record in scoped_materialized)
            indexing_values: list[RepositoryRecord] = []
            for record in materialized:
                if record.id not in scoped_ids:
                    indexing_values.append(record)
                    continue
                indexing_values.append(
                    self._catalog.upsert_repository(
                        _updated_repository(record, status=RepositoryStatus.INDEXING)
                        if record.id in changed_repository_ids
                        else _preserve_index_metadata(original_by_id[record.id], record)
                    )
                )
            indexing_records = tuple(indexing_values)
            stage = IndexingStage.SCANNING
            job = self._jobs.update_job(
                job.id,
                status=IndexingJobStatus.RUNNING,
                stage=stage,
            )
            self._record_event(
                job,
                level=JobEventLevel.INFO,
                code="SCAN_STARTED",
                message=(
                    "Repository materialization completed; indexing changed repositories "
                    + ", ".join(sorted(changed_repository_ids))
                    + "."
                ),
            )

            def begin_publication() -> None:
                nonlocal job, stage
                self._raise_if_cancelled(job.id)
                self._record_event(
                    job,
                    level=JobEventLevel.INFO,
                    code="SCAN_COMPLETED",
                    message="Graph and knowledge generation completed successfully.",
                )
                try:
                    job = self._jobs.update_job(
                        job.id,
                        status=IndexingJobStatus.RUNNING,
                        stage=IndexingStage.PUBLISHING,
                    )
                except ValueError:
                    self._raise_if_cancelled(job.id)
                    raise
                stage = IndexingStage.PUBLISHING
                self._record_event(
                    job,
                    level=JobEventLevel.INFO,
                    code="PUBLISHING_STARTED",
                    message="Publishing graph, repository metadata, and latest knowledge manifest.",
                )

            artifacts = self._indexing.run(
                repositories=tuple(record.to_repository_definition() for record in indexing_records),
                changed_repository_ids=changed_repository_ids,
                repository_change_sets=repository_change_sets or None,
                persist_neo4j=True,
                index_embeddings=self._index_embeddings,
                persist_metadata=True,
                require_incremental_baseline=job.requested_repository_id is not None,
                cancellation_check=lambda: self._raise_if_cancelled(job.id),
                begin_publication=begin_publication,
            )
            job = self._jobs.update_job(
                job.id,
                status=IndexingJobStatus.RUNNING,
                stage=stage,
                scan_run_id=artifacts.result.scan_run_id,
            )
            indexed = self._record_success(indexing_records, artifacts)
            self._manifest_publisher.publish_manifest(_manifest(indexed, artifacts))
            completed = self._jobs.update_job(
                job.id,
                status=IndexingJobStatus.SUCCEEDED,
                stage=IndexingStage.COMPLETE,
                scan_run_id=artifacts.result.scan_run_id,
            )
            self._record_event(
                completed,
                level=JobEventLevel.INFO,
                code="JOB_SUCCEEDED",
                message="The portfolio graph and knowledge bases were published successfully.",
            )
            return completed
        except JobCancellationRequested:
            return self._complete_cancelled_job(job.id, stage, scoped_original_records)
        except Exception as error:
            current = self._jobs.get_job(job.id)
            if current is not None and current.cancellation_requested_at is not None:
                return self._complete_cancelled_job(job.id, stage, scoped_original_records)
            _LOGGER.exception(
                "Repository portfolio indexing failed",
                extra={"fields": {"job_id": job.id, "stage": stage.value}},
            )
            self._restore_after_failure(scoped_original_records, job.requested_repository_id)
            failed = self._jobs.update_job(
                job.id,
                status=IndexingJobStatus.FAILED,
                stage=stage,
                error_code=f"{stage.value.upper()}_FAILED",
            )
            self._record_event(
                failed,
                level=JobEventLevel.ERROR,
                code=f"{stage.value.upper()}_FAILED",
                message=_safe_failure_message(error),
                error_type=type(error).__name__,
            )
            return failed
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)

    def _raise_if_cancelled(self, job_id: str) -> None:
        current = self._jobs.get_job(job_id)
        if current is not None and (
            current.status is IndexingJobStatus.CANCELLED
            or current.cancellation_requested_at is not None
        ):
            raise JobCancellationRequested(job_id)

    def _complete_cancelled_job(
        self,
        job_id: str,
        stage: IndexingStage,
        originals: tuple[RepositoryRecord, ...],
    ) -> IndexingJob:
        self._restore_after_failure(
            originals,
            None,
            mark_registered_failed=False,
        )
        current = self._jobs.get_job(job_id)
        if current is not None and current.status is IndexingJobStatus.CANCELLED:
            return current
        cancelled = self._jobs.update_job(
            job_id,
            status=IndexingJobStatus.CANCELLED,
            stage=stage,
        )
        self._record_event(
            cancelled,
            level=JobEventLevel.WARNING,
            code="JOB_CANCELLED",
            message=(
                "The worker stopped at a safe boundary. Previously published graph and knowledge "
                "remain active."
            ),
        )
        return cancelled

    def _materialize(self, record: RepositoryRecord) -> RepositoryRecord:
        if record.provider is RepositoryProvider.LOCAL:
            path = Path(record.local_path).resolve(strict=True)
            if not path.is_dir():
                raise RuntimeError(f"registered local repository is not a directory: {record.id}")
            return _updated_repository(record, local_path=str(path), status=RepositoryStatus.READY)
        assert record.canonical_url is not None
        checkout = self._checkout.checkout(record.id, record.canonical_url, record.canonical_ref)
        return self._catalog.upsert_repository(
            _updated_repository(
                record,
                local_path=str(checkout.path),
                exact_commit=checkout.commit_sha,
                status=RepositoryStatus.READY,
            )
        )

    def _record_success(
        self,
        records: tuple[RepositoryRecord, ...],
        artifacts: IndexingArtifacts,
    ) -> tuple[RepositoryRecord, ...]:
        repository_uris = dict(
            zip(artifacts.result.repository_ids, artifacts.result.knowledge_uris, strict=False)
        )
        if len(artifacts.result.knowledge_uris) != len(artifacts.result.repository_ids) + 1:
            raise RuntimeError("indexing result did not contain one knowledge URI per repository")
        indexed_at = datetime.now(UTC)
        changed_repository_ids = frozenset(artifacts.changed_repository_ids)
        values: list[RepositoryRecord] = []
        for record in records:
            node_count = sum(1 for node in artifacts.graph.nodes if node.repository_id == record.id)
            relationship_count = sum(
                1 for relationship in artifacts.graph.relationships if relationship.repository_id == record.id
            )
            values.append(
                _updated_repository(
                    record,
                    status=(
                        RepositoryStatus.INDEXED
                        if record.id in changed_repository_ids
                        else record.status
                    ),
                    knowledge_base_uri=repository_uris.get(record.id),
                    graph_node_count=node_count,
                    graph_relationship_count=relationship_count,
                    last_scan_run_id=artifacts.result.scan_run_id,
                    indexed_at=indexed_at if record.id in changed_repository_ids else record.indexed_at,
                )
            )
        return self._catalog.publish_index_metadata(values)

    def _repository_reuse_state(
        self,
        records: tuple[RepositoryRecord, ...],
    ) -> tuple[frozenset[str], frozenset[str]]:
        state_resolver = getattr(self._indexing, "repository_reuse_state", None)
        if callable(state_resolver):
            reusable, published = state_resolver(records)
            return (
                frozenset(str(value) for value in reusable),
                frozenset(str(value) for value in published),
            )
        resolver = getattr(self._indexing, "reusable_repository_ids", None)
        reusable = (
            frozenset(str(value) for value in resolver(records))
            if callable(resolver)
            else frozenset()
        )
        published = frozenset(record.id for record in records if _has_published_metadata(record))
        return reusable, published

    def _complete_no_changes(
        self,
        job: IndexingJob,
        originals: tuple[RepositoryRecord, ...],
        materialized: tuple[RepositoryRecord, ...],
        *,
        message: str,
    ) -> IndexingJob:
        self._raise_if_cancelled(job.id)
        original_by_id = {record.id: record for record in originals}
        for record in materialized:
            self._catalog.upsert_repository(
                _preserve_index_metadata(original_by_id[record.id], record)
            )
        materialized_ids = {record.id for record in materialized}
        prior_scan_ids = {
            record.last_scan_run_id
            for record in originals
            if record.id in materialized_ids and record.last_scan_run_id is not None
        }
        completed = self._jobs.update_job(
            job.id,
            status=IndexingJobStatus.SUCCEEDED,
            stage=IndexingStage.COMPLETE,
            scan_run_id=next(iter(prior_scan_ids)) if len(prior_scan_ids) == 1 else None,
        )
        self._record_event(
            completed,
            level=JobEventLevel.INFO,
            code="NO_CHANGES",
            message=message,
        )
        return completed

    def _restore_after_failure(
        self,
        originals: tuple[RepositoryRecord, ...],
        requested_repository_id: str | None,
        *,
        mark_registered_failed: bool = True,
    ) -> None:
        for record in originals:
            target = record
            if (
                record.status is RepositoryStatus.REGISTERED
                and mark_registered_failed
                and (requested_repository_id is None or record.id == requested_repository_id)
            ):
                target = _updated_repository(record, status=RepositoryStatus.FAILED)
            try:
                self._catalog.upsert_repository(target)
            except Exception:
                _LOGGER.exception(
                    "Failed to restore repository state after indexing failure",
                    extra={"fields": {"repository_id": record.id}},
                )

    def _existing_active_job(self, repository_id: str | None) -> IndexingJob | None:
        return next(
            (
                job
                for job in self._jobs.list_jobs()
                if job.requested_repository_id == repository_id
                and job.status in {IndexingJobStatus.QUEUED, IndexingJobStatus.RUNNING}
            ),
            None,
        )

    def _existing_queued_portfolio_job(self) -> IndexingJob | None:
        """Return a not-yet-claimed portfolio job that will observe new registrations."""

        return next(
            (
                job
                for job in self._jobs.list_jobs()
                if job.status is IndexingJobStatus.QUEUED
            ),
            None,
        )

    def _register_and_ensure_queued_job(
        self,
        records: tuple[RepositoryRecord, ...],
        *,
        requested_repository_id: str | None = None,
    ) -> tuple[tuple[RepositoryRecord, ...], IndexingJob, bool]:
        combined = getattr(
            self._catalog,
            "register_repositories_and_ensure_queued_job",
            None,
        )
        if cast(object, self._catalog) is cast(object, self._jobs) and callable(combined):
            result = combined(
                records,
                requested_repository_id=requested_repository_id,
            )
            return cast(tuple[tuple[RepositoryRecord, ...], IndexingJob, bool], result)
        stored = self._catalog.upsert_repositories(records)
        existing = self._existing_queued_portfolio_job()
        if existing is not None:
            return stored, existing, False
        return (
            stored,
            self._jobs.enqueue_job(IndexingJob.create(requested_repository_id)),
            True,
        )

    def _prepare_gitlab_registrations(
        self,
        registrations: tuple[GitLabRepositoryRegistration, ...],
    ) -> tuple[RepositoryRecord, ...]:
        if not registrations:
            raise ValueError("at least one GitLab repository is required")
        if len(registrations) > _MAX_BULK_REPOSITORIES:
            raise ValueError(
                f"at most {_MAX_BULK_REPOSITORIES} GitLab repositories may be registered at once"
            )

        existing = self._catalog.list_repositories()
        existing_by_id = {record.id: record for record in existing}
        existing_by_url = {
            record.canonical_url: record
            for record in existing
            if record.canonical_url is not None
        }
        canonical: list[tuple[GitLabRepositoryRegistration, str, str | None]] = []
        seen_urls: dict[str, int] = {}
        for position, registration in enumerate(registrations, start=1):
            canonical_url = validate_gitlab_url(registration.url, self._allowed_hosts)
            prior_position = seen_urls.get(canonical_url)
            if prior_position is not None:
                raise ValueError(
                    f"GitLab URL at position {position} duplicates position {prior_position}"
                )
            seen_urls[canonical_url] = position
            canonical.append(
                (registration, canonical_url, validate_git_ref(registration.ref))
            )

        prepared: list[RepositoryRecord] = []
        seen_ids: dict[str, int] = {}
        for position, (registration, canonical_url, canonical_ref) in enumerate(
            canonical,
            start=1,
        ):
            existing_url_record = existing_by_url.get(canonical_url)
            requested_id = registration.repository_id or _repository_id_from_url(canonical_url)
            if existing_url_record is not None and existing_url_record.id != requested_id:
                if registration.repository_id is not None:
                    raise ValueError(
                        f"GitLab URL at position {position} is already registered as repository "
                        f"{existing_url_record.id}"
                    )
                requested_id = existing_url_record.id

            prior_position = seen_ids.get(requested_id)
            if prior_position is not None:
                raise ValueError(
                    f"repository ID {requested_id} at position {position} duplicates position "
                    f"{prior_position}"
                )
            seen_ids[requested_id] = position

            existing_id_record = existing_by_id.get(requested_id)
            if existing_id_record is not None and existing_id_record.canonical_url != canonical_url:
                raise ValueError(
                    f"repository ID {requested_id} is already registered to another source"
                )

            local_path = (
                Path(existing_url_record.local_path)
                if existing_url_record is not None
                else (self._checkout_root / requested_id).resolve()
            )
            if existing_url_record is None and self._checkout_root not in local_path.parents:
                raise ValueError("managed repository path escaped the configured checkout root")

            if existing_url_record is None:
                prepared.append(
                    RepositoryRecord(
                        id=requested_id,
                        provider=RepositoryProvider.GITLAB,
                        name=registration.name or _repository_name_from_url(canonical_url),
                        description=registration.description,
                        local_path=str(local_path),
                        canonical_url=canonical_url,
                        canonical_ref=canonical_ref,
                        status=RepositoryStatus.REGISTERED,
                    )
                )
                continue
            prepared.append(
                _updated_repository(
                    existing_url_record,
                    name=registration.name or existing_url_record.name,
                    description=(
                        registration.description
                        if registration.description is not None
                        else existing_url_record.description
                    ),
                    canonical_ref=canonical_ref,
                )
            )
        return tuple(prepared)

    def _heartbeat_job(self, job_id: str, stop: Event) -> None:
        while not stop.wait(self._heartbeat_seconds):
            try:
                current = self._jobs.get_job(job_id)
                if current is None or current.status is not IndexingJobStatus.RUNNING:
                    return
                self._jobs.update_job(
                    current.id,
                    status=IndexingJobStatus.RUNNING,
                    stage=current.stage,
                    scan_run_id=current.scan_run_id,
                )
            except (KeyError, ValueError):
                return
            except Exception as error:
                _LOGGER.warning(
                    "Unable to renew indexing job lease",
                    extra={
                        "fields": {
                            "job_id": job_id,
                            "error_type": type(error).__name__,
                        }
                    },
                )

    def _record_event(
        self,
        job: IndexingJob,
        *,
        level: JobEventLevel,
        code: str,
        message: str,
        error_type: str | None = None,
    ) -> None:
        try:
            self._events.append_job_event(
                IndexingJobEvent.create(
                    job.id,
                    level=level,
                    stage=job.stage,
                    code=code,
                    message=message,
                    error_type=error_type,
                )
            )
        except Exception as error:
            _LOGGER.warning(
                "Unable to persist indexing job event",
                extra={
                    "fields": {
                        "job_id": job.id,
                        "event_code": code,
                        "error_type": type(error).__name__,
                    }
                },
            )


class IndexingWorker:
    """Polling worker for durable onboarding and reindex jobs."""

    def __init__(self, onboarding: RepositoryOnboardingService, poll_seconds: float) -> None:
        self._onboarding = onboarding
        self._poll_seconds = poll_seconds

    def run_forever(self, stop: Event | None = None) -> None:
        """Process jobs until the supplied stop event is set."""

        stop_event = stop or Event()
        while not stop_event.is_set():
            started = monotonic()
            processed = self._onboarding.process_next_job()
            if processed is None:
                stop_event.wait(self._poll_seconds)
                continue
            _LOGGER.info(
                "Indexing job completed",
                extra={
                    "fields": {
                        "job_id": processed.id,
                        "status": processed.status.value,
                        "elapsed_seconds": round(monotonic() - started, 3),
                    }
                },
            )


def _batch_event_message(
    action: str,
    records: tuple[RepositoryRecord, ...],
) -> str:
    visible = tuple(record.id for record in records[:12])
    remainder = len(records) - len(visible)
    suffix = f" (+{remainder} more)" if remainder else ""
    return f"{action} for {len(records)} repositories: {', '.join(visible)}{suffix}."


def _safe_failure_message(error: Exception) -> str:
    value = " ".join(redact_sensitive_text(str(error)).split())
    if not value:
        return "The worker raised an error without a diagnostic message."
    return value[:2_000]


def _records_in_job_scope(
    job: IndexingJob,
    records: tuple[RepositoryRecord, ...],
) -> tuple[RepositoryRecord, ...]:
    """Select only the requested repository while retaining portfolio context elsewhere."""

    if job.requested_repository_id is None:
        return records
    selected = tuple(record for record in records if record.id == job.requested_repository_id)
    if not selected:
        raise RuntimeError(
            f"requested repository is not enabled: {job.requested_repository_id}"
        )
    return selected


def _records_in_index_context(
    job: IndexingJob,
    records: tuple[RepositoryRecord, ...],
    reusable_repository_ids: frozenset[str],
    published_repository_ids: frozenset[str],
) -> tuple[RepositoryRecord, ...]:
    """Keep only the scoped target and repositories proven safe to reuse."""

    if job.requested_repository_id is None:
        return records
    target_id = job.requested_repository_id
    record_by_id = {record.id: record for record in records}
    unsafe_ids = {
        record.id
        for record in records
        if record.id != target_id
        and record.id not in reusable_repository_ids
        and (record.id in published_repository_ids or _has_published_metadata(record))
    }
    unsafe_ids.update(
        repository_id
        for repository_id in published_repository_ids
        if repository_id != target_id
        and (
            repository_id not in record_by_id
            or repository_id not in reusable_repository_ids
        )
    )
    if unsafe_ids:
        raise RuntimeError(
            "repository-scoped indexing cannot safely reuse published repositories: "
            + ", ".join(sorted(unsafe_ids))
            + "; run a full portfolio reindex first"
        )
    context_ids = reusable_repository_ids | {target_id}
    return tuple(record for record in records if record.id in context_ids)


def _has_published_metadata(record: RepositoryRecord) -> bool:
    return (
        record.status is RepositoryStatus.INDEXED
        or record.knowledge_base_uri is not None
        or record.last_scan_run_id is not None
        or record.indexed_at is not None
        or record.graph_node_count > 0
        or record.graph_relationship_count > 0
    )


def _preserve_index_metadata(
    published: RepositoryRecord,
    materialized: RepositoryRecord,
) -> RepositoryRecord:
    """Keep a validated publication while accepting its refreshed local checkout path."""

    return _updated_repository(
        materialized,
        status=RepositoryStatus.INDEXED,
        knowledge_base_uri=published.knowledge_base_uri,
        graph_node_count=published.graph_node_count,
        graph_relationship_count=published.graph_relationship_count,
        last_scan_run_id=published.last_scan_run_id,
        indexed_at=published.indexed_at,
    )


def _updated_repository(record: RepositoryRecord, **changes: object) -> RepositoryRecord:
    values = record.model_dump(mode="python")
    values.update(changes)
    values["updated_at"] = datetime.now(UTC)
    return RepositoryRecord.model_validate(values)


def _repository_id_from_url(canonical_url: str) -> str:
    raw = Path(urlsplit(canonical_url).path).stem.lower()
    value = _SLUG_CHARACTER.sub("-", raw).strip("-")[:63].rstrip("-")
    if not value or not value[0].isalnum():
        raise ValueError("GitLab repository name cannot produce a safe repository id")
    return value


def _repository_name_from_url(canonical_url: str) -> str:
    name = Path(urlsplit(canonical_url).path).stem.replace("-", " ").replace("_", " ").strip()
    return name or "GitLab Repository"


def _manifest(
    repositories: tuple[RepositoryRecord, ...],
    artifacts: IndexingArtifacts,
) -> Mapping[str, Any]:
    central_uri = artifacts.result.knowledge_uris[-1]
    human_uris = tuple(item.uri for item in artifacts.human_stored_documents)
    if human_uris and len(human_uris) != len(repositories) + 1:
        raise RuntimeError("indexing result did not contain a complete reader-edition portfolio")
    human_by_repository = dict(zip((record.id for record in repositories), human_uris, strict=False))
    manifest: dict[str, Any] = {
        "schemaVersion": 2 if human_uris else 1,
        "scanRunId": artifacts.result.scan_run_id,
        "generatedAt": artifacts.graph.generated_at.isoformat(),
        "repositories": [
            {
                "id": record.id,
                "commitSha": record.exact_commit,
                "knowledgeUri": record.knowledge_base_uri,
                **({"humanKnowledgeUri": human_by_repository[record.id]} if human_uris else {}),
            }
            for record in repositories
        ],
        "centralKnowledgeUri": central_uri,
        "graph": {
            "neo4jSnapshotId": artifacts.result.neo4j_snapshot_id,
            "statistics": artifacts.result.statistics.model_dump(mode="json"),
        },
    }
    if human_uris:
        manifest["humanCentralKnowledgeUri"] = human_uris[-1]
    return manifest
