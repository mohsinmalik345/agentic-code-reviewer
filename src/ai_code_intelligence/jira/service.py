from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Protocol
from urllib.parse import urlsplit

from ai_code_intelligence.jira.models import (
    JiraAuthType,
    JiraConnection,
    JiraConnectionTestResult,
    JiraEdition,
    JiraIssue,
    JiraProjectMapping,
)
from ai_code_intelligence.jira.store import JiraConfigurationStore
from ai_code_intelligence.persistence.catalog import RepositoryCatalog
from ai_code_intelligence.utils.ids import stable_id


class JiraReader(Protocol):
    """Read-only Jira operations consumed by the application layer."""

    def test_connection(self) -> JiraConnectionTestResult: ...

    def fetch_issue(
        self,
        issue_key: str,
        acceptance_criteria_fields: tuple[str, ...] = (),
    ) -> JiraIssue: ...


class JiraIntegrationService:
    """Coordinates secure Jira configuration, mappings, and read-only probes."""

    def __init__(
        self,
        store: JiraConfigurationStore,
        repositories: RepositoryCatalog,
        reader_factory: Callable[[JiraConnection], JiraReader],
        allowed_hosts: tuple[str, ...],
    ) -> None:
        self._store = store
        self._repositories = repositories
        self._reader_factory = reader_factory
        self._allowed_hosts = frozenset(allowed_hosts)

    def initialize(self) -> None:
        """Initialize and validate the selected durable configuration store."""

        self._store.initialize()

    def list_connections(self, *, include_disabled: bool = False) -> tuple[JiraConnection, ...]:
        return self._store.list_connections(include_disabled=include_disabled)

    def save_connection(self, values: Mapping[str, object]) -> JiraConnection:
        """Validate and persist credential-free Jira connection metadata."""

        connection = JiraConnection(
            id=_required_string(values, "id"),
            name=_required_string(values, "name"),
            edition=JiraEdition(_required_string(values, "edition")),
            base_url=_required_string(values, "baseUrl"),
            auth_type=JiraAuthType(_required_string(values, "authType")),
            username=_optional_string(values, "username"),
            credential_env=_required_string(values, "credentialEnv"),
            enabled=_optional_bool(values, "enabled", default=True),
        )
        self._validate_allowed_host(connection)
        return self._store.upsert_connection(connection)

    def delete_connection(self, connection_id: str) -> bool:
        return self._store.delete_connection(connection_id)

    def credential_configured(self, connection: JiraConnection) -> bool:
        """Report credential presence without reading or returning its value."""

        value = os.getenv(connection.credential_env)
        return value is not None and bool(value.strip())

    def test_connection(self, connection_id: str) -> JiraConnectionTestResult:
        connection = self._required_connection(connection_id)
        if not connection.enabled:
            raise ValueError("Jira connection is disabled")
        self._validate_allowed_host(connection)
        return self._reader_factory(connection).test_connection()

    def list_mappings(
        self,
        *,
        repository_id: str | None = None,
    ) -> tuple[JiraProjectMapping, ...]:
        return self._store.list_mappings(repository_id=repository_id)

    def save_mapping(self, values: Mapping[str, object]) -> JiraProjectMapping:
        """Validate repository ownership and persist a project mapping."""

        connection_id = _required_string(values, "connectionId")
        repository_id = _required_string(values, "repositoryId")
        self._required_connection(connection_id)
        if self._repositories.get_repository(repository_id) is None:
            raise ValueError("repository was not found")
        requested_id = _optional_string(values, "id")
        fields = _string_tuple(values, "acceptanceCriteriaFields")
        mapping = JiraProjectMapping(
            id=requested_id or stable_id("jira-mapping", connection_id, repository_id),
            connection_id=connection_id,
            repository_id=repository_id,
            jira_project_key=_required_string(values, "jiraProjectKey"),
            acceptance_criteria_fields=fields,
            issue_key_pattern=_required_string(values, "issueKeyPattern"),
        )
        return self._store.upsert_mapping(mapping)

    def delete_mapping(self, mapping_id: str) -> bool:
        return self._store.delete_mapping(mapping_id)

    def fetch_issue(
        self,
        connection_id: str,
        issue_key: str,
        *,
        repository_id: str | None = None,
    ) -> JiraIssue:
        """Fetch one issue using the mapped acceptance-criteria fields when available."""

        connection = self._required_connection(connection_id)
        if not connection.enabled:
            raise ValueError("Jira connection is disabled")
        fields: tuple[str, ...] = ()
        if repository_id is not None:
            mapping = next(
                (
                    item
                    for item in self._store.list_mappings(
                        connection_id=connection_id,
                        repository_id=repository_id,
                    )
                ),
                None,
            )
            if mapping is None:
                raise ValueError("repository has no Jira project mapping for this connection")
            if not issue_key.upper().startswith(f"{mapping.jira_project_key}-"):
                raise ValueError("Jira issue does not belong to the mapped project")
            fields = mapping.acceptance_criteria_fields
        return self._reader_factory(connection).fetch_issue(issue_key, fields)

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if callable(close):
            close()

    def _required_connection(self, connection_id: str) -> JiraConnection:
        connection = self._store.get_connection(connection_id)
        if connection is None:
            raise ValueError("Jira connection was not found")
        return connection

    def _validate_allowed_host(self, connection: JiraConnection) -> None:
        host = urlsplit(connection.base_url).hostname
        if host is None or host.lower() not in self._allowed_hosts:
            raise ValueError(
                "Jira host is not allowlisted; add it to JIRA_ALLOWED_HOSTS and restart the application"
            )


def _required_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    normalized = value.strip()
    return normalized or None


def _optional_bool(values: Mapping[str, object], key: str, *, default: bool) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string_tuple(values: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = values.get(key, ())
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())
