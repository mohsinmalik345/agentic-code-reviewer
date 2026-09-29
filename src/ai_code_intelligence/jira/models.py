from __future__ import annotations

import ipaddress
import re
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, StrictBool, field_validator, model_validator

from ai_code_intelligence.domain.models import FrozenModel

_SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$"
_REPOSITORY_ID = r"^[a-z0-9][a-z0-9-]{0,62}$"
_PROJECT_KEY = r"^[A-Z][A-Z0-9_]{0,63}$"
_CREDENTIAL_ENV = r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"
_FIELD_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DNS_HOST = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_BASE_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._~-]+$")
_ISSUE_KEY = r"^[A-Z][A-Z0-9_]{0,63}-[0-9]{1,18}$"


def _now() -> datetime:
    return datetime.now(UTC)


class JiraEdition(StrEnum):
    """Supported Jira deployment editions and their REST API families."""

    CLOUD = "cloud"
    DATA_CENTER = "data_center"


class JiraAuthType(StrEnum):
    """Credential mechanisms supported for each Jira edition."""

    API_TOKEN = "api_token"
    PERSONAL_ACCESS_TOKEN = "personal_access_token"


class JiraConnection(FrozenModel):
    """Credential-free configuration for one allowlisted Jira installation.

    ``credential_env`` is an indirection key only. Secret material is deliberately
    absent from this model so it cannot be serialized by configuration stores or APIs.
    """

    id: str = Field(pattern=_SAFE_ID)
    name: str = Field(min_length=1, max_length=200)
    edition: JiraEdition
    base_url: str = Field(min_length=1, max_length=2_048)
    auth_type: JiraAuthType
    username: str | None = Field(default=None, max_length=320)
    credential_env: str = Field(pattern=_CREDENTIAL_ENV)
    enabled: StrictBool = True
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("name")
    @classmethod
    def normalized_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or _has_control_character(normalized):
            raise ValueError("Jira connection name must be non-empty and contain no controls")
        return normalized

    @field_validator("base_url", mode="before")
    @classmethod
    def safe_base_url(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Jira base URL must be a string")
        return normalize_jira_base_url(value)

    @field_validator("username")
    @classmethod
    def normalized_username(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            not normalized
            or len(normalized) > 320
            or _has_control_character(normalized)
            or any(character.isspace() for character in normalized)
            or ":" in normalized
        ):
            raise ValueError("Jira username must be a bounded credential-free account name")
        return normalized

    @field_validator("created_at", "updated_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Jira connection timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def valid_edition_authentication(self) -> JiraConnection:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.edition is JiraEdition.CLOUD:
            if self.auth_type is not JiraAuthType.API_TOKEN:
                raise ValueError("Jira Cloud requires API-token authentication")
            if self.username is None or not _looks_like_email(self.username):
                raise ValueError("Jira Cloud requires an email username")
        else:
            if self.auth_type is not JiraAuthType.PERSONAL_ACCESS_TOKEN:
                raise ValueError("Jira Data Center requires personal-access-token authentication")
            if self.username is not None:
                raise ValueError("Jira Data Center bearer authentication must not store a username")
        return self


class JiraProjectMapping(FrozenModel):
    """Repository-to-Jira-project mapping and issue-discovery policy."""

    id: str = Field(pattern=_SAFE_ID)
    connection_id: str = Field(pattern=_SAFE_ID)
    repository_id: str = Field(pattern=_REPOSITORY_ID)
    jira_project_key: str = Field(pattern=_PROJECT_KEY)
    acceptance_criteria_fields: tuple[str, ...] = Field(default=(), max_length=32)
    issue_key_pattern: str = Field(min_length=1, max_length=1_000)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("jira_project_key")
    @classmethod
    def canonical_project_key(cls, value: str) -> str:
        if value != value.upper():
            raise ValueError("Jira project key must be uppercase")
        return value

    @field_validator("acceptance_criteria_fields")
    @classmethod
    def safe_acceptance_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 32:
            raise ValueError("at most 32 Jira acceptance-criteria fields may be configured")
        if len(value) != len(set(value)):
            raise ValueError("Jira acceptance-criteria fields must be unique")
        if any(not _FIELD_ID.fullmatch(field_id) for field_id in value):
            raise ValueError("Jira acceptance-criteria field IDs are invalid")
        return value

    @field_validator("issue_key_pattern")
    @classmethod
    def valid_issue_pattern(cls, value: str) -> str:
        if value != value.strip() or _has_control_character(value):
            raise ValueError("Jira issue-key pattern must not contain surrounding space or controls")
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError("Jira issue-key pattern is not valid") from error
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Jira mapping timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def valid_timestamps(self) -> JiraProjectMapping:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self


class JiraConnectionTestResult(FrozenModel):
    """Sanitized result of Jira server and authenticated-user probes."""

    ok: StrictBool
    message: str = Field(min_length=1, max_length=500)
    server_title: str | None = Field(default=None, max_length=500)
    server_version: str | None = Field(default=None, max_length=100)
    authenticated_user: str | None = Field(default=None, max_length=500)

    @field_validator("message", "server_title", "server_version", "authenticated_user")
    @classmethod
    def safe_display_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized or _has_control_character(normalized):
            raise ValueError("Jira connection-test text must be non-empty and contain no controls")
        return normalized


class JiraIssue(FrozenModel):
    """Normalized, bounded Jira issue content independent of REST API edition."""

    key: str = Field(pattern=_ISSUE_KEY, max_length=84)
    project_key: str = Field(pattern=_PROJECT_KEY)
    summary: str = Field(min_length=1, max_length=10_000)
    description: str | None = Field(default=None, max_length=1_000_000)
    acceptance_criteria: tuple[str, ...] = Field(default=(), max_length=32)
    status: str | None = Field(default=None, max_length=500)
    issue_type: str | None = Field(default=None, max_length=500)
    assignee: str | None = Field(default=None, max_length=500)
    reporter: str | None = Field(default=None, max_length=500)
    labels: tuple[str, ...] = Field(default=(), max_length=1_000)
    url: str = Field(min_length=1, max_length=2_048)

    @field_validator("summary", "description", "status", "issue_type", "assignee", "reporter")
    @classmethod
    def normalized_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or _has_unsafe_text_control(normalized):
            raise ValueError("normalized Jira text must be non-empty and contain no unsafe controls")
        return normalized

    @field_validator("acceptance_criteria")
    @classmethod
    def normalized_acceptance_criteria(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or _has_unsafe_text_control(item) for item in normalized):
            raise ValueError("Jira acceptance criteria must contain safe non-empty text")
        return normalized

    @field_validator("labels")
    @classmethod
    def normalized_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Jira labels must be unique")
        if any(
            not label or label != label.strip() or len(label) > 255 or _has_control_character(label)
            for label in value
        ):
            raise ValueError("Jira labels must be bounded safe strings")
        return value

    @model_validator(mode="after")
    def key_matches_project(self) -> JiraIssue:
        if not self.key.startswith(f"{self.project_key}-"):
            raise ValueError("Jira issue key must belong to project_key")
        return self


def normalize_jira_base_url(value: str) -> str:
    """Validate and canonicalize a credential-free Jira HTTPS site root."""

    if not value or value != value.strip() or len(value) > 2_048:
        raise ValueError("Jira base URL must be a non-empty bounded string")
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Jira base URL must not contain whitespace or controls")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Jira base URL is malformed") from error
    if parsed.scheme.lower() != "https":
        raise ValueError("Jira base URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Jira base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Jira base URL must not contain a query or fragment")
    if port not in (None, 443):
        raise ValueError("Jira base URL must use the default HTTPS port")
    hostname = parsed.hostname
    if hostname is None or hostname.endswith(".") or "%" in parsed.netloc:
        raise ValueError("Jira base URL host is invalid")
    hostname = hostname.lower()
    if not _DNS_HOST.fullmatch(hostname):
        raise ValueError("Jira base URL must use a valid DNS hostname")
    if hostname.replace(".", "").isdigit():
        raise ValueError("Jira base URL must not use a numeric address")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("Jira base URL must not use an IP-literal host")

    path = parsed.path
    if path in {"", "/"}:
        canonical_path = ""
    else:
        if "%" in path or "\\" in path or "//" in path:
            raise ValueError("Jira base URL path contains unsafe encoding or separators")
        segments = path.strip("/").split("/")
        if any(
            segment in {"", ".", ".."} or not _BASE_PATH_SEGMENT.fullmatch(segment) for segment in segments
        ):
            raise ValueError("Jira base URL path is unsafe")
        canonical_path = f"/{'/'.join(segments)}"
    return urlunsplit(("https", hostname, canonical_path, "", ""))


def _looks_like_email(value: str) -> bool:
    local, separator, domain = value.rpartition("@")
    return bool(separator and local and domain and "." in domain and not domain.startswith("."))


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _has_unsafe_text_control(value: str) -> bool:
    return any(
        (ord(character) < 32 and character not in {"\n", "\r", "\t"}) or ord(character) == 127
        for character in value
    )
