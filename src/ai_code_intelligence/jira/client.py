from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import os
import re
import ssl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from pydantic import ValidationError

from ai_code_intelligence.jira.models import (
    JiraAuthType,
    JiraConnection,
    JiraConnectionTestResult,
    JiraEdition,
    JiraIssue,
)

_DNS_HOST = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9_]{0,63}-[0-9]{1,18}$")
_FIELD_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DEFAULT_FIELDS = (
    "summary",
    "description",
    "status",
    "issuetype",
    "project",
    "assignee",
    "reporter",
    "labels",
)
_MAX_CREDENTIAL_LENGTH = 16_384
_MAX_ADF_DEPTH = 64
_MAX_ADF_NODES = 50_000


class JiraClientError(RuntimeError):
    """A credential-free Jira failure safe to expose at an application boundary."""


class CredentialProvider(Protocol):
    """Resolves a secret from the non-secret reference stored on a connection."""

    def get_credential(self, reference: str) -> str: ...


class EnvironmentCredentialProvider:
    """Reads Jira credentials lazily from an injected environment mapping."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def get_credential(self, reference: str) -> str:
        """Return the configured secret without retaining it on a Jira model."""

        value = self._environment.get(reference)
        if value is None or not value:
            raise JiraClientError("Jira credential is not configured")
        return value

    def is_configured(self, reference: str) -> bool:
        """Return whether a non-empty value exists for the credential reference."""

        return bool(self._environment.get(reference))


@dataclass(frozen=True, slots=True)
class JiraHttpRequest:
    """One validated outbound Jira request passed to an injectable transport."""

    method: str
    url: str
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class JiraHttpResponse:
    """Bounded HTTP response returned by a Jira transport."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class JiraHttpTransport(Protocol):
    """Sends an HTTP request with explicit connection, read, and size bounds."""

    def request(
        self,
        request: JiraHttpRequest,
        *,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_response_bytes: int,
    ) -> JiraHttpResponse: ...


class StdlibJiraHttpTransport:
    """HTTPS transport using ``http.client`` with verified TLS and no redirects."""

    def request(
        self,
        request: JiraHttpRequest,
        *,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_response_bytes: int,
    ) -> JiraHttpResponse:
        if request.method != "GET":
            raise JiraClientError("Jira transport supports only GET requests")
        parsed = urlsplit(request.url)
        if parsed.scheme != "https" or parsed.hostname is None:
            raise JiraClientError("Jira transport received an unsafe endpoint")
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise JiraClientError("Jira transport received an unsafe endpoint")
        try:
            port = parsed.port or 443
        except ValueError:
            raise JiraClientError("Jira transport received an unsafe endpoint") from None
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection = http.client.HTTPSConnection(
            parsed.hostname,
            port=port,
            timeout=connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
        response: http.client.HTTPResponse | None = None
        try:
            connection.request(request.method, target, headers=dict(request.headers))
            response = connection.getresponse()
            if connection.sock is not None:
                connection.sock.settimeout(read_timeout_seconds)
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    raise JiraClientError("Jira returned an invalid response length") from None
                if declared_length < 0 or declared_length > max_response_bytes:
                    raise JiraClientError("Jira response exceeded the configured size limit")
            body = response.read(max_response_bytes + 1)
            if len(body) > max_response_bytes:
                raise JiraClientError("Jira response exceeded the configured size limit")
            headers = {name.lower(): value for name, value in response.getheaders()}
            return JiraHttpResponse(status=response.status, headers=headers, body=body)
        except JiraClientError:
            raise
        except TimeoutError:
            raise JiraClientError("Jira request timed out") from None
        except ssl.SSLError:
            raise JiraClientError("Jira TLS validation failed") from None
        except (OSError, http.client.HTTPException):
            raise JiraClientError("Jira network request failed") from None
        finally:
            if response is not None:
                response.close()
            connection.close()


class JiraClient:
    """Read-only Jira REST client with allowlisted endpoints and secret indirection.

    Jira Cloud calls REST v3 with email/API-token Basic authentication. Jira Data
    Center calls REST v2 with personal-access-token Bearer authentication. Credentials
    are resolved for each public operation and are never retained on the client.
    """

    def __init__(
        self,
        connection: JiraConnection,
        allowed_hosts: Sequence[str],
        *,
        credential_provider: CredentialProvider | Callable[[str], str] | None = None,
        transport: JiraHttpTransport | None = None,
        connect_timeout_seconds: float = 10,
        read_timeout_seconds: float = 30,
        max_response_bytes: int = 2_000_000,
    ) -> None:
        if not isinstance(connection, JiraConnection):
            raise TypeError("connection must be a JiraConnection")
        if not _is_bounded_number(connect_timeout_seconds, maximum=60):
            raise ValueError("connect_timeout_seconds must be between 0 and 60")
        if not _is_bounded_number(read_timeout_seconds, maximum=300):
            raise ValueError("read_timeout_seconds must be between 0 and 300")
        if (
            not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or not 0 < max_response_bytes <= 20_000_000
        ):
            raise ValueError("max_response_bytes must be between 1 and 20000000")
        hosts = tuple(_validate_allowed_host(host) for host in allowed_hosts)
        if not hosts:
            raise ValueError("at least one Jira host must be allowlisted")
        if len(hosts) != len(set(hosts)):
            raise ValueError("allowlisted Jira hosts must be unique")
        hostname = urlsplit(connection.base_url).hostname
        if hostname is None or hostname.lower() not in hosts:
            raise ValueError("Jira base URL host is not allowlisted")

        self._connection = connection
        self._allowed_hosts = frozenset(hosts)
        self._credential_resolver = _credential_resolver(
            credential_provider or EnvironmentCredentialProvider()
        )
        self._transport = transport or StdlibJiraHttpTransport()
        self._connect_timeout_seconds = connect_timeout_seconds
        self._read_timeout_seconds = read_timeout_seconds
        self._max_response_bytes = max_response_bytes

    @property
    def connection(self) -> JiraConnection:
        """Return the immutable, credential-free connection configuration."""

        return self._connection

    def test_connection(self) -> JiraConnectionTestResult:
        """Probe Jira ``serverInfo`` and ``myself`` without exposing failure details."""

        server_title: str | None = None
        server_version: str | None = None
        try:
            authorization = self._authorization_header()
            server = self._request_json("serverInfo", authorization=authorization)
            server_title = _optional_scalar_text(server.get("serverTitle"), limit=500)
            server_version = _optional_scalar_text(server.get("version"), limit=100)
            myself = self._request_json("myself", authorization=authorization)
            authenticated_user = _first_text(
                myself,
                ("displayName", "emailAddress", "name", "accountId"),
                limit=500,
            )
            return JiraConnectionTestResult(
                ok=True,
                message="Jira connection succeeded",
                server_title=server_title,
                server_version=server_version,
                authenticated_user=authenticated_user,
            )
        except JiraClientError as error:
            return JiraConnectionTestResult(
                ok=False,
                message=str(error),
                server_title=server_title,
                server_version=server_version,
            )
        except Exception:
            return JiraConnectionTestResult(ok=False, message="Jira connection test failed")

    def fetch_issue(
        self,
        issue_key: str,
        acceptance_criteria_fields: Sequence[str] = (),
    ) -> JiraIssue:
        """Fetch and normalize one explicitly supplied Jira issue key."""

        if not self._connection.enabled:
            raise JiraClientError("Jira connection is disabled")
        key = _validate_issue_key(issue_key)
        acceptance_fields = _validate_field_ids(acceptance_criteria_fields)
        requested_fields = tuple(dict.fromkeys((*_DEFAULT_FIELDS, *acceptance_fields)))
        authorization = self._authorization_header()
        payload = self._request_json(
            f"issue/{quote(key, safe='')}",
            query={"fields": ",".join(requested_fields)},
            authorization=authorization,
        )
        return self._normalize_issue(payload, key, acceptance_fields)

    def _authorization_header(self) -> str:
        try:
            credential = self._credential_resolver(self._connection.credential_env)
        except JiraClientError:
            raise
        except Exception:
            raise JiraClientError("Jira credential could not be resolved") from None
        if (
            not isinstance(credential, str)
            or not credential
            or len(credential) > _MAX_CREDENTIAL_LENGTH
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in credential
            )
        ):
            raise JiraClientError("Jira credential is invalid")

        if self._connection.auth_type is JiraAuthType.API_TOKEN:
            username = self._connection.username
            if username is None:
                raise JiraClientError("Jira Cloud username is not configured")
            encoded = base64.b64encode(f"{username}:{credential}".encode()).decode("ascii")
            return f"Basic {encoded}"
        return f"Bearer {credential}"

    def _request_json(
        self,
        resource: str,
        *,
        authorization: str,
        query: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        api_version = "3" if self._connection.edition is JiraEdition.CLOUD else "2"
        url = f"{self._connection.base_url}/rest/api/{api_version}/{resource}"
        if query:
            url = f"{url}?{urlencode(query)}"
        self._validate_endpoint(url)
        request = JiraHttpRequest(
            method="GET",
            url=url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Authorization": authorization,
                "User-Agent": "ai-code-intelligence/0.1",
            },
        )
        try:
            response = self._transport.request(
                request,
                connect_timeout_seconds=self._connect_timeout_seconds,
                read_timeout_seconds=self._read_timeout_seconds,
                max_response_bytes=self._max_response_bytes,
            )
        except JiraClientError:
            raise
        except Exception:
            raise JiraClientError("Jira network request failed") from None
        if not isinstance(response, JiraHttpResponse):
            raise JiraClientError("Jira transport returned an invalid response")
        if (
            not isinstance(response.status, int)
            or isinstance(response.status, bool)
            or not isinstance(response.body, bytes)
            or not isinstance(response.headers, Mapping)
        ):
            raise JiraClientError("Jira transport returned an invalid response")
        if len(response.body) > self._max_response_bytes:
            raise JiraClientError("Jira response exceeded the configured size limit")
        if not 200 <= response.status < 300:
            raise JiraClientError(_http_error_message(response.status))
        content_type = _header(response.headers, "content-type")
        if content_type is not None and "json" not in content_type.lower():
            raise JiraClientError("Jira returned a non-JSON response")
        try:
            decoded = response.body.decode("utf-8")
            payload = json.loads(decoded, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
            raise JiraClientError("Jira returned malformed JSON") from None
        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise JiraClientError("Jira returned an invalid JSON object")
        return payload

    def _validate_endpoint(self, value: str) -> None:
        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError:
            raise JiraClientError("Jira endpoint validation failed") from None
        hostname = parsed.hostname
        if (
            parsed.scheme != "https"
            or hostname is None
            or hostname.lower() not in self._allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or port not in (None, 443)
        ):
            raise JiraClientError("Jira endpoint validation failed")

    def _normalize_issue(
        self,
        payload: Mapping[str, Any],
        requested_key: str,
        acceptance_fields: tuple[str, ...],
    ) -> JiraIssue:
        key = payload.get("key")
        fields = payload.get("fields")
        if key != requested_key or not isinstance(fields, dict):
            raise JiraClientError("Jira returned an invalid issue")
        project = fields.get("project")
        project_key = project.get("key") if isinstance(project, dict) else requested_key.rsplit("-", 1)[0]
        summary = fields.get("summary")
        if not isinstance(project_key, str) or not isinstance(summary, str):
            raise JiraClientError("Jira returned an invalid issue")

        acceptance_criteria = tuple(
            text
            for field_id in acceptance_fields
            if (text := _jira_field_text(fields.get(field_id))) is not None
        )
        labels_value = fields.get("labels")
        labels = (
            tuple(dict.fromkeys(label for label in labels_value if isinstance(label, str)))
            if isinstance(labels_value, list)
            else ()
        )
        try:
            return JiraIssue(
                key=key,
                project_key=project_key,
                summary=summary,
                description=_jira_field_text(fields.get("description")),
                acceptance_criteria=acceptance_criteria,
                status=_nested_text(fields.get("status"), "name"),
                issue_type=_nested_text(fields.get("issuetype"), "name"),
                assignee=_person_name(fields.get("assignee")),
                reporter=_person_name(fields.get("reporter")),
                labels=labels,
                url=f"{self._connection.base_url}/browse/{quote(key, safe='')}",
            )
        except ValidationError:
            raise JiraClientError("Jira returned invalid issue content") from None


def _credential_resolver(
    provider: CredentialProvider | Callable[[str], str],
) -> Callable[[str], str]:
    method = getattr(provider, "get_credential", None)
    if callable(method):
        return cast("Callable[[str], str]", method)
    if callable(provider):
        return provider
    raise TypeError("credential_provider must resolve a credential reference")


def _is_bounded_number(value: object, *, maximum: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= maximum


def _validate_allowed_host(value: str) -> str:
    if not isinstance(value, str) or value != value.strip() or value != value.lower():
        raise ValueError("allowlisted Jira hosts must be lowercase DNS names")
    if not _DNS_HOST.fullmatch(value) or value.endswith("."):
        raise ValueError("allowlisted Jira host is invalid")
    if value.replace(".", "").isdigit():
        raise ValueError("allowlisted Jira hosts must not be numeric addresses")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise ValueError("allowlisted Jira hosts must not be IP literals")
    return value


def _validate_issue_key(value: str) -> str:
    if not isinstance(value, str) or not _ISSUE_KEY.fullmatch(value):
        raise ValueError("Jira issue key is invalid")
    return value


def _validate_field_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("acceptance_criteria_fields must be a sequence of field IDs")
    fields = tuple(values)
    if len(fields) > 32 or len(fields) != len(set(fields)):
        raise ValueError("acceptance_criteria_fields must contain at most 32 unique fields")
    if any(not isinstance(field_id, str) or not _FIELD_ID.fullmatch(field_id) for field_id in fields):
        raise ValueError("acceptance_criteria_fields contains an invalid field ID")
    return fields


def _http_error_message(status: int) -> str:
    if status in {401, 403}:
        return "Jira authentication was rejected"
    if status == 404:
        return "Jira resource was not found"
    if status == 429:
        return "Jira rate limit was exceeded"
    if 500 <= status < 600:
        return "Jira server returned an error"
    return f"Jira request failed with HTTP status {status}"


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    return next((value for key, value in headers.items() if key.lower() == wanted), None)


def _reject_json_constant(value: str) -> None:
    del value
    raise ValueError("non-finite JSON number")


def _optional_scalar_text(value: object, *, limit: int) -> str | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized[:limit]


def _first_text(payload: Mapping[str, Any], fields: Sequence[str], *, limit: int) -> str | None:
    for field in fields:
        value = _optional_scalar_text(payload.get(field), limit=limit)
        if value is not None:
            return value
    return None


def _nested_text(value: object, field: str) -> str | None:
    return _jira_field_text(value.get(field)) if isinstance(value, dict) else None


def _person_name(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    return _first_text(value, ("displayName", "emailAddress", "name", "accountId"), limit=500)


def _jira_field_text(
    value: object,
    *,
    _depth: int = 0,
    _remaining: list[int] | None = None,
) -> str | None:
    remaining = _remaining if _remaining is not None else [_MAX_ADF_NODES]
    if _depth > _MAX_ADF_DEPTH or remaining[0] <= 0:
        raise JiraClientError("Jira field response exceeded normalization limits")
    remaining[0] -= 1
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        items = tuple(
            text
            for item in value
            if (
                text := _jira_field_text(
                    item,
                    _depth=_depth + 1,
                    _remaining=remaining,
                )
            )
            is not None
        )
        return "\n".join(items) or None
    if isinstance(value, dict):
        if value.get("type") == "doc" or isinstance(value.get("content"), list):
            return _adf_to_text(value)
        return _first_text(value, ("value", "name", "displayName"), limit=1_000_000)
    return None


def _adf_to_text(document: Mapping[str, Any]) -> str | None:
    remaining = [_MAX_ADF_NODES]

    def render(node: object, depth: int) -> str:
        if depth > _MAX_ADF_DEPTH or remaining[0] <= 0:
            raise JiraClientError("Jira rich-text response exceeded normalization limits")
        remaining[0] -= 1
        if not isinstance(node, dict):
            return ""
        node_type = node.get("type")
        if node_type == "text":
            text = node.get("text")
            return text if isinstance(text, str) else ""
        if node_type == "hardBreak":
            return "\n"
        if node_type == "rule":
            return "\n---\n"
        if node_type == "emoji":
            attrs = node.get("attrs")
            return (
                _first_text(attrs, ("text", "shortName"), limit=200) or "" if isinstance(attrs, dict) else ""
            )
        if node_type == "mention":
            attrs = node.get("attrs")
            return (
                _first_text(attrs, ("text", "displayName"), limit=500) or ""
                if isinstance(attrs, dict)
                else ""
            )
        content = node.get("content")
        rendered = "".join(render(child, depth + 1) for child in content) if isinstance(content, list) else ""
        if node_type in {
            "paragraph",
            "heading",
            "blockquote",
            "codeBlock",
            "panel",
            "listItem",
            "tableRow",
            "tableCell",
            "tableHeader",
        }:
            return f"{rendered}\n"
        if node_type in {"bulletList", "orderedList", "table", "doc"}:
            return rendered
        return rendered

    text = render(document, 0)
    lines = tuple(line.rstrip() for line in text.splitlines())
    normalized_lines: list[str] = []
    previous_empty = True
    for line in lines:
        empty = not line
        if empty and previous_empty:
            continue
        normalized_lines.append(line)
        previous_empty = empty
    normalized = "\n".join(normalized_lines).strip()
    return normalized or None
