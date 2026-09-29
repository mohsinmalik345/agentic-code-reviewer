from __future__ import annotations

import re
from typing import Any

_SECRET_ASSIGNMENT = re.compile(
    r"(?im)^(?P<prefix>\s*[+-]?\s*(?:export\s+)?[A-Z][A-Z0-9_]*"
    r"(?:SECRET|PASSWORD|PASS|TOKEN|API_KEY|PRIVATE_KEY)"
    r"[A-Z0-9_]*\s*[=:]\s*)(?P<value>[^\r\n]+)$"
)
_SECRET_LITERAL = re.compile(
    r"(?i)(?P<prefix>['\"]?(?:api[_-]?key|secret|password|pass|token|private[_-]?key)"
    r"['\"]?\s*[:=]\s*)(?P<quote>['\"`])(?P<value>[^'\"`\r\n]+)(?P=quote)"
)
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")
_URI_CREDENTIAL = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/\s]+:)([^@/\s]+)(@)")


def redact_sensitive_text(value: str) -> str:
    """Redact common credentials while preserving line boundaries and source addressing."""

    value = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group('prefix')}<redacted>", value)
    value = _SECRET_LITERAL.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}<redacted>{match.group('quote')}",
        value,
    )
    value = _AWS_ACCESS_KEY.sub("<redacted-aws-access-key>", value)
    value = _BEARER_TOKEN.sub("Bearer <redacted>", value)
    return _URI_CREDENTIAL.sub(r"\1<redacted>\3", value)


def redact_json_strings(value: Any) -> Any:
    """Recursively redact every string in a JSON-compatible value."""

    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, (list, tuple)):
        return [redact_json_strings(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_json_strings(item) for key, item in value.items()}
    return value
