"""Secure Jira configuration models and read-only REST integration client."""

from ai_code_intelligence.jira.client import (
    CredentialProvider,
    EnvironmentCredentialProvider,
    JiraClient,
    JiraClientError,
    JiraHttpRequest,
    JiraHttpResponse,
    JiraHttpTransport,
    StdlibJiraHttpTransport,
)
from ai_code_intelligence.jira.models import (
    JiraAuthType,
    JiraConnection,
    JiraConnectionTestResult,
    JiraEdition,
    JiraIssue,
    JiraProjectMapping,
    normalize_jira_base_url,
)

__all__ = [
    "CredentialProvider",
    "EnvironmentCredentialProvider",
    "JiraAuthType",
    "JiraClient",
    "JiraClientError",
    "JiraConnection",
    "JiraConnectionTestResult",
    "JiraEdition",
    "JiraHttpRequest",
    "JiraHttpResponse",
    "JiraHttpTransport",
    "JiraIssue",
    "JiraProjectMapping",
    "StdlibJiraHttpTransport",
    "normalize_jira_base_url",
]
