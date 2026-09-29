from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_code_intelligence.domain.models import RepositoryDefinition

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigModel(BaseModel):
    """Strict configuration base that rejects misspelled or unsupported keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScannerConfig(ConfigModel):
    """Safety and exclusion limits for read-only repository inventory."""

    ignored_directories: tuple[str, ...]
    max_file_bytes: int = Field(gt=0)
    max_repository_files: int = Field(default=100_000, ge=1_000, le=1_000_000)
    max_repository_bytes: int = Field(
        default=2_000_000_000,
        ge=10_000_000,
        le=100_000_000_000,
    )


class AgentConfig(ConfigModel):
    """Batching, validation, concurrency, and inference controls for agents."""

    native_structured_output: bool
    adaptive_thinking: bool
    validation_retries: int = Field(ge=0, le=2)
    max_batch_characters: int = Field(ge=10_000, le=500_000)
    max_files_per_batch: int = Field(ge=1, le=25)
    chunk_overlap_lines: int = Field(ge=0, le=200)
    max_discovery_characters: int = Field(ge=10_000, le=1_000_000)
    max_knowledge_context_characters: int = Field(ge=50_000, le=4_000_000)
    max_concurrent_invocations: int = Field(ge=1, le=20)
    audit_passes: int = Field(ge=0, le=2)
    max_output_tokens: int = Field(ge=1_000, le=128_000)
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    strict_evidence: bool
    max_evidence_rejection_ratio: float = Field(ge=0, le=1)
    guardrail_identifier_env: str
    guardrail_version_env: str


class AwsConfig(ConfigModel):
    """AWS Region and configurable Bedrock model identifiers and limits."""

    region: str
    analysis_model_id: str
    analysis_context_window_tokens: int = Field(default=200_000, ge=8_000, le=2_000_000)
    analysis_max_output_tokens: int = Field(default=128_000, ge=1_000, le=1_000_000)
    embedding_model_id: str
    embedding_dimensions: Literal[1024]
    connect_timeout_seconds: int = Field(default=10, ge=1, le=60)
    read_timeout_seconds: int = Field(default=3600, ge=60, le=3600)
    retry_max_attempts: int = Field(default=6, ge=1, le=10)


class Neo4jConfig(ConfigModel):
    """Neo4j connection and database settings."""

    uri: str
    username: str
    password_env: str
    database: str


class PostgresConfig(ConfigModel):
    """PostgreSQL connection indirection and schema name."""

    connection_string_env: str
    schema_name: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")


class KnowledgeConfig(ConfigModel):
    """Local or S3 knowledge-document storage settings."""

    driver: Literal["local", "s3"]
    local_directory: str
    s3_bucket_env: str
    s3_prefix: str


class ArchitectureChatConfig(ConfigModel):
    """Grounded architecture-chat retrieval and Bedrock inference limits."""

    enabled: bool = True
    model_id: str = Field(min_length=1)
    context_window_tokens: int = Field(default=128_000, ge=8_000, le=2_000_000)
    max_output_tokens: int = Field(default=4_096, ge=256, le=16_000)
    max_context_characters: int = Field(default=180_000, ge=20_000, le=500_000)
    max_question_characters: int = Field(default=4_000, ge=100, le=20_000)
    max_history_messages: int = Field(default=8, ge=0, le=20)
    retrieval_chunks: int = Field(default=12, ge=2, le=30)
    chunk_size_characters: int = Field(default=8_000, ge=1_000, le=20_000)
    max_concurrent_requests: int = Field(default=2, ge=1, le=10)
    graph_retrieval_enabled: bool = True
    graph_seed_nodes: int = Field(default=8, ge=1, le=50)
    graph_max_nodes: int = Field(default=24, ge=1, le=200)
    graph_max_relationships: int = Field(default=36, ge=1, le=500)
    graph_evidence_characters: int = Field(default=2_000, ge=500, le=8_000)

    @model_validator(mode="after")
    def graph_limits_include_seeds(self) -> ArchitectureChatConfig:
        if self.graph_max_nodes < self.graph_seed_nodes:
            raise ValueError("chat.graph_max_nodes cannot be smaller than chat.graph_seed_nodes")
        return self


class EmbeddingsConfig(ConfigModel):
    """Embedding feature switch and deterministic chunk sizes."""

    enabled: bool
    chunk_size_characters: int = Field(ge=1_000, le=45_000)
    chunk_overlap_characters: int = Field(ge=0)
    max_input_utf8_bytes: int = Field(default=7_000, ge=1_000, le=8_000)

    @model_validator(mode="after")
    def valid_overlap(self) -> EmbeddingsConfig:
        if self.chunk_overlap_characters >= self.chunk_size_characters:
            raise ValueError("embedding overlap must be smaller than chunk size")
        if self.chunk_overlap_characters >= self.max_input_utf8_bytes:
            raise ValueError("embedding overlap must be smaller than maximum UTF-8 bytes")
        return self


class AnalysisConfig(ConfigModel):
    """Impact traversal and fail-closed report settings."""

    max_traversal_depth: int = Field(ge=1, le=20)
    max_traversal_nodes: int = Field(ge=100, le=100_000)
    fail_closed_on_ai_error: bool
    report_directory: str


class GraphqlConfig(ConfigModel):
    """GraphQL bind address, authentication, and introspection policy."""

    host: str
    port: int = Field(ge=1, le=65_535)
    path: str = Field(pattern=r"^/[A-Za-z0-9/_-]*$")
    api_key_env: str
    allow_introspection: bool


class GitLabConfig(ConfigModel):
    """Managed GitLab checkout policy; credentials remain environment-backed."""

    checkout_root: str
    allowed_hosts: tuple[str, ...] = Field(min_length=1)
    token_env: str
    username: str = Field(default="oauth2", min_length=1)
    clone_timeout_seconds: int = Field(default=900, ge=30, le=7200)
    fetch_timeout_seconds: int = Field(default=600, ge=30, le=7200)

    @model_validator(mode="after")
    def normalized_hosts(self) -> GitLabConfig:
        if any(host != host.lower() or ":" in host or "/" in host for host in self.allowed_hosts):
            raise ValueError("gitlab.allowed_hosts must contain lowercase hostnames without ports")
        if len(set(self.allowed_hosts)) != len(self.allowed_hosts):
            raise ValueError("gitlab.allowed_hosts must be unique")
        return self


class JiraConfig(ConfigModel):
    """Jira connectivity policy and credential-free configuration storage."""

    allowed_hosts: tuple[str, ...] = ()
    local_path: str = "artifacts/jira-configuration.json"
    connect_timeout_seconds: int = Field(default=10, ge=1, le=60)
    read_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_response_bytes: int = Field(default=2_000_000, ge=10_000, le=20_000_000)

    @model_validator(mode="after")
    def normalized_hosts(self) -> JiraConfig:
        if any(host != host.lower() or ":" in host or "/" in host for host in self.allowed_hosts):
            raise ValueError("jira.allowed_hosts must contain lowercase hostnames without ports")
        if len(set(self.allowed_hosts)) != len(self.allowed_hosts):
            raise ValueError("jira.allowed_hosts must be unique")
        return self


class RepositoryRegistryConfig(ConfigModel):
    """Durable repository catalog and indexing-job queue configuration."""

    driver: Literal["local", "postgres"]
    local_path: str
    worker_poll_seconds: float = Field(default=2, ge=0.1, le=60)
    job_lease_seconds: int = Field(default=43_200, ge=300, le=86_400)


class ApplicationConfig(ConfigModel):
    """Fully resolved application configuration."""

    version: Literal[1]
    repositories: tuple[RepositoryDefinition, ...]
    scanner: ScannerConfig
    agents: AgentConfig
    aws: AwsConfig
    neo4j: Neo4jConfig
    postgres: PostgresConfig
    knowledge: KnowledgeConfig
    chat: ArchitectureChatConfig
    embeddings: EmbeddingsConfig
    analysis: AnalysisConfig
    graphql: GraphqlConfig
    gitlab: GitLabConfig
    jira: JiraConfig = Field(default_factory=JiraConfig)
    repository_registry: RepositoryRegistryConfig
    project_root: Path
    config_path: Path

    @model_validator(mode="after")
    def unique_repository_ids(self) -> ApplicationConfig:
        ids = [repository.id for repository in self.repositories]
        if len(ids) != len(set(ids)):
            raise ValueError("repository ids must be unique")
        if self.agents.max_output_tokens > self.aws.analysis_max_output_tokens:
            raise ValueError("agents.max_output_tokens cannot exceed aws.analysis_max_output_tokens")
        return self


def load_config(path: str | Path) -> ApplicationConfig:
    """Load, validate, resolve, and environment-override a YAML configuration file."""

    config_path = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a YAML mapping")

    project_root = config_path.parent.parent
    _load_environment_file(project_root / ".env")
    repository_values = raw.get("repositories")
    if not isinstance(repository_values, list):
        raise ValueError("repositories must be a list")
    resolved_repositories: list[dict[str, object]] = []
    for value in repository_values:
        if not isinstance(value, dict):
            raise ValueError("each repository must be a mapping")
        repository = dict(value)
        raw_path = repository.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("repository path must be a string")
        candidate = Path(raw_path).expanduser()
        repository["path"] = str(
            candidate if candidate.is_absolute() else (project_root / candidate).resolve()
        )
        resolved_repositories.append(repository)

    raw["repositories"] = resolved_repositories
    raw["project_root"] = project_root
    raw["config_path"] = config_path

    aws = raw.get("aws")
    if not isinstance(aws, dict):
        raise ValueError("aws must be a mapping")
    aws = dict(aws)
    aws["region"] = os.getenv("AWS_REGION") or aws.get("region")
    aws["analysis_model_id"] = os.getenv("BEDROCK_ANALYSIS_MODEL_ID") or aws.get("analysis_model_id")
    aws["analysis_context_window_tokens"] = os.getenv("BEDROCK_ANALYSIS_CONTEXT_WINDOW_TOKENS") or aws.get(
        "analysis_context_window_tokens", 200_000
    )
    aws["analysis_max_output_tokens"] = os.getenv("BEDROCK_ANALYSIS_MAX_OUTPUT_TOKENS") or aws.get(
        "analysis_max_output_tokens", 128_000
    )
    aws["embedding_model_id"] = os.getenv("BEDROCK_EMBEDDING_MODEL_ID") or aws.get("embedding_model_id")
    aws["connect_timeout_seconds"] = os.getenv("BEDROCK_CONNECT_TIMEOUT_SECONDS") or aws.get(
        "connect_timeout_seconds", 10
    )
    aws["read_timeout_seconds"] = os.getenv("BEDROCK_READ_TIMEOUT_SECONDS") or aws.get(
        "read_timeout_seconds", 3600
    )
    aws["retry_max_attempts"] = os.getenv("BEDROCK_RETRY_MAX_ATTEMPTS") or aws.get("retry_max_attempts", 6)
    raw["aws"] = aws

    chat = raw.get("chat")
    if not isinstance(chat, dict):
        raise ValueError("chat must be a mapping")
    chat = dict(chat)
    chat["model_id"] = os.getenv("BEDROCK_CHAT_MODEL_ID") or chat.get("model_id")
    chat["max_output_tokens"] = os.getenv("BEDROCK_CHAT_MAX_OUTPUT_TOKENS") or chat.get(
        "max_output_tokens", 4_096
    )
    raw["chat"] = chat

    for section, key in (
        ("knowledge", "local_directory"),
        ("analysis", "report_directory"),
        ("gitlab", "checkout_root"),
        ("repository_registry", "local_path"),
    ):
        values = raw.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        values = dict(values)
        location = Path(str(values[key])).expanduser()
        values[key] = str(location if location.is_absolute() else (project_root / location).resolve())
        raw[section] = values

    gitlab = raw.get("gitlab")
    if not isinstance(gitlab, dict):
        raise ValueError("gitlab must be a mapping")
    gitlab = dict(gitlab)
    allowed_hosts = os.getenv("GITLAB_ALLOWED_HOSTS")
    if allowed_hosts:
        gitlab["allowed_hosts"] = [host.strip().lower() for host in allowed_hosts.split(",") if host.strip()]
    raw["gitlab"] = gitlab

    jira = raw.get("jira", {})
    if not isinstance(jira, dict):
        raise ValueError("jira must be a mapping")
    jira = dict(jira)
    allowed_jira_hosts = os.getenv("JIRA_ALLOWED_HOSTS")
    if allowed_jira_hosts:
        jira["allowed_hosts"] = [
            host.strip().lower()
            for host in allowed_jira_hosts.split(",")
            if host.strip()
        ]
    jira_location = Path(
        str(jira.get("local_path", "artifacts/jira-configuration.json"))
    ).expanduser()
    jira["local_path"] = str(
        jira_location
        if jira_location.is_absolute()
        else (project_root / jira_location).resolve()
    )
    raw["jira"] = jira

    return ApplicationConfig.model_validate(raw)


def _load_environment_file(path: Path) -> None:
    """Load a project-local .env without overriding process or workload credentials."""

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return
    except OSError as error:
        raise RuntimeError(f"unable to read environment file {path}") from error

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"invalid .env assignment on line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            os.environ.setdefault(name, value)


def required_secret(environment_name: str) -> str:
    """Return a non-empty secret from the environment or raise a safe error."""

    value = os.getenv(environment_name)
    if value is None or not value.strip():
        raise RuntimeError(f"required environment variable {environment_name} is not set")
    return value


def optional_secret(environment_name: str) -> str | None:
    """Return an optional non-empty environment secret without exposing its value."""

    value = os.getenv(environment_name)
    return value.strip() if value is not None and value.strip() else None
