from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast
from uuid import uuid4

import boto3
from botocore.config import Config as BotocoreConfig
from pydantic import BaseModel, ValidationError

from ai_code_intelligence.agents.contracts import (
    KNOWLEDGE_EVIDENCE_SUMMARY_MAX_CHARACTERS,
    AgentRelationship,
    CodeGraphOutput,
    KnowledgeEvidenceSummary,
)
from ai_code_intelligence.config import AgentConfig, AwsConfig
from ai_code_intelligence.domain.models import AgentInvocation, AgentRole

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_INPUT_CHARACTERS_PER_TOKEN_ESTIMATE = 2
_CONTEXT_SAFETY_MARGIN_TOKENS = 4_096


@dataclass(frozen=True, slots=True)
class StructuredRequest(Generic[ResponseT]):
    """Typed request contract for a schema-validated specialist invocation."""

    role: AgentRole
    schema_name: str
    schema_description: str
    response_model: type[ResponseT]
    system_prompt: str
    user_prompt: str
    repository_id: str | None = None
    batch_id: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredResponse(Generic[ResponseT]):
    """Validated specialist value paired with invocation telemetry."""

    value: ResponseT
    invocation: AgentInvocation


@dataclass(frozen=True, slots=True)
class _QuarantinedCodeGraphOutput:
    value: CodeGraphOutput
    dropped_relationship_count: int


@dataclass(frozen=True, slots=True)
class _BoundedKnowledgeEvidenceSummary:
    value: KnowledgeEvidenceSummary
    original_character_count: int


@dataclass(frozen=True, slots=True)
class _RecoveredStructuredOutput(Generic[ResponseT]):
    value: ResponseT
    removed_leading_braces: int
    removed_trailing_braces: int


class StructuredModelClient(Protocol):
    """Port implemented by Bedrock and by deterministic test doubles."""

    def invoke(self, request: StructuredRequest[ResponseT]) -> StructuredResponse[ResponseT]: ...


class BedrockConverseClient:
    """Calls Bedrock Converse and accepts output only after strict local schema validation."""

    def __init__(
        self,
        aws: AwsConfig,
        agents: AgentConfig,
        *,
        runtime_client: Any | None = None,
    ) -> None:
        self._aws = aws
        self._agents = agents
        self._client = runtime_client or boto3.client(
            "bedrock-runtime",
            region_name=aws.region,
            config=BotocoreConfig(
                connect_timeout=aws.connect_timeout_seconds,
                read_timeout=aws.read_timeout_seconds,
                tcp_keepalive=True,
                retries={"total_max_attempts": aws.retry_max_attempts, "mode": "standard"},
            ),
        )
        self._logger = logging.getLogger(__name__)

    def invoke(self, request: StructuredRequest[ResponseT]) -> StructuredResponse[ResponseT]:
        invocation_id = str(uuid4())
        started = time.perf_counter()
        schema = _agent_schema(request.response_model)
        prompt = _json_prompt(
            request.user_prompt,
            request.schema_description,
            None if self._agents.native_structured_output else schema,
        )
        last_error: ValidationError | json.JSONDecodeError | None = None
        total_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        validated_value: ResponseT | None = None
        quarantined_value: _QuarantinedCodeGraphOutput | None = None
        bounded_summary: _BoundedKnowledgeEvidenceSummary | None = None
        recovered_output: _RecoveredStructuredOutput[ResponseT] | None = None

        for attempt in range(self._agents.validation_retries + 1):
            output_token_ceiling = self._output_token_ceiling(request, prompt, schema)
            max_tokens = min(self._agents.max_output_tokens, output_token_ceiling)
            while True:
                response = self._converse(request, prompt, schema, invocation_id, max_tokens)
                usage = response.get("usage", {})
                for key in total_usage:
                    total_usage[key] += int(usage.get(key, 0))
                stop_reason = response.get("stopReason")
                if stop_reason == "end_turn":
                    break
                if stop_reason == "max_tokens" and max_tokens < output_token_ceiling:
                    next_max_tokens = min(max_tokens * 2, output_token_ceiling)
                    self._logger.warning(
                        "Bedrock specialist reached its output limit; retrying with a larger limit",
                        extra={
                            "fields": {
                                "agent_role": request.role.value,
                                "invocation_id": invocation_id,
                                "repository_id": request.repository_id,
                                "batch_id": request.batch_id,
                                "previous_max_tokens": max_tokens,
                                "next_max_tokens": next_max_tokens,
                                "output_tokens": int(usage.get("outputTokens", 0)),
                            }
                        },
                    )
                    max_tokens = next_max_tokens
                    continue
                raise RuntimeError(
                    f"Bedrock {request.role.value} agent stopped with reason "
                    f"{stop_reason or 'unknown'} after {int(usage.get('outputTokens', 0))} output tokens"
                )
            text = _response_text(response)
            try:
                validated_value = request.response_model.model_validate_json(text)
            except (ValidationError, json.JSONDecodeError) as error:
                last_error = error
                recovered_output = _structured_output_with_redundant_braces(
                    text,
                    request.response_model,
                )
                if recovered_output is not None:
                    validated_value = recovered_output.value
                    break
                if request.response_model is CodeGraphOutput:
                    candidate = _code_graph_output_with_quarantined_relationships(text)
                    if candidate is not None:
                        quarantined_value = candidate
                elif request.response_model is KnowledgeEvidenceSummary:
                    summary_candidate = _knowledge_evidence_summary_with_bounded_content(text)
                    if summary_candidate is not None:
                        bounded_summary = summary_candidate
                        validated_value = cast(ResponseT, summary_candidate.value)
                        break
                if attempt >= self._agents.validation_retries:
                    break
                prompt = _repair_prompt(
                    prompt,
                    text,
                    error,
                    None if self._agents.native_structured_output else schema,
                )
                continue
            break

        if recovered_output is not None:
            self._logger.warning(
                "Bedrock structured output recovered from redundant JSON braces",
                extra={
                    "fields": {
                        "agent_role": request.role.value,
                        "invocation_id": invocation_id,
                        "repository_id": request.repository_id,
                        "batch_id": request.batch_id,
                        "removed_leading_braces": recovered_output.removed_leading_braces,
                        "removed_trailing_braces": recovered_output.removed_trailing_braces,
                    }
                },
            )

        if bounded_summary is not None:
            self._logger.warning(
                "Bedrock knowledge evidence summary deterministically bounded",
                extra={
                    "fields": {
                        "agent_role": request.role.value,
                        "invocation_id": invocation_id,
                        "repository_id": request.repository_id,
                        "batch_id": request.batch_id,
                        "original_character_count": bounded_summary.original_character_count,
                        "retained_character_count": len(bounded_summary.value.summary),
                        "retained_node_citation_count": len(
                            bounded_summary.value.evidence_node_ids
                        ),
                        "retained_relationship_citation_count": len(
                            bounded_summary.value.evidence_relationship_ids
                        ),
                    }
                },
            )

        if validated_value is None:
            if quarantined_value is None:
                self._logger.warning(
                    "Bedrock structured output validation failed",
                    extra={
                        "fields": {
                            "agent_role": request.role.value,
                            "invocation_id": invocation_id,
                            "repository_id": request.repository_id,
                            "batch_id": request.batch_id,
                            "validation_kind": _validation_error_kind(last_error),
                            "validation_attempts": self._agents.validation_retries + 1,
                        }
                    },
                )
                raise RuntimeError(
                    f"Bedrock {request.role.value} agent returned invalid JSON after "
                    f"{self._agents.validation_retries + 1} attempt(s): {last_error}"
                )
            validated_value = cast(ResponseT, quarantined_value.value)
            self._logger.warning(
                "Bedrock code graph relationships quarantined after structured-output validation failed",
                extra={
                    "fields": {
                        "agent_role": request.role.value,
                        "invocation_id": invocation_id,
                        "repository_id": request.repository_id,
                        "batch_id": request.batch_id,
                        "dropped_relationship_count": quarantined_value.dropped_relationship_count,
                        "validation_attempts": self._agents.validation_retries + 1,
                    }
                },
            )

        invocation = AgentInvocation(
            role=request.role,
            invocation_id=invocation_id,
            model_id=self._aws.analysis_model_id,
            repository_id=request.repository_id,
            batch_id=request.batch_id,
            input_tokens=total_usage["inputTokens"],
            output_tokens=total_usage["outputTokens"],
            total_tokens=total_usage["totalTokens"],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._logger.info(
            "Bedrock specialist completed",
            extra={
                "fields": {
                    "agent_role": request.role.value,
                    "invocation_id": invocation_id,
                    "repository_id": request.repository_id,
                    "batch_id": request.batch_id,
                    "input_tokens": invocation.input_tokens,
                    "output_tokens": invocation.output_tokens,
                    "latency_ms": invocation.latency_ms,
                }
            },
        )
        return StructuredResponse(value=validated_value, invocation=invocation)

    def _output_token_ceiling(
        self,
        request: StructuredRequest[ResponseT],
        prompt: str,
        schema: dict[str, Any],
    ) -> int:
        schema_characters = (
            len(json.dumps(schema, separators=(",", ":"))) if self._agents.native_structured_output else 0
        )
        estimated_input_tokens = math.ceil(
            (len(request.system_prompt) + len(prompt) + schema_characters)
            / _INPUT_CHARACTERS_PER_TOKEN_ESTIMATE
        )
        available = (
            self._aws.analysis_context_window_tokens - estimated_input_tokens - _CONTEXT_SAFETY_MARGIN_TOKENS
        )
        ceiling = min(self._aws.analysis_max_output_tokens, available)
        if ceiling < 1_000:
            raise RuntimeError(
                f"Bedrock {request.role.value} prompt is too large for the configured "
                f"{self._aws.analysis_context_window_tokens}-token context window; "
                "reduce the applicable agent character limit or repository scope"
            )
        return ceiling

    def _converse(
        self,
        request: StructuredRequest[ResponseT],
        prompt: str,
        schema: dict[str, Any],
        invocation_id: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "modelId": self._aws.analysis_model_id,
            "system": [{"text": request.system_prompt}],
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            # Sampling is intentionally omitted; prompts and schema constraints drive consistency.
            "inferenceConfig": {"maxTokens": max_tokens},
            "requestMetadata": {
                "agentRole": request.role.value,
                "invocationId": invocation_id,
                **({"repositoryId": request.repository_id} if request.repository_id else {}),
                **({"batchId": request.batch_id} if request.batch_id else {}),
            },
        }
        if self._agents.adaptive_thinking:
            payload["additionalModelRequestFields"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": self._agents.effort},
            }
        if self._agents.native_structured_output:
            payload["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": request.schema_name,
                            "description": request.schema_description,
                            "schema": json.dumps(schema, separators=(",", ":")),
                        }
                    },
                }
            }

        guardrail_id = _environment_value(self._agents.guardrail_identifier_env)
        guardrail_version = _environment_value(self._agents.guardrail_version_env)
        if guardrail_id and guardrail_version:
            payload["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
                "trace": "enabled",
            }

        response: dict[str, Any] = self._client.converse(**payload)
        return response


def _agent_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the Bedrock-supported JSON Schema subset for a Pydantic response model."""

    return _sanitize_schema(model.model_json_schema())


def _structured_output_with_redundant_braces[RecoveredT: BaseModel](
    text: str,
    model: type[RecoveredT],
) -> _RecoveredStructuredOutput[RecoveredT] | None:
    """Recover one schema-valid object wrapped only in redundant JSON braces.

    Some open-weight Converse responses have been observed to prepend an extra
    opening brace to an otherwise grammar-valid JSON object. Recovery stays
    fail-closed: prose, Markdown fences, multiple valid objects, and any other
    non-whitespace prefix or suffix are rejected.
    """

    decoder = json.JSONDecoder()
    candidates: list[_RecoveredStructuredOutput[RecoveredT]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        prefix = text[:index].strip()
        if prefix and set(prefix) != {"{"}:
            continue
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        suffix = text[end:].strip()
        if suffix and set(suffix) != {"}"}:
            continue
        try:
            value = model.model_validate(payload)
        except ValidationError:
            continue
        candidates.append(
            _RecoveredStructuredOutput(
                value=value,
                removed_leading_braces=len(prefix),
                removed_trailing_braces=len(suffix),
            )
        )
    if len(candidates) != 1:
        return None
    return candidates[0]


_KNOWLEDGE_CITATION_PATTERN = re.compile(
    r"\[(node|relationship):([^\]\r\n]+)\]"
)


def _knowledge_evidence_summary_with_bounded_content(
    text: str,
) -> _BoundedKnowledgeEvidenceSummary | None:
    """Extract a strict, bounded evidence memo from an oversized model response.

    Recovery is intentionally limited to JSON that fails only because ``summary`` is
    too long. The untouched envelope is validated with a short sentinel first, so
    wrong types, missing required fields, and unknown fields still fail closed. When
    graph citations are present, only complete citation-bearing statements are kept
    and the evidence arrays are rebuilt from those exact retained citations. Normal
    graph existence and request-scope checks still run in the knowledge layer.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    raw_summary = payload.get("summary")
    if (
        not isinstance(raw_summary, str)
        or len(raw_summary) <= KNOWLEDGE_EVIDENCE_SUMMARY_MAX_CHARACTERS
    ):
        return None

    envelope_payload = dict(payload)
    envelope_payload["summary"] = "Schema-valid length sentinel."
    try:
        envelope = KnowledgeEvidenceSummary.model_validate(envelope_payload)
    except ValidationError:
        return None

    all_citations = tuple(_KNOWLEDGE_CITATION_PATTERN.finditer(raw_summary))
    if (
        not all_citations
        and (envelope.evidence_node_ids or envelope.evidence_relationship_ids)
    ):
        # Array-only provenance cannot be tied to a retained extract, so retry rather
        # than treating an arbitrary prefix as grounded.
        return None

    segments = _complete_summary_segments(raw_summary)
    if all_citations:
        segments = tuple(
            segment for segment in segments if _KNOWLEDGE_CITATION_PATTERN.search(segment)
        )
    bounded_summary = _pack_summary_segments(
        segments,
        KNOWLEDGE_EVIDENCE_SUMMARY_MAX_CHARACTERS,
    )
    if not bounded_summary:
        return None

    retained_citations = tuple(_KNOWLEDGE_CITATION_PATTERN.finditer(bounded_summary))
    if all_citations and not retained_citations:
        return None
    node_ids = tuple(
        sorted(
            {
                match.group(2).strip()
                for match in retained_citations
                if match.group(1) == "node"
            }
        )
    )
    relationship_ids = tuple(
        sorted(
            {
                match.group(2).strip()
                for match in retained_citations
                if match.group(1) == "relationship"
            }
        )
    )
    try:
        value = KnowledgeEvidenceSummary(
            summary=bounded_summary,
            evidence_node_ids=node_ids,
            evidence_relationship_ids=relationship_ids,
        )
    except ValidationError:
        return None
    return _BoundedKnowledgeEvidenceSummary(
        value=value,
        original_character_count=len(raw_summary),
    )


def _complete_summary_segments(summary: str) -> tuple[str, ...]:
    """Return complete lines, splitting pathological long lines at prose boundaries."""

    segments: list[str] = []
    for line in re.split(r"[\r\n]+", summary):
        normalized = line.strip()
        if not normalized:
            continue
        if len(normalized) <= KNOWLEDGE_EVIDENCE_SUMMARY_MAX_CHARACTERS:
            segments.append(normalized)
            continue
        segments.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?;])\s+", normalized)
            if part.strip()
            and len(part.strip()) <= KNOWLEDGE_EVIDENCE_SUMMARY_MAX_CHARACTERS
        )
    return tuple(segments)


def _pack_summary_segments(segments: tuple[str, ...], maximum: int) -> str:
    """Pack exact complete segments in source order without truncating a claim."""

    retained: list[str] = []
    size = 0
    for segment in segments:
        added = len(segment) + (1 if retained else 0)
        if size + added > maximum:
            continue
        retained.append(segment)
        size += added
    return "\n".join(retained)


def _code_graph_output_with_quarantined_relationships(
    text: str,
) -> _QuarantinedCodeGraphOutput | None:
    """Strictly retain valid graph facts while dropping only invalid relationship records.

    Nodes, warnings, envelope keys, and the relationships container remain fail-closed. This
    deliberately does not repair or synthesize any relationship field: each retained record
    must independently satisfy the normal ``AgentRelationship`` contract.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    raw_relationships = payload.get("relationships", [])
    if not isinstance(raw_relationships, list):
        return None

    envelope_payload = dict(payload)
    envelope_payload["relationships"] = []
    try:
        envelope = CodeGraphOutput.model_validate(envelope_payload)
    except ValidationError:
        return None

    relationships: list[AgentRelationship] = []
    quarantine_warnings: list[str] = []
    for index, raw_relationship in enumerate(raw_relationships):
        try:
            relationships.append(AgentRelationship.model_validate(raw_relationship))
        except ValidationError as error:
            quarantine_warnings.append(_relationship_quarantine_warning(index, raw_relationship, error))

    if not quarantine_warnings:
        return None
    value = envelope.model_copy(
        update={
            "relationships": tuple(relationships),
            "warnings": (*envelope.warnings, *quarantine_warnings),
        }
    )
    return _QuarantinedCodeGraphOutput(
        value=value,
        dropped_relationship_count=len(quarantine_warnings),
    )


def _relationship_quarantine_warning(
    index: int,
    relationship: object,
    error: ValidationError,
) -> str:
    identity = "unknown relationship"
    if isinstance(relationship, dict):
        relationship_type = _bounded_warning_value(relationship.get("type"))
        source = _bounded_warning_value(relationship.get("source_ref"))
        target = _bounded_warning_value(relationship.get("target_ref"))
        identity = f"{relationship_type} {source} -> {target}"
    details = "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()
    )
    return (
        f"Quarantined relationship candidate at response index {index} ({identity}); "
        f"no relationship was created: {details[:1000]}"
    )


def _bounded_warning_value(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    return " ".join(value.split())[:160]


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"$defs", "definitions", "properties"} and isinstance(value, dict):
            result[key] = {
                name: _sanitize_schema(definition)
                for name, definition in value.items()
                if isinstance(definition, dict)
            }
        elif key in {"items"} and isinstance(value, dict):
            result[key] = _sanitize_schema(value)
        elif key in {"anyOf", "allOf"} and isinstance(value, list):
            result[key] = [_sanitize_schema(item) for item in value if isinstance(item, dict)]
        elif key == "additionalProperties":
            if value is not False:
                raise ValueError(
                    "Bedrock structured output requires additionalProperties=false; "
                    "free-form mappings are not supported in agent response models"
                )
            result[key] = False
        elif key == "minItems":
            if value in {0, 1}:
                result[key] = value
        elif key in {
            "$ref",
            "type",
            "required",
            "enum",
            "const",
            "description",
            "format",
        }:
            result[key] = value

    if result.get("type") == "object" or "properties" in result:
        result["additionalProperties"] = False
    return result


def _json_prompt(
    user_prompt: str,
    description: str,
    schema: dict[str, Any] | None,
) -> str:
    prompt = (
        f"{user_prompt}\n\n"
        "Return exactly one JSON object and no markdown fences or commentary. "
        f"The object represents {description}."
    )
    if schema is None:
        return prompt
    return f"{prompt} It MUST validate against this JSON Schema:\n{json.dumps(schema, separators=(',', ':'))}"


def _repair_prompt(
    original_prompt: str,
    invalid_text: str,
    error: ValidationError | json.JSONDecodeError,
    schema: dict[str, Any] | None,
) -> str:
    prompt = (
        f"{original_prompt}\n\n"
        "Your previous response failed validation. Correct only its JSON shape and values; do not add unsupported facts.\n"
        f"Validation error: {str(error)[:2000]}\n"
        f"Invalid response: {invalid_text[:20000]}"
    )
    if schema is None:
        return prompt
    return f"{prompt}\nRequired schema: {json.dumps(schema, separators=(',', ':'))}"


def _response_text(response: dict[str, Any]) -> str:
    output = response.get("output")
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        raise RuntimeError("Bedrock returned no message content")
    values = [
        block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    text = "\n".join(values).strip()
    if not text:
        raise RuntimeError("Bedrock returned no text content")
    return text


def _environment_value(name: str) -> str | None:
    if not name:
        return None
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _validation_error_kind(error: ValidationError | json.JSONDecodeError | None) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "json_syntax"
    if isinstance(error, ValidationError):
        return "schema_validation"
    return "unknown"
