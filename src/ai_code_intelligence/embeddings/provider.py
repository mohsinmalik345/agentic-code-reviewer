from __future__ import annotations

import json
from typing import Any, Protocol

import boto3


class EmbeddingProvider(Protocol):
    """Provider-neutral embedding port."""

    @property
    def model_id(self) -> str: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


class TitanEmbeddingProvider:
    """Generates normalized Titan Text Embeddings V2 vectors through Bedrock Runtime."""

    def __init__(
        self,
        model_id: str,
        region: str,
        dimensions: int,
        *,
        runtime_client: Any | None = None,
    ) -> None:
        self._model_id = model_id
        self._dimensions = dimensions
        self._client = runtime_client or boto3.client("bedrock-runtime", region_name=region)

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            raise ValueError("cannot embed empty text")
        response = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {"inputText": text, "dimensions": self._dimensions, "normalize": True},
                separators=(",", ":"),
            ),
        )
        payload = json.loads(response["body"].read())
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != self._dimensions:
            raise RuntimeError(
                f"Titan returned {len(embedding) if isinstance(embedding, list) else 0} dimensions; "
                f"expected {self._dimensions}"
            )
        return tuple(float(value) for value in embedding)
