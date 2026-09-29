from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import boto3
from botocore.exceptions import ClientError

from ai_code_intelligence.domain.knowledge import (
    KnowledgeDocument,
    KnowledgeManifest,
    StoredDocument,
)


class DocumentStore(Protocol):
    """Storage port for generated knowledge documents."""

    def put(self, document: KnowledgeDocument) -> StoredDocument: ...

    def get(self, uri: str) -> str: ...

    def latest_manifest(self) -> KnowledgeManifest | None: ...


class ManifestPublisher(Protocol):
    """Publishes the pointer to the latest fully persisted portfolio snapshot."""

    def publish_manifest(self, manifest: Mapping[str, Any]) -> str: ...

    def delete_manifest(self) -> None: ...


class LocalDocumentStore:
    """Stores Markdown atomically on local disk for development and test runs."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def put(self, document: KnowledgeDocument) -> StoredDocument:
        relative = _document_suffix(document)
        target = (self._root / relative).resolve()
        if self._root != target and self._root not in target.parents:
            raise ValueError("knowledge target escaped configured directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(document.markdown, encoding="utf-8", newline="\n")
        temporary.replace(target)
        payload = document.markdown.encode("utf-8")
        return StoredDocument(
            document_id=document.id,
            uri=str(target),
            content_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def get(self, uri: str) -> str:
        target = self._target(uri, strict=True)
        return target.read_text(encoding="utf-8")

    def latest_manifest(self) -> KnowledgeManifest | None:
        """Read and validate the latest completed local portfolio, when one exists."""

        target = self._root / "latest.json"
        try:
            payload = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        manifest = KnowledgeManifest.model_validate_json(payload)
        for uri in _manifest_uris(manifest):
            self._target(uri, strict=False)
        return manifest

    def publish_manifest(self, manifest: Mapping[str, Any]) -> str:
        """Atomically publish a local development manifest after a successful run."""

        target = (self._root / "latest.json").resolve()
        if self._root != target and self._root not in target.parents:
            raise ValueError("knowledge manifest escaped configured directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(target)
        return str(target)

    def delete_manifest(self) -> None:
        """Remove only the stable publication pointer during a failed first publication."""

        target = (self._root / "latest.json").resolve()
        if self._root != target and self._root not in target.parents:
            raise ValueError("knowledge manifest escaped configured directory")
        target.unlink(missing_ok=True)

    def _target(self, uri: str, *, strict: bool) -> Path:
        target = Path(uri).resolve(strict=strict)
        if self._root != target and self._root not in target.parents:
            raise ValueError("knowledge URI escaped configured directory")
        return target


class S3DocumentStore:
    """Stores versionable Markdown objects in S3 with content hashes and encryption."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region: str,
        *,
        client: Any | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = client or boto3.client("s3", region_name=region)

    def put(self, document: KnowledgeDocument) -> StoredDocument:
        suffix = _document_suffix(document).as_posix()
        versioned_suffix = f"{document.id}/{suffix}"
        versioned_key = self._key(versioned_suffix)
        latest_key = self._key(suffix)
        payload = document.markdown.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        request = {
            "Bucket": self._bucket,
            "Body": payload,
            "ContentType": "text/markdown; charset=utf-8",
            "ServerSideEncryption": "AES256",
            "Metadata": {"sha256": digest, "document-id": document.id},
        }
        self._client.put_object(Key=versioned_key, **request)
        self._client.put_object(Key=latest_key, **request)
        return StoredDocument(
            document_id=document.id,
            uri=f"s3://{self._bucket}/{versioned_key}",
            content_sha256=digest,
            size_bytes=len(payload),
        )

    def publish_manifest(self, manifest: Mapping[str, Any]) -> str:
        """Publish the stable manifest after all referenced resources are durable."""

        payload = json.dumps(
            dict(manifest),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        key = self._key("latest.json")
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=payload,
            ContentType="application/json; charset=utf-8",
            ServerSideEncryption="AES256",
            Metadata={"sha256": digest},
        )
        return f"s3://{self._bucket}/{key}"

    def delete_manifest(self) -> None:
        """Remove only the stable publication pointer during a failed first publication."""

        self._client.delete_object(Bucket=self._bucket, Key=self._key("latest.json"))

    def get(self, uri: str) -> str:
        return self._read_object(self._key_from_uri(uri)).decode("utf-8")

    def latest_manifest(self) -> KnowledgeManifest | None:
        """Read and validate the latest completed S3 portfolio, when one exists."""

        key = self._key("latest.json")
        try:
            payload = self._read_object(key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        manifest = KnowledgeManifest.model_validate_json(payload)
        for uri in _manifest_uris(manifest):
            self._key_from_uri(uri)
        return manifest

    def _key(self, suffix: str) -> str:
        return f"{self._prefix}/{suffix}" if self._prefix else suffix

    def _key_from_uri(self, uri: str) -> str:
        bucket_prefix = f"s3://{self._bucket}/"
        if not uri.startswith(bucket_prefix):
            raise ValueError("S3 URI does not belong to configured bucket")
        key = uri[len(bucket_prefix) :]
        if not key or any(character.isspace() or ord(character) < 32 for character in key):
            raise ValueError("S3 URI contains an invalid object key")
        if self._prefix and not key.startswith(f"{self._prefix}/"):
            raise ValueError("S3 URI does not belong to configured knowledge prefix")
        return key

    def _read_object(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        payload = response["Body"].read()
        if not isinstance(payload, bytes):
            raise RuntimeError("S3 returned a non-byte response body")
        return payload


def _manifest_uris(manifest: KnowledgeManifest) -> tuple[str, ...]:
    evidence = (
        manifest.central_knowledge_uri,
        *(repository.knowledge_uri for repository in manifest.repositories),
    )
    human = tuple(
        uri
        for uri in (
            manifest.human_central_knowledge_uri,
            *(repository.human_knowledge_uri for repository in manifest.repositories),
        )
        if uri is not None
    )
    return (*evidence, *human)


def _document_suffix(document: KnowledgeDocument) -> Path:
    if document.kind in {"repository", "repository-human"}:
        suffix = Path("repositories") / f"{document.repository_id}.md"
    else:
        suffix = Path("central.md")
    return Path("human") / suffix if document.kind.endswith("-human") else suffix
