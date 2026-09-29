from __future__ import annotations

import hashlib


def stable_id(namespace: str, *parts: object) -> str:
    """Return a readable, deterministic identifier for graph entities."""

    material = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"{namespace}:{digest}"
