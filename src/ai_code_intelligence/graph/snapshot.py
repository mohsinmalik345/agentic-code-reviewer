from __future__ import annotations

from pathlib import Path

from ai_code_intelligence.domain.models import KnowledgeGraph


class LocalGraphSnapshotStore:
    """Persists the latest validated graph atomically for local analysis and GraphQL reads."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()

    @property
    def path(self) -> Path:
        return self._path

    def put(self, graph: KnowledgeGraph) -> str:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(graph.model_dump_json(indent=2), encoding="utf-8", newline="\n")
        temporary.replace(self._path)
        return str(self._path)

    def get(self) -> KnowledgeGraph:
        return KnowledgeGraph.model_validate_json(self._path.read_text(encoding="utf-8"))

    def exists(self) -> bool:
        return self._path.exists()

    def delete(self) -> None:
        """Remove only this managed snapshot during a failed first publication."""

        self._path.unlink(missing_ok=True)
