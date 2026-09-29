from __future__ import annotations

import json

from ai_code_intelligence.domain.change import (
    ChangedFile,
    CodeChangeAnalysis,
    DependencyChange,
)
from ai_code_intelligence.domain.models import GraphNode, KnowledgeGraph, NodeType
from ai_code_intelligence.git.diff_parser import UnifiedDiffParser
from ai_code_intelligence.git.local import GitProvider


class GitChangeAnalyzer:
    """Maps line-level Git changes to graph entities; no model participates in this step."""

    def __init__(self, provider: GitProvider, parser: UnifiedDiffParser | None = None) -> None:
        self._provider = provider
        self._parser = parser or UnifiedDiffParser()

    def analyze(
        self,
        repository_id: str,
        repository_path: str,
        graph: KnowledgeGraph,
        *,
        base_revision: str | None = None,
        head_revision: str | None = None,
        staged: bool = False,
    ) -> CodeChangeAnalysis:
        diff = self._provider.diff(
            repository_path,
            base_revision=base_revision,
            head_revision=head_revision,
            staged=staged,
        )
        files = self._parser.parse(diff)
        changed_nodes = tuple(
            node for node in graph.nodes if node.repository_id == repository_id and _node_changed(node, files)
        )
        return CodeChangeAnalysis(
            repository_id=repository_id,
            base_revision=base_revision,
            head_revision=head_revision,
            staged=staged,
            diff=diff,
            changed_files=files,
            changed_functions=_of_types(changed_nodes, {NodeType.FUNCTION, NodeType.METHOD}),
            changed_classes=_of_types(changed_nodes, {NodeType.CLASS, NodeType.INTERFACE, NodeType.ENUM}),
            changed_endpoints=_of_types(changed_nodes, {NodeType.ENDPOINT}),
            changed_apis=_of_types(changed_nodes, {NodeType.API}),
            changed_business_rules=_of_types(changed_nodes, {NodeType.BUSINESS_RULE}),
            changed_dependencies=self._dependency_changes(
                repository_path,
                files,
                base_revision=base_revision,
                head_revision=head_revision,
                staged=staged,
            ),
        )

    def _dependency_changes(
        self,
        repository_path: str,
        files: tuple[ChangedFile, ...],
        *,
        base_revision: str | None,
        head_revision: str | None,
        staged: bool,
    ) -> tuple[DependencyChange, ...]:
        if not any(file.path == "package.json" or file.old_path == "package.json" for file in files):
            return ()
        old_content = self._provider.read_file(
            repository_path,
            "package.json",
            revision=base_revision or "HEAD",
        )
        if staged:
            new_content = self._provider.read_file(repository_path, "package.json", staged=True)
        elif head_revision:
            new_content = self._provider.read_file(
                repository_path,
                "package.json",
                revision=head_revision,
            )
        else:
            new_content = self._provider.read_file(repository_path, "package.json")
        old = _manifest_dependencies(old_content, "base package.json")
        new = _manifest_dependencies(new_content, "head package.json")
        changes: list[DependencyChange] = []
        for name in sorted(old.keys() | new.keys()):
            old_version = old.get(name)
            new_version = new.get(name)
            if old_version == new_version:
                continue
            changes.append(
                DependencyChange(
                    package_name=name,
                    change=(
                        "updated"
                        if old_version is not None and new_version is not None
                        else "removed"
                        if old_version is not None
                        else "added"
                    ),
                    old_version=old_version,
                    new_version=new_version,
                )
            )
        return tuple(changes)


def _node_changed(node: GraphNode, files: tuple[ChangedFile, ...]) -> bool:
    if not node.file_path:
        return False
    for file in files:
        paths = {file.path, *([file.old_path] if file.old_path else [])}
        if node.file_path not in paths:
            continue
        if file.binary or not file.hunks or node.start_line is None:
            return True
        node_end = node.end_line or node.start_line
        for hunk in file.hunks:
            if file.status == "deleted":
                start = hunk.old_start
                end = start + max(hunk.old_lines, 1) - 1
            else:
                start = hunk.new_start
                end = start + max(hunk.new_lines, 1) - 1
            if node.start_line <= end and node_end >= start:
                return True
    return False


def _of_types(nodes: tuple[GraphNode, ...], types: set[NodeType]) -> tuple[GraphNode, ...]:
    return tuple(node for node in nodes if node.type in types)


def _manifest_dependencies(content: str | None, label: str) -> dict[str, str]:
    if content is None:
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} root is not an object")
    values: dict[str, str] = {}
    for section in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
        dependencies = parsed.get(section)
        if dependencies is None:
            continue
        if not isinstance(dependencies, dict):
            raise RuntimeError(f"{label} field {section} is not an object")
        for name, version in dependencies.items():
            if isinstance(name, str) and isinstance(version, str):
                values[name] = version
    return values
