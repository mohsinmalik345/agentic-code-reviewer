from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from ai_code_intelligence.domain.change import ChangedFile, DiffHunk

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(slots=True)
class _MutableFile:
    path: str
    old_path: str | None = None
    status: str = "modified"
    additions: int = 0
    deletions: int = 0
    binary: bool = False
    hunks: list[DiffHunk] = field(default_factory=list)


class UnifiedDiffParser:
    """Parses Git unified diff metadata and hunks without interpreting source semantics."""

    def parse(self, value: str) -> tuple[ChangedFile, ...]:
        files: list[ChangedFile] = []
        current: _MutableFile | None = None
        hunk_header: str | None = None
        hunk_lines: list[str] = []

        def finish_hunk() -> None:
            nonlocal hunk_header, hunk_lines
            if current is None or hunk_header is None:
                return
            match = _HUNK.match(hunk_header)
            if match is None:
                raise ValueError(f"invalid unified diff hunk header: {hunk_header}")
            current.hunks.append(
                DiffHunk(
                    old_start=int(match.group(1)),
                    old_lines=int(match.group(2) or "1"),
                    new_start=int(match.group(3)),
                    new_lines=int(match.group(4) or "1"),
                    header=hunk_header,
                    lines=tuple(hunk_lines),
                )
            )
            hunk_header = None
            hunk_lines = []

        def finish_file() -> None:
            nonlocal current
            finish_hunk()
            if current is None:
                return
            files.append(
                ChangedFile(
                    path=current.path,
                    old_path=current.old_path,
                    status=current.status,
                    additions=current.additions,
                    deletions=current.deletions,
                    binary=current.binary,
                    hunks=tuple(current.hunks),
                )
            )
            current = None

        for line in value.splitlines():
            if line.startswith("diff --git "):
                finish_file()
                paths = _diff_paths(line)
                if paths is None:
                    raise ValueError(f"invalid Git diff file header: {line}")
                current = _MutableFile(path=paths[1], old_path=paths[0])
                continue
            if current is None:
                continue
            if line.startswith("new file mode "):
                current.status = "added"
                current.old_path = None
            elif line.startswith("deleted file mode "):
                current.status = "deleted"
            elif line.startswith("rename from "):
                current.status = "renamed"
                current.old_path = _normalize_path(line[len("rename from ") :])
            elif line.startswith("rename to "):
                current.status = "renamed"
                current.path = _normalize_path(line[len("rename to ") :])
            elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
                current.binary = True
            elif line.startswith("@@ "):
                finish_hunk()
                hunk_header = line
            elif hunk_header is not None:
                hunk_lines.append(line)
                if line.startswith("+") and not line.startswith("+++"):
                    current.additions += 1
                elif line.startswith("-") and not line.startswith("---"):
                    current.deletions += 1
        finish_file()
        return tuple(files)


def _diff_paths(line: str) -> tuple[str, str] | None:
    value = line[len("diff --git ") :]
    if value.startswith('"'):
        parts = shlex.split(value, posix=True)
        if len(parts) == 2 and parts[0].startswith("a/") and parts[1].startswith("b/"):
            return _normalize_path(parts[0][2:]), _normalize_path(parts[1][2:])
    separator = value.rfind(" b/")
    if not value.startswith("a/") or separator < 0:
        return None
    return _normalize_path(value[2:separator]), _normalize_path(value[separator + 3 :])


def _normalize_path(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
        value = bytes(value, "utf-8").decode("unicode_escape")
    return value.replace("\\", "/")
