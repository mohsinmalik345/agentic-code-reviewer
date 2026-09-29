from __future__ import annotations

import re

_CENTRAL_REPOSITORY_SECTIONS = (
    "Purpose",
    "Architecture",
)
_LEVEL_TWO = re.compile(r"(?m)^##[ \t]+([^\r\n#]+?)[ \t]*(?:#+[ \t]*)?$")
_CITATION = re.compile(r"\[(node|relationship):([^\]\r\n]+)\]")


def central_repository_markdown(markdown: str) -> str:
    """Project repository knowledge to sections actually consumed by central generation."""

    title = next(
        (
            line.strip()
            for line in markdown.splitlines()
            if line.startswith("# ") and line.removeprefix("# ").strip()
        ),
        "# Repository",
    )
    matches = tuple(_LEVEL_TWO.finditer(markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = " ".join(match.group(1).split())
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[name] = markdown[start:end].strip()
    selected = [
        f"## {name}\n{sections[name]}"
        for name in _CENTRAL_REPOSITORY_SECTIONS
        if name in sections
    ]
    return "\n\n".join((title, *selected)).rstrip() + "\n"


def central_repository_citations(
    markdown: str,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return graph IDs cited by the central repository projection."""

    citations = tuple(_CITATION.findall(central_repository_markdown(markdown)))
    return (
        frozenset(identifier.strip() for kind, identifier in citations if kind == "node"),
        frozenset(
            identifier.strip() for kind, identifier in citations if kind == "relationship"
        ),
    )
