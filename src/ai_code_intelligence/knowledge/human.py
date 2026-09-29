from __future__ import annotations

import re

from ai_code_intelligence.domain.knowledge import KnowledgeDocument
from ai_code_intelligence.utils.ids import stable_id

_GRAPH_CITATION = re.compile(r"\[(?:node|relationship):[^\]\r\n]+\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([,.;:])")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_READER_NOTE = (
    "> Reader edition: internal graph identifiers are hidden for clarity. "
    "The evidence edition remains the traceable source of truth."
)


def human_readable_document(document: KnowledgeDocument) -> KnowledgeDocument:
    """Create a deterministic reader edition without adding or rewriting any claim."""

    if document.kind not in {"repository", "central"}:
        raise ValueError("only evidence knowledge documents can produce a reader edition")
    markdown = strip_graph_citations(document.markdown)
    lines = markdown.splitlines()
    heading_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("# ")),
        None,
    )
    if heading_index is None:
        raise ValueError("knowledge document must contain a level-one heading")
    lines[heading_index + 1 : heading_index + 1] = ["", _READER_NOTE]
    reader_markdown = _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip() + "\n"
    kind = "repository-human" if document.kind == "repository" else "central-human"
    return KnowledgeDocument(
        id=stable_id("knowledge", "human", document.id, reader_markdown),
        kind=kind,
        title=document.title,
        markdown=reader_markdown,
        repository_id=document.repository_id,
        evidence_node_ids=document.evidence_node_ids,
        evidence_relationship_ids=document.evidence_relationship_ids,
        generated_at=document.generated_at,
    )


def strip_graph_citations(markdown: str) -> str:
    """Remove only internal graph citation markers while preserving document wording."""

    values: list[str] = []
    in_fence = False
    for raw_line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line
        if line.lstrip().startswith(chr(96) * 3):
            in_fence = not in_fence
        if not in_fence:
            line = _GRAPH_CITATION.sub("", line)
            line = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", line)
            line = line.rstrip()
        values.append(line)
    return "\n".join(values)
