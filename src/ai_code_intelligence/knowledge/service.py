from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping

from ai_code_intelligence.agents.contracts import AgentGraphResult, RepositoryProfile
from ai_code_intelligence.analysis.incremental import RepositoryIncrementalPlan
from ai_code_intelligence.domain.knowledge import (
    KnowledgeDocument,
    KnowledgeManifest,
    StoredDocument,
)
from ai_code_intelligence.domain.models import (
    AgentInvocation,
    KnowledgeGraph,
    RepositoryDefinition,
    ScannedRepository,
)
from ai_code_intelligence.knowledge.agents import KnowledgeAgents
from ai_code_intelligence.knowledge.human import human_readable_document
from ai_code_intelligence.knowledge.projection import central_repository_markdown
from ai_code_intelligence.knowledge.store import DocumentStore
from ai_code_intelligence.utils.ids import stable_id

_CITATION = re.compile(r"\[(node|relationship):([^\]\r\n]+)\]")


class KnowledgeResult:
    """Immutable aggregate returned after all documents are validated and stored."""

    def __init__(
        self,
        repository_documents: tuple[KnowledgeDocument, ...],
        central_document: KnowledgeDocument,
        stored_documents: tuple[StoredDocument, ...],
        human_repository_documents: tuple[KnowledgeDocument, ...],
        human_central_document: KnowledgeDocument,
        human_stored_documents: tuple[StoredDocument, ...],
        invocations: tuple[AgentInvocation, ...],
        updated_repository_ids: tuple[str, ...] = (),
        central_updated: bool = True,
    ) -> None:
        self.repository_documents = repository_documents
        self.central_document = central_document
        self.stored_documents = stored_documents
        self.human_repository_documents = human_repository_documents
        self.human_central_document = human_central_document
        self.human_stored_documents = human_stored_documents
        self.invocations = invocations
        self.updated_repository_ids = updated_repository_ids
        self.central_updated = central_updated


class KnowledgeService:
    """Orchestrates specialist generation and storage without embedding transport logic."""

    def __init__(self, agents: KnowledgeAgents, store: DocumentStore) -> None:
        self._agents = agents
        self._store = store

    def generate(
        self,
        repositories: tuple[ScannedRepository, ...],
        graph_result: AgentGraphResult,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> KnowledgeResult:
        documents: list[KnowledgeDocument] = []
        invocations: list[AgentInvocation] = []
        for repository in repositories:
            profile = graph_result.profiles[repository.definition.id]
            document, document_invocations = self._agents.repository_document(
                repository,
                profile,
                graph_result.graph,
                cancellation_check=cancellation_check,
            )
            documents.append(document)
            invocations.extend(document_invocations)
        central, central_invocations = self._agents.central_document(
            tuple(documents),
            graph_result.graph,
            cancellation_check=cancellation_check,
        )
        invocations.extend(central_invocations)
        all_documents = (*documents, central)
        stored = tuple(self._store.put(document) for document in all_documents)
        human_documents = tuple(human_readable_document(document) for document in all_documents)
        human_stored = tuple(self._store.put(document) for document in human_documents)
        return KnowledgeResult(
            tuple(documents),
            central,
            stored,
            human_documents[:-1],
            human_documents[-1],
            human_stored,
            tuple(invocations),
            tuple(repository.definition.id for repository in repositories),
        )

    def latest_manifest(self) -> KnowledgeManifest | None:
        """Return the latest atomic publication pointer from the configured store."""

        return self._store.latest_manifest()

    def documents_are_readable(self, uris: tuple[str, ...]) -> bool:
        """Verify that every immutable manifest document exists and contains Markdown."""

        if not uris:
            return False
        try:
            return all(bool(self._store.get(uri).strip()) for uri in uris)
        except Exception:
            return False

    def generate_incremental(
        self,
        repositories: tuple[ScannedRepository, ...],
        graph_result: AgentGraphResult,
        repository_ids: tuple[str, ...],
        previous_manifest: KnowledgeManifest,
        *,
        incremental_plans: Mapping[str, RepositoryIncrementalPlan] | None = None,
        regenerate_central: bool = True,
        cancellation_check: Callable[[], None] | None = None,
    ) -> KnowledgeResult:
        """Generate changed repository documents while reusing published unchanged ones."""

        changed = {repository.definition.id: repository for repository in repositories}
        if not changed or not set(changed).issubset(repository_ids):
            raise ValueError("incremental knowledge requires changed repositories in portfolio order")
        manifest_by_id = {item.id: item for item in previous_manifest.repositories}
        missing_reused_ids = set(repository_ids).difference(changed, manifest_by_id)
        if missing_reused_ids:
            raise RuntimeError(
                "published knowledge manifest is missing reusable repositories: "
                + ", ".join(sorted(missing_reused_ids))
            )
        if previous_manifest.schema_version != 2 or any(
            item.human_knowledge_uri is None for item in previous_manifest.repositories
        ):
            raise RuntimeError("incremental knowledge reuse requires a complete reader-edition manifest")

        documents: list[KnowledgeDocument] = []
        stored_by_repository: dict[str, StoredDocument] = {}
        human_by_repository: dict[str, KnowledgeDocument] = {}
        human_stored_by_repository: dict[str, StoredDocument] = {}
        updated_repository_ids: set[str] = set()
        invocations: list[AgentInvocation] = []
        for repository_id in repository_ids:
            repository = changed.get(repository_id)
            if repository is not None:
                profile = graph_result.profiles[repository_id]
                plan = incremental_plans.get(repository_id) if incremental_plans else None
                if plan is not None and plan.is_selective and repository_id in manifest_by_id:
                    prior = _load_reused_document(
                        self._store,
                        manifest_by_id[repository_id].knowledge_uri,
                        repository_id,
                        "repository",
                        previous_manifest,
                        graph_result.graph,
                        require_current_citations=False,
                    )
                    patch, document_invocations = self._agents.repository_sections_document(
                        repository,
                        profile,
                        graph_result.graph,
                        plan.knowledge_sections,
                        plan.directly_affected_node_ids | plan.dependency_node_ids,
                        plan.invalidated_paths,
                        cancellation_check=cancellation_check,
                    )
                    try:
                        document = _merge_repository_sections(
                            prior,
                            patch,
                            plan.knowledge_sections,
                            graph_result.graph,
                        )
                    except (RuntimeError, ValueError):
                        document, fallback_invocations = self._agents.repository_document(
                            repository,
                            profile,
                            graph_result.graph,
                            cancellation_check=cancellation_check,
                        )
                        document_invocations = (*document_invocations, *fallback_invocations)
                else:
                    document, document_invocations = self._agents.repository_document(
                        repository,
                        profile,
                        graph_result.graph,
                        cancellation_check=cancellation_check,
                    )
                stored_by_repository[repository_id] = self._store.put(document)
                human = human_readable_document(document)
                human_by_repository[repository_id] = human
                human_stored_by_repository[repository_id] = self._store.put(human)
                invocations.extend(document_invocations)
                updated_repository_ids.add(repository_id)
            else:
                manifest_entry = manifest_by_id[repository_id]
                prior = _load_reused_document(
                    self._store,
                    manifest_entry.knowledge_uri,
                    repository_id,
                    "repository",
                    previous_manifest,
                    graph_result.graph,
                    require_current_citations=False,
                )
                stale_sections = _stale_citation_sections(prior.markdown, graph_result.graph)
                if stale_sections:
                    synthetic = ScannedRepository(
                        definition=RepositoryDefinition(
                            id=repository_id,
                            name=prior.title,
                            path=f"/published/{repository_id}",
                        ),
                        root_path=f"/published/{repository_id}",
                        directories=(),
                        files=(),
                    )
                    profile = RepositoryProfile(
                        purpose="Published repository knowledge refresh",
                        architecture="Validated knowledge graph",
                        runtime="Not evidenced in this knowledge-only refresh",
                    )
                    patch, patch_invocations = self._agents.repository_sections_document(
                        synthetic,
                        profile,
                        graph_result.graph,
                        stale_sections,
                        frozenset(),
                        frozenset(),
                        cancellation_check=cancellation_check,
                    )
                    document = _merge_repository_sections(
                        prior,
                        patch,
                        stale_sections,
                        graph_result.graph,
                    )
                    stored_by_repository[repository_id] = self._store.put(document)
                    human = human_readable_document(document)
                    human_by_repository[repository_id] = human
                    human_stored_by_repository[repository_id] = self._store.put(human)
                    invocations.extend(patch_invocations)
                    updated_repository_ids.add(repository_id)
                else:
                    document = prior
                    stored_by_repository[repository_id] = _stored(
                        document,
                        manifest_entry.knowledge_uri,
                    )
                    assert manifest_entry.human_knowledge_uri is not None
                    human = _load_reused_document(
                        self._store,
                        manifest_entry.human_knowledge_uri,
                        repository_id,
                        "repository-human",
                        previous_manifest,
                        graph_result.graph,
                        evidence_document_id=document.id,
                    )
                    human_by_repository[repository_id] = human
                    human_stored_by_repository[repository_id] = _stored(
                        human,
                        manifest_entry.human_knowledge_uri,
                    )
            documents.append(document)

        if not regenerate_central:
            regenerate_central = _central_repository_inputs_changed(
                self._store,
                tuple(documents),
                previous_manifest,
            )
        central_updated = regenerate_central
        if regenerate_central:
            central, central_invocations = self._agents.central_document(
                tuple(documents),
                graph_result.graph,
                cancellation_check=cancellation_check,
            )
            invocations.extend(central_invocations)
            central_stored = self._store.put(central)
            human_central = human_readable_document(central)
            human_central_stored = self._store.put(human_central)
        else:
            try:
                central = _load_reused_central_document(
                    self._store,
                    previous_manifest.central_knowledge_uri,
                    "central",
                    previous_manifest,
                    graph_result.graph,
                )
                if previous_manifest.human_central_knowledge_uri is None:
                    raise RuntimeError("published manifest has no reader-edition central document")
                human_central = _load_reused_central_document(
                    self._store,
                    previous_manifest.human_central_knowledge_uri,
                    "central-human",
                    previous_manifest,
                    graph_result.graph,
                    evidence_document_id=central.id,
                )
                central_stored = _stored(central, previous_manifest.central_knowledge_uri)
                human_central_stored = _stored(
                    human_central,
                    previous_manifest.human_central_knowledge_uri,
                )
            except (RuntimeError, ValueError):
                central_updated = True
                central, central_invocations = self._agents.central_document(
                    tuple(documents),
                    graph_result.graph,
                    cancellation_check=cancellation_check,
                )
                invocations.extend(central_invocations)
                central_stored = self._store.put(central)
                human_central = human_readable_document(central)
                human_central_stored = self._store.put(human_central)
        return KnowledgeResult(
            tuple(documents),
            central,
            (
                *(stored_by_repository[repository_id] for repository_id in repository_ids),
                central_stored,
            ),
            tuple(human_by_repository[repository_id] for repository_id in repository_ids),
            human_central,
            (
                *(human_stored_by_repository[repository_id] for repository_id in repository_ids),
                human_central_stored,
            ),
            tuple(invocations),
            tuple(sorted(updated_repository_ids)),
            central_updated,
        )


def _load_reused_document(
    store: DocumentStore,
    uri: str,
    repository_id: str,
    kind: str,
    manifest: KnowledgeManifest,
    graph: KnowledgeGraph,
    *,
    evidence_document_id: str | None = None,
    require_current_citations: bool = True,
) -> KnowledgeDocument:
    markdown = store.get(uri)
    title = next(
        (
            line.removeprefix("# ").strip()
            for line in markdown.splitlines()
            if line.startswith("# ") and line.removeprefix("# ").strip()
        ),
        repository_id,
    )
    graph_node_ids = {node.id for node in graph.nodes}
    graph_relationship_ids = {relationship.id for relationship in graph.relationships}
    citations = tuple(_CITATION.findall(markdown))
    raw_node_ids = {
        identifier.strip() for namespace, identifier in citations if namespace == "node"
    }
    raw_relationship_ids = {
        identifier.strip() for namespace, identifier in citations if namespace == "relationship"
    }
    if require_current_citations and (
        not raw_node_ids.issubset(graph_node_ids)
        or not raw_relationship_ids.issubset(graph_relationship_ids)
    ):
        raise RuntimeError("published repository knowledge contains stale graph citations")
    evidence_node_ids = tuple(
        sorted(raw_node_ids & graph_node_ids)
    )
    evidence_relationship_ids = tuple(
        sorted(
            raw_relationship_ids & graph_relationship_ids
        )
    )
    if kind == "repository":
        original_document_id = stable_id("knowledge", "repository", repository_id, markdown)
        document_id = stable_id(
            "knowledge",
            "reused",
            original_document_id,
            graph.generated_at.isoformat(),
        )
    elif kind == "repository-human" and evidence_document_id is not None:
        document_id = stable_id("knowledge", "human", evidence_document_id, markdown)
    else:
        raise ValueError(f"unsupported reused knowledge kind: {kind}")
    return KnowledgeDocument(
        id=document_id,
        kind=kind,
        title=title,
        markdown=markdown,
        repository_id=repository_id,
        evidence_node_ids=evidence_node_ids,
        evidence_relationship_ids=evidence_relationship_ids,
        generated_at=manifest.generated_at,
    )


def _stored(document: KnowledgeDocument, uri: str) -> StoredDocument:
    payload = document.markdown.encode("utf-8")
    return StoredDocument(
        document_id=document.id,
        uri=uri,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _load_reused_central_document(
    store: DocumentStore,
    uri: str,
    kind: str,
    manifest: KnowledgeManifest,
    graph: KnowledgeGraph,
    *,
    evidence_document_id: str | None = None,
) -> KnowledgeDocument:
    markdown = store.get(uri)
    title = next(
        (
            line.removeprefix("# ").strip()
            for line in markdown.splitlines()
            if line.startswith("# ") and line.removeprefix("# ").strip()
        ),
        "Central Knowledge Base",
    )
    node_ids = {node.id for node in graph.nodes}
    relationship_ids = {relationship.id for relationship in graph.relationships}
    citations = tuple(_CITATION.findall(markdown))
    cited_nodes = {identifier.strip() for namespace, identifier in citations if namespace == "node"}
    cited_relationships = {
        identifier.strip() for namespace, identifier in citations if namespace == "relationship"
    }
    if not cited_nodes.issubset(node_ids) or not cited_relationships.issubset(relationship_ids):
        raise RuntimeError("published central knowledge contains stale graph citations")
    if kind == "central":
        document_id = stable_id("knowledge", "reused-central", markdown, graph.generated_at.isoformat())
        repository_id = None
    elif kind == "central-human" and evidence_document_id is not None:
        document_id = stable_id("knowledge", "human", evidence_document_id, markdown)
        repository_id = None
    else:
        raise ValueError(f"unsupported reused central knowledge kind: {kind}")
    return KnowledgeDocument(
        id=document_id,
        kind=kind,
        title=title,
        markdown=markdown,
        repository_id=repository_id,
        evidence_node_ids=tuple(sorted(cited_nodes)),
        evidence_relationship_ids=tuple(sorted(cited_relationships)),
        generated_at=manifest.generated_at,
    )


def _merge_repository_sections(
    previous: KnowledgeDocument,
    patch: KnowledgeDocument,
    selected_sections: frozenset[str],
    graph: KnowledgeGraph,
) -> KnowledgeDocument:
    previous_sections = _split_level_two_sections(previous.markdown)
    patch_sections = _split_level_two_sections(patch.markdown)
    missing_patch = selected_sections.difference(patch_sections)
    if missing_patch:
        raise RuntimeError(
            "repository section update omitted: " + ", ".join(sorted(missing_patch))
        )
    merged_sections = {**previous_sections}
    for section in selected_sections:
        merged_sections[section] = patch_sections[section]
    missing = set(_REPOSITORY_SECTION_ORDER).difference(merged_sections)
    if missing:
        raise RuntimeError(
            "published repository knowledge is missing sections: " + ", ".join(sorted(missing))
        )
    title = previous.title.strip() or patch.title.strip()
    markdown = f"# {title}\n\n" + "\n\n".join(
        f"## {section}\n{merged_sections[section].strip()}"
        for section in _REPOSITORY_SECTION_ORDER
    )
    markdown = markdown.rstrip() + "\n"
    citations = tuple(_CITATION.findall(markdown))
    evidence_node_ids = {
        identifier.strip() for namespace, identifier in citations if namespace == "node"
    }
    evidence_relationship_ids = {
        identifier.strip()
        for namespace, identifier in citations
        if namespace == "relationship"
    }
    graph_nodes = {node.id for node in graph.nodes}
    graph_relationships = {relationship.id for relationship in graph.relationships}
    if not evidence_node_ids.issubset(graph_nodes) or not evidence_relationship_ids.issubset(
        graph_relationships
    ):
        raise RuntimeError("retained knowledge section contains stale graph citations")
    return KnowledgeDocument(
        id=stable_id("knowledge", "repository", previous.repository_id, markdown),
        kind="repository",
        title=title,
        markdown=markdown,
        repository_id=previous.repository_id,
        evidence_node_ids=tuple(sorted(evidence_node_ids)),
        evidence_relationship_ids=tuple(sorted(evidence_relationship_ids)),
    )


def _split_level_two_sections(markdown: str) -> dict[str, str]:
    matches = tuple(re.finditer(r"(?m)^##[ \t]+([^\r\n#]+?)[ \t]*(?:#+[ \t]*)?$", markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = " ".join(match.group(1).split())
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        if name in sections:
            raise ValueError(f"duplicate level-two knowledge section: {name}")
        sections[name] = markdown[start:end].strip()
    return sections


def _stale_citation_sections(
    markdown: str,
    graph: KnowledgeGraph,
) -> frozenset[str]:
    graph_node_ids = {node.id for node in graph.nodes}
    graph_relationship_ids = {relationship.id for relationship in graph.relationships}

    def invalid_citations(value: str) -> set[tuple[str, str]]:
        return {
            (namespace, identifier.strip())
            for namespace, identifier in _CITATION.findall(value)
            if (
                namespace == "node" and identifier.strip() not in graph_node_ids
            )
            or (
                namespace == "relationship"
                and identifier.strip() not in graph_relationship_ids
            )
        }

    invalid = invalid_citations(markdown)
    if not invalid:
        return frozenset()
    sections = _split_level_two_sections(markdown)
    selected = {
        section
        for section, content in sections.items()
        if invalid_citations(content)
    }
    if not selected or not selected.issubset(_REPOSITORY_SECTION_ORDER):
        return frozenset(_REPOSITORY_SECTION_ORDER)
    return frozenset(selected)


_REPOSITORY_SECTION_ORDER = (
    "Purpose",
    "Architecture",
    "Folder Structure",
    "Controllers",
    "Services",
    "Business Flow",
    "API List",
    "Environment Variables",
    "Database Usage",
    "External Dependencies",
    "Socket Events",
    "Internal Dependency Tree",
    "Important Functions",
    "Main Business Logic",
    "Known Risks",
    "Code Statistics",
)


def _central_repository_inputs_changed(
    store: DocumentStore,
    documents: tuple[KnowledgeDocument, ...],
    manifest: KnowledgeManifest,
) -> bool:
    previous_by_id = {item.id: item for item in manifest.repositories}
    current_ids = {document.repository_id for document in documents}
    if None in current_ids or current_ids != set(previous_by_id):
        return True
    try:
        for document in documents:
            assert document.repository_id is not None
            previous_markdown = store.get(
                previous_by_id[document.repository_id].knowledge_uri
            )
            if central_repository_markdown(previous_markdown) != central_repository_markdown(
                document.markdown
            ):
                return True
    except Exception:
        return True
    return False
