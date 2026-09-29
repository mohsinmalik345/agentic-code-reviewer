from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ai_code_intelligence.agents.assembler import AgentGraphAssembler, AssemblyResult
from ai_code_intelligence.agents.bedrock import StructuredModelClient
from ai_code_intelligence.agents.content import RepositoryAgentContent, RepositoryContentLoader, SourceBatch
from ai_code_intelligence.agents.contracts import (
    AgentGraphResult,
    CodeGraphBatchResult,
    CodeGraphOutput,
    RepositoryProfile,
)
from ai_code_intelligence.agents.dependency import (
    DependencyGraphAgent,
    dependency_input_fingerprint,
)
from ai_code_intelligence.agents.specialists import (
    CodeGraphAgent,
    CodeGraphAuditAgent,
    RepositoryDiscoveryAgent,
)
from ai_code_intelligence.analysis.incremental import RepositoryIncrementalPlan
from ai_code_intelligence.config import AgentConfig, ScannerConfig
from ai_code_intelligence.domain.models import (
    AgentInvocation,
    GraphNode,
    GraphRelationship,
    KnowledgeGraph,
    ScannedRepository,
)
from ai_code_intelligence.graph.builder import (
    merge_graphs,
    patch_repository_graph,
    retain_unaffected_graph,
)


@dataclass(frozen=True, slots=True)
class _RepositoryWork:
    repository: ScannedRepository
    content: RepositoryAgentContent
    profile: RepositoryProfile
    discovery_invocation: AgentInvocation


class CodeIntelligenceSupervisor:
    """Coordinates specialist agents while deterministic validators retain final authority."""

    def __init__(
        self,
        model: StructuredModelClient,
        agents: AgentConfig,
        scanner: ScannerConfig,
    ) -> None:
        self._model = model
        self._agents = agents
        self._scanner = scanner
        self._logger = logging.getLogger(__name__)

    def analyze(
        self,
        repositories: tuple[ScannedRepository, ...],
        *,
        previous_graph: KnowledgeGraph | None = None,
        changed_repository_ids: frozenset[str] | None = None,
        incremental_plans: Mapping[str, RepositoryIncrementalPlan] | None = None,
        cancellation_check: Callable[[], None] | None = None,
    ) -> AgentGraphResult:
        """Analyze fresh repositories and optionally reuse unaffected prior graph slices."""

        if (previous_graph is None) != (changed_repository_ids is None):
            raise ValueError("incremental analysis requires both previous graph and changed IDs")
        scanned_ids = frozenset(item.definition.id for item in repositories)
        if changed_repository_ids is not None and scanned_ids != changed_repository_ids:
            raise ValueError("incremental analysis must scan exactly the changed repositories")
        if incremental_plans is not None and not set(incremental_plans).issubset(scanned_ids):
            raise ValueError("incremental plans may cover only scanned repositories")
        loader = RepositoryContentLoader(self._agents, self._scanner)
        discovery = RepositoryDiscoveryAgent(self._model)
        checkpoint = cancellation_check or (lambda: None)

        def discover(repository: ScannedRepository) -> _RepositoryWork:
            plan = incremental_plans.get(repository.definition.id) if incremental_plans else None
            included_paths = (
                plan.affected_source_paths if plan is not None and plan.is_selective else None
            )
            content = loader.load(repository, included_paths)
            if included_paths is not None:
                loaded_paths = frozenset(document.file_path for document in content.documents)
                missing_paths = included_paths.difference(loaded_paths)
                if missing_paths:
                    raise RuntimeError(
                        f"incremental analysis could not load {len(missing_paths)} planned source "
                        f"file(s) for {repository.definition.id}: "
                        + ", ".join(sorted(missing_paths)[:10])
                    )
            checkpoint()
            profile, invocation = discovery.discover(repository, content.discovery_context)
            return _RepositoryWork(repository, content, profile, invocation)

        with ThreadPoolExecutor(max_workers=self._agents.max_concurrent_invocations) as executor:
            work = list(executor.map(discover, repositories))

        code_agent = CodeGraphAgent(self._model)
        audit_agent = CodeGraphAuditAgent(self._model)

        graphs: list[KnowledgeGraph] = []
        working_graph = previous_graph
        warnings: list[str] = []
        invocations: list[AgentInvocation] = [item.discovery_invocation for item in work]
        for item in work:
            plan = incremental_plans.get(item.repository.definition.id) if incremental_plans else None
            results = self._extract_repository(
                item,
                code_agent,
                audit_agent,
                cancellation_check=checkpoint,
            )
            assembly = AgentGraphAssembler(self._agents.strict_evidence).assemble(
                item.repository,
                item.content.documents,
                results,
                reference_graph=(
                    _safe_reference_graph(
                        previous_graph,
                        item.repository.definition.id,
                        plan.invalidated_paths,
                    )
                    if previous_graph is not None and plan is not None and plan.is_selective
                    else None
                ),
            )
            self._validate_assembly(item, assembly)
            if working_graph is not None and changed_repository_ids is not None:
                if plan is not None and plan.is_selective:
                    working_graph = patch_repository_graph(
                        working_graph,
                        assembly.graph,
                        item.repository.definition.id,
                        plan.invalidated_paths,
                    )
                else:
                    working_graph = merge_graphs(
                        [
                            retain_unaffected_graph(
                                working_graph,
                                frozenset((item.repository.definition.id,)),
                            ),
                            assembly.graph,
                        ]
                    )
            else:
                graphs.append(assembly.graph)
            warnings.extend(item.content.warnings)
            warnings.extend(assembly.evidence_warnings)
            invocations.extend(result.invocation for result in results)

        merged = working_graph if working_graph is not None else merge_graphs(graphs)
        reuse_dependencies = (
            previous_graph is not None
            and changed_repository_ids is not None
            and dependency_input_fingerprint(previous_graph, changed_repository_ids)
            == dependency_input_fingerprint(merged, changed_repository_ids)
        )
        if reuse_dependencies:
            assert previous_graph is not None and changed_repository_ids is not None
            final_graph = _restore_generated_dependencies(
                previous_graph,
                merged,
                changed_repository_ids,
            )
        else:
            dependency_input = (
                _remove_generated_dependencies(merged, changed_repository_ids)
                if changed_repository_ids is not None
                else merged
            )
            dependencies = DependencyGraphAgent(
                self._model,
                self._agents.max_knowledge_context_characters,
            )
            checkpoint()
            dependency_result = dependencies.link(dependency_input, changed_repository_ids)
            invocations.append(dependency_result.invocation)
            warnings.extend(dependency_result.evidence_warnings)
            final_graph = dependency_result.graph
        final_result = AgentGraphResult(
            graph=final_graph,
            profiles={item.repository.definition.id: item.profile for item in work},
            invocations=tuple(invocations),
            evidence_warnings=tuple(warnings),
        )
        self._logger.info(
            "code intelligence supervisor completed",
            extra={
                "fields": {
                    "repository_count": len(repositories),
                    "reused_repository_count": (
                        len(
                            {
                                node.repository_id
                                for node in previous_graph.nodes
                                if node.repository_id is not None
                                and node.repository_id not in changed_repository_ids
                            }
                        )
                        if previous_graph is not None and changed_repository_ids is not None
                        else 0
                    ),
                    "invocation_count": len(invocations),
                    "warning_count": len(warnings),
                    "dependency_links_reused": reuse_dependencies,
                    "node_count": final_result.graph.statistics.node_count,
                    "relationship_count": final_result.graph.statistics.relationship_count,
                }
            },
        )
        return final_result

    def _extract_repository(
        self,
        item: _RepositoryWork,
        code_agent: CodeGraphAgent,
        audit_agent: CodeGraphAuditAgent,
        *,
        cancellation_check: Callable[[], None],
    ) -> list[CodeGraphBatchResult]:
        def extract(batch: SourceBatch) -> CodeGraphBatchResult:
            cancellation_check()
            return code_agent.analyze(item.repository, item.profile, batch)

        with ThreadPoolExecutor(max_workers=self._agents.max_concurrent_invocations) as executor:
            completed = list(executor.map(extract, item.content.batches))

        for pass_number in range(1, self._agents.audit_passes + 1):
            prior_by_batch: dict[str, tuple[CodeGraphOutput, ...]] = {}
            for result in completed:
                prior_by_batch[result.batch_id] = (
                    *prior_by_batch.get(result.batch_id, ()),
                    result.output,
                )

            def audit(
                batch: SourceBatch,
                *,
                prior: dict[str, tuple[CodeGraphOutput, ...]] = prior_by_batch,
                audit_pass: int = pass_number,
            ) -> CodeGraphBatchResult:
                cancellation_check()
                return audit_agent.audit(
                    item.repository,
                    item.profile,
                    batch,
                    prior.get(batch.id, ()),
                    audit_pass,
                )

            with ThreadPoolExecutor(max_workers=self._agents.max_concurrent_invocations) as executor:
                completed.extend(executor.map(audit, item.content.batches))
        return completed

    def _validate_assembly(self, item: _RepositoryWork, assembly: AssemblyResult) -> None:
        repository_id = item.repository.definition.id
        if item.content.documents and assembly.candidate_fact_count == 0:
            raise RuntimeError(f"Bedrock agents returned no semantic facts for repository {repository_id}")
        rejection_ratio = (
            assembly.rejected_fact_count / assembly.candidate_fact_count
            if assembly.candidate_fact_count
            else 0
        )
        summary = dict(assembly.rejection_reason_counts[:5])
        self._logger.info(
            "Repository evidence assembly completed",
            extra={
                "fields": {
                    "repository_id": repository_id,
                    "candidate_fact_count": assembly.candidate_fact_count,
                    "rejected_fact_count": assembly.rejected_fact_count,
                    "evidence_rejection_ratio": round(rejection_ratio, 6),
                    "top_rejection_categories": summary,
                }
            },
        )
        if rejection_ratio <= self._agents.max_evidence_rejection_ratio:
            return
        category_text = ", ".join(f"{name}={count}" for name, count in summary.items())
        self._logger.error(
            "Repository evidence rejection threshold exceeded",
            extra={
                "fields": {
                    "repository_id": repository_id,
                    "candidate_fact_count": assembly.candidate_fact_count,
                    "rejected_fact_count": assembly.rejected_fact_count,
                    "evidence_rejection_ratio": round(rejection_ratio, 6),
                    "maximum_evidence_rejection_ratio": self._agents.max_evidence_rejection_ratio,
                    "top_rejection_categories": summary,
                }
            },
        )
        raise RuntimeError(
            f"evidence rejection ratio {rejection_ratio:.3f} exceeded configured maximum "
            f"{self._agents.max_evidence_rejection_ratio} for {repository_id}; "
            f"rejected {assembly.rejected_fact_count}/{assembly.candidate_fact_count} facts"
            + (f"; top categories: {category_text}" if category_text else "")
        )


def _restore_generated_dependencies(
    previous: KnowledgeGraph,
    current: KnowledgeGraph,
    affected_repository_ids: frozenset[str],
) -> KnowledgeGraph:
    current_node_ids = {node.id for node in current.nodes}
    current_relationship_ids = {relationship.id for relationship in current.relationships}
    restored = [*current.relationships]
    for relationship in previous.relationships:
        if not _generated_dependency_touches(
            relationship,
            previous,
            affected_repository_ids,
        ):
            continue
        evidence_id = relationship.metadata.get("evidence_relationship_id")
        if (
            relationship.source_id in current_node_ids
            and relationship.target_id in current_node_ids
            and isinstance(evidence_id, str)
            and evidence_id in current_relationship_ids
            and relationship.id not in current_relationship_ids
        ):
            restored.append(relationship)
    return KnowledgeGraph.create(list(current.nodes), restored)


def _safe_reference_graph(
    graph: KnowledgeGraph,
    repository_id: str,
    invalidated_paths: frozenset[str],
) -> KnowledgeGraph:
    """Expose only retained nodes to partial assembly reference resolution."""

    nodes = [
        node
        for node in graph.nodes
        if node.repository_id != repository_id
        or not invalidated_paths.intersection(_node_occurrence_paths(node))
    ]
    return KnowledgeGraph.create(nodes, [])


def _node_occurrence_paths(node: GraphNode) -> frozenset[str]:
    values = {node.file_path} if node.file_path is not None else set()
    raw_occurrences = node.metadata.get("occurrence_file_paths")
    if isinstance(raw_occurrences, list):
        values.update(path for path in raw_occurrences if isinstance(path, str))
    return frozenset(values)


def _remove_generated_dependencies(
    graph: KnowledgeGraph,
    affected_repository_ids: frozenset[str],
) -> KnowledgeGraph:
    relationships = [
        relationship
        for relationship in graph.relationships
        if not _generated_dependency_touches(
            relationship,
            graph,
            affected_repository_ids,
        )
    ]
    return KnowledgeGraph.create(list(graph.nodes), relationships)


def _generated_dependency_touches(
    relationship: GraphRelationship,
    graph: KnowledgeGraph,
    affected_repository_ids: frozenset[str],
) -> bool:
    if relationship.metadata.get("agent_proposed") is not True:
        return False
    nodes = {node.id: node for node in graph.nodes}
    source = nodes.get(relationship.source_id)
    target = nodes.get(relationship.target_id)
    if source is None or target is None or source.repository_id == target.repository_id:
        return False
    return bool(
        {source.repository_id, target.repository_id}.intersection(affected_repository_ids)
    )
