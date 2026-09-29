from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from ai_code_intelligence.agents.contracts import DeploymentRiskOutput
from ai_code_intelligence.domain.analysis import DeploymentReport, ImpactAnalysis
from ai_code_intelligence.domain.change import CodeChangeAnalysis
from ai_code_intelligence.utils.ids import stable_id
from ai_code_intelligence.utils.redaction import redact_json_strings


class DeploymentReportGenerator:
    """Combines deterministic risks and validated AI reasoning under a fail-safe policy."""

    def create(
        self,
        change: CodeChangeAnalysis,
        impact: ImpactAnalysis,
        assessment: DeploymentRiskOutput,
    ) -> DeploymentReport:
        severities = {signal.severity for signal in impact.deterministic_risk_signals}
        floor = (
            100
            if "critical" in severities
            else 70
            if "high" in severities
            else 40
            if "medium" in severities
            else 0
        )
        risk_score = max(floor, assessment.risk_score)
        recommendation = (
            "BLOCK"
            if (
                assessment.recommendation == "BLOCK"
                or not assessment.implementation_satisfies_intent
                or risk_score >= 70
                or impact.traversal_truncated
            )
            else "PASS"
        )
        reasoning = (
            *assessment.reasoning,
            *(
                f"Deterministic {signal.severity} signal {signal.code}: {signal.reason}"
                for signal in impact.deterministic_risk_signals
            ),
        )
        return DeploymentReport(
            id=stable_id(
                "deployment-report",
                change.repository_id,
                change.base_revision,
                change.head_revision,
                change.diff,
            ),
            recommendation=recommendation,
            risk_score=risk_score,
            implementation_satisfies_intent=assessment.implementation_satisfies_intent,
            breakage_risk=assessment.breakage_risk,
            hidden_side_effects=assessment.hidden_side_effects,
            regression_test_node_ids=assessment.regression_test_node_ids,
            reasoning=reasoning,
            assumptions=assessment.assumptions,
            change=change,
            impact=impact,
        )


class LocalReportStore:
    """Writes Markdown and JSON reports atomically to the configured artifact directory."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory).resolve()

    def put(self, report: DeploymentReport) -> tuple[str, str]:
        self._directory.mkdir(parents=True, exist_ok=True)
        file_stem = re.sub(r"[^A-Za-z0-9._-]", "-", report.id)
        markdown_path = self._directory / f"{file_stem}.md"
        json_path = self._directory / f"{file_stem}.json"
        _atomic_write(markdown_path, render_markdown(report))
        _atomic_write(
            json_path,
            json.dumps(redact_json_strings(report.model_dump(mode="json")), indent=2),
        )
        return str(markdown_path), str(json_path)


def render_markdown(report: DeploymentReport) -> str:
    """Render the human deployment gate while retaining graph identifiers for audit."""

    change = report.change
    impact = report.impact
    return "\n".join(
        [
            "# Deployment Analysis",
            "",
            f"**Recommendation:** {report.recommendation}",
            f"**Risk Score:** {report.risk_score}/100",
            f"**Implementation satisfies intent:** {'Yes' if report.implementation_satisfies_intent else 'No'}",
            f"**Breakage risk:** {report.breakage_risk}",
            "",
            "## Changed Services",
            "",
            f"- {change.repository_id}",
            "",
            "## Changed Files",
            "",
            *_items(file.path for file in change.changed_files),
            "",
            "## Changed Functions",
            "",
            *_items(f"{node.qualified_name} [{node.id}]" for node in change.changed_functions),
            "",
            "## Affected Services",
            "",
            *_items(f"{node.name} [{node.id}]" for node in impact.affected_services),
            "",
            "## Affected APIs",
            "",
            *_items(
                f"{node.name} [{node.id}]"
                for node in (*impact.affected_endpoints, *impact.affected_rest_calls)
            ),
            "",
            "## Dependency Chain",
            "",
            *_items(" -> ".join(path.node_ids) for path in impact.dependency_paths),
            "",
            "## Business Logic Risks",
            "",
            *_items(signal.reason for signal in impact.deterministic_risk_signals),
            "",
            "## Regression Risks",
            "",
            *_items(impact.potential_regressions),
            "",
            "## Hidden Side Effects",
            "",
            *_items(report.hidden_side_effects),
            "",
            "## Functions to Regression Test",
            "",
            *_items(report.regression_test_node_ids),
            "",
            "## Detailed Reasoning",
            "",
            *_items(report.reasoning),
            "",
            "## Assumptions",
            "",
            *_items(report.assumptions),
            "",
            "## Deployment Recommendation",
            "",
            report.recommendation,
            "",
        ]
    )


def _items(values: Iterable[object]) -> list[str]:
    materialized = list(values)
    return [f"- {value}" for value in materialized] if materialized else ["- None evidenced"]


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
