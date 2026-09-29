from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from ai_code_intelligence.api.server import create_app
from ai_code_intelligence.api.service import PlatformService
from ai_code_intelligence.application.container import ApplicationContainer
from ai_code_intelligence.application.onboarding import GitLabSourceMigration
from ai_code_intelligence.config import load_config, required_secret
from ai_code_intelligence.domain.models import AGENT_DESCRIPTORS
from ai_code_intelligence.jira.store import PostgresJiraConfigurationStore
from ai_code_intelligence.logging import configure_logging
from ai_code_intelligence.persistence.postgres import PostgresMetadataStore

app = typer.Typer(
    name="code-intel",
    help="Bedrock-powered, evidence-grounded deployment intelligence.",
    no_args_is_help=True,
)
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="YAML configuration path.")]


@app.command("validate-config")
def validate_config(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """Validate configuration and repository paths without calling AWS."""

    loaded = load_config(config)
    missing = [item.path for item in loaded.repositories if not Path(item.path).is_dir()]
    if missing:
        raise typer.BadParameter(f"repository paths do not exist: {missing}")
    typer.echo(f"Configuration is valid: {loaded.config_path}")


@app.command()
def agents() -> None:
    """List the specialist agent team."""

    typer.echo(
        json.dumps(
            [item.model_dump(mode="json") for item in AGENT_DESCRIPTORS],
            indent=2,
        )
    )


@app.command()
def scan(
    config: ConfigOption = _DEFAULT_CONFIG,
    persist_neo4j: Annotated[bool, typer.Option("--persist-neo4j")] = False,
    index_embeddings: Annotated[bool, typer.Option("--index-embeddings")] = False,
    persist_metadata: Annotated[bool, typer.Option("--persist-metadata")] = False,
) -> None:
    """Scan configured repositories and build graph plus knowledge bases."""

    configure_logging()
    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        definitions = tuple(
            item.to_repository_definition()
            for item in container.repository_store.list_repositories(include_disabled=False)
        )
        artifacts = container.indexing.run(
            repositories=definitions,
            persist_neo4j=persist_neo4j,
            index_embeddings=index_embeddings,
            persist_metadata=persist_metadata,
        )
        typer.echo(artifacts.result.model_dump_json(indent=2))
    finally:
        container.close()


@app.command()
def analyze(
    repository_id: Annotated[str, typer.Argument(help="Configured repository ID.")],
    config: ConfigOption = _DEFAULT_CONFIG,
    base_revision: Annotated[str | None, typer.Option("--base")] = None,
    head_revision: Annotated[str | None, typer.Option("--head")] = None,
    staged: Annotated[bool, typer.Option("--staged")] = False,
    intent: Annotated[str | None, typer.Option("--intent")] = None,
    use_neo4j_impact: Annotated[bool, typer.Option("--use-neo4j-impact")] = False,
    persist_metadata: Annotated[bool, typer.Option("--persist-metadata")] = False,
) -> None:
    """Refresh intelligence, analyze a Git change, and emit PASS/BLOCK reports."""

    configure_logging()
    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        definitions = tuple(
            item.to_repository_definition()
            for item in container.repository_store.list_repositories(include_disabled=False)
        )
        artifacts = container.indexing.run(
            repositories=definitions,
            persist_neo4j=use_neo4j_impact,
            persist_metadata=persist_metadata,
        )
        result = container.deployment.run(
            artifacts,
            repository_id,
            base_revision=base_revision,
            head_revision=head_revision,
            staged=staged,
            intent=intent,
            use_neo4j_impact=use_neo4j_impact,
            persist_metadata=persist_metadata,
        )
        typer.echo(result.model_dump_json(indent=2))
    finally:
        container.close()


@app.command()
def migrate(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """Create PostgreSQL/pgvector schema objects idempotently."""

    configure_logging()
    loaded = load_config(config)
    store = PostgresMetadataStore(
        required_secret(loaded.postgres.connection_string_env),
        loaded.postgres.schema_name,
    )
    try:
        store.initialize()
    finally:
        store.close()
    jira_store = PostgresJiraConfigurationStore(
        required_secret(loaded.postgres.connection_string_env),
        loaded.postgres.schema_name,
    )
    try:
        jira_store.initialize()
    finally:
        jira_store.close()
    typer.echo("PostgreSQL schema is ready.")


@app.command("onboard")
def onboard_repository(
    gitlab_url: Annotated[str, typer.Argument(help="Allowlisted HTTPS GitLab clone URL.")],
    config: ConfigOption = _DEFAULT_CONFIG,
    repository_id: Annotated[str | None, typer.Option("--id")] = None,
    name: Annotated[str | None, typer.Option("--name")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    ref: Annotated[str | None, typer.Option("--ref")] = None,
) -> None:
    """Register a GitLab repository and enqueue a full Beelinks rebuild."""

    configure_logging()
    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        result = container.onboarding.register_gitlab(
            gitlab_url,
            repository_id=repository_id,
            name=name,
            description=description,
            ref=ref,
        )
        typer.echo(
            json.dumps(
                {
                    "repository": result.repository.model_dump(mode="json"),
                    "job": result.job.model_dump(mode="json"),
                },
                indent=2,
            )
        )
    finally:
        container.close()


@app.command("repositories")
def list_repositories(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """List all registered local and GitLab repositories."""

    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        typer.echo(
            json.dumps(
                [
                    item.model_dump(mode="json", exclude={"local_path"})
                    for item in container.repository_store.list_repositories()
                ],
                indent=2,
            )
        )
    finally:
        container.close()


@app.command("migrate-local-to-gitlab")
def migrate_local_repositories_to_gitlab(
    sources: Annotated[
        list[str],
        typer.Option(
            "--source",
            help="Repeat ID,HTTPS_CLONE_URL,REF for every local repository in one atomic migration.",
        ),
    ],
    config: ConfigOption = _DEFAULT_CONFIG,
) -> None:
    """Pre-clone and atomically migrate local sources, then queue one portfolio refresh."""

    configure_logging()
    parsed = tuple(_parse_gitlab_migration_source(source) for source in sources)
    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        result = container.onboarding.migrate_local_sources_to_gitlab(parsed)
        typer.echo(
            json.dumps(
                {
                    "repositories": [
                        record.model_dump(mode="json", exclude={"local_path"})
                        for record in result.repositories
                    ],
                    "job": result.job.model_dump(mode="json"),
                },
                indent=2,
            )
        )
    finally:
        container.close()


def _parse_gitlab_migration_source(value: str) -> GitLabSourceMigration:
    parts = tuple(part.strip() for part in value.split(",", 2))
    if len(parts) != 3 or any(not part for part in parts):
        raise typer.BadParameter("each --source must be ID,HTTPS_CLONE_URL,REF")
    return GitLabSourceMigration(parts[0], parts[1], parts[2])


@app.command("reindex")
def enqueue_reindex(
    config: ConfigOption = _DEFAULT_CONFIG,
    repository_id: Annotated[str | None, typer.Option("--repository-id")] = None,
) -> None:
    """Enqueue a commit-aware portfolio refresh, optionally attributed to one repository."""

    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        job = container.onboarding.enqueue_reindex(repository_id)
        typer.echo(job.model_dump_json(indent=2))
    finally:
        container.close()


@app.command("adopt-existing")
def adopt_existing_snapshot(
    graph: Annotated[
        Path,
        typer.Option(
            "--graph",
            help="Existing KnowledgeGraph JSON file (mount it read-only in containers).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    knowledge_directory: Annotated[
        Path,
        typer.Option(
            "--knowledge-directory",
            help="Directory containing central.md and repositories/<repository-id>.md.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    config: ConfigOption = _DEFAULT_CONFIG,
    validate_only: Annotated[
        bool,
        typer.Option("--validate-only", help="Run all read-only preflight checks without publishing."),
    ] = False,
    replace_neo4j: Annotated[
        bool,
        typer.Option(
            "--replace-neo4j",
            help="Install or replace Neo4j only when it differs; matching graphs are always preserved.",
        ),
    ] = False,
) -> None:
    """Adopt existing graph and Markdown artifacts without invoking Bedrock."""

    configure_logging()
    container = ApplicationContainer(load_config(config))
    try:
        if validate_only:
            result = container.adoption.preflight(
                graph,
                knowledge_directory,
                replace_neo4j=replace_neo4j,
            )
        else:
            container.initialize_registry()
            result = container.adoption.adopt(
                graph,
                knowledge_directory,
                replace_neo4j=replace_neo4j,
            )
        typer.echo(result.model_dump_json(indent=2))
    finally:
        container.close()


@app.command("separate-internal-repository")
def separate_internal_repository(
    repository_id: Annotated[
        str,
        typer.Option("--repository-id", help="Published repository to move to the internal index."),
    ],
    restore_scan_run_id: Annotated[
        str,
        typer.Option(
            "--restore-scan-run-id",
            help="Historical public scan containing exactly the repositories that should remain.",
        ),
    ],
    internal_config: Annotated[
        Path,
        typer.Option(
            "--internal-config",
            help="Isolated runtime configuration for the destination internal index.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    staging_directory: Annotated[
        Path,
        typer.Option(
            "--staging-directory",
            help="Internal artifact directory used for the validated no-model adoption.",
        ),
    ],
    config: ConfigOption = _DEFAULT_CONFIG,
) -> None:
    """Split one published repository into an internal-only graph without re-scanning."""

    configure_logging()
    source = ApplicationContainer(load_config(config))
    destination = ApplicationContainer(load_config(internal_config))

    def initialize_internal() -> None:
        destination.initialize_database()
        destination.initialize_registry()

    try:
        source.initialize_registry()
        result = source.separation.separate(
            repository_id,
            restore_scan_run_id,
            staging_directory=staging_directory,
            initialize_internal=initialize_internal,
            internal_adoption=destination.adoption,
        )
        typer.echo(result.model_dump_json(indent=2))
    finally:
        destination.close()
        source.close()


@app.command("backfill-readers")
def backfill_readers(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """Publish reader editions for the latest evidence manifest without invoking a model."""

    configure_logging()
    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        result = container.reader_backfill.run()
        typer.echo(result.model_dump_json(indent=2))
    finally:
        container.close()


@app.command("worker")
def worker(
    config: ConfigOption = _DEFAULT_CONFIG,
    once: Annotated[bool, typer.Option("--once", help="Process at most one queued job.")] = False,
) -> None:
    """Process durable GitLab onboarding and portfolio reindex jobs."""

    configure_logging()
    container = ApplicationContainer(load_config(config))
    try:
        container.initialize_registry()
        container.recover_interrupted_jobs()
        if once:
            job = container.onboarding.process_next_job()
            typer.echo(job.model_dump_json(indent=2) if job is not None else "No queued jobs.")
            return
        container.worker.run_forever()
    finally:
        container.close()


@app.command()
def serve(config: ConfigOption = _DEFAULT_CONFIG) -> None:
    """Run the GraphQL API."""

    configure_logging()
    loaded = load_config(config)
    container = ApplicationContainer(loaded)
    api = create_app(PlatformService(container), loaded)
    uvicorn.run(api, host=loaded.graphql.host, port=loaded.graphql.port)


if __name__ == "__main__":
    app()
