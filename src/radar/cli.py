from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from radar.collectors.arxiv_collector import ingest_arxiv as ingest_arxiv_feed
from radar.collectors.github_collector import ingest_github as ingest_github_feed
from radar.config import AppConfig, load_config
from radar.db import connect, init_db
from radar.digest import write_digest
from radar.discovery import (
    DiscoveryParams,
    discover_from_db,
    render_config_yaml,
)
from radar.llm_topics import discover_with_llm, load_llm_config
from radar.matching import match_database
from radar.sample_data import load_sample_data
from radar.scoring import score_database
from radar.tagging import tag_database

app = typer.Typer(help="AI Infra Radar CLI")
ConfigPath = Annotated[Path, typer.Option("--config", "-c")]
DbPath = Annotated[Path | None, typer.Option("--db")]
DigestDate = Annotated[str, typer.Option("--date")]
SampleDbPath = Annotated[Path, typer.Option("--db")]
FetchReadme = Annotated[bool, typer.Option("--fetch-readme")]
DirectionsArg = Annotated[
    int, typer.Option("--directions", help="Max number of sub-directions to keep.")
]
MaxKeywordsArg = Annotated[
    int, typer.Option("--max-keywords", help="Max keywords per sub-direction.")
]
MinFreqArg = Annotated[
    int, typer.Option("--min-freq", help="Min paper frequency for a phrase to be kept.")
]
OutputPath = Annotated[
    Path | None,
    typer.Option("--out", "-o", help="Write config YAML here instead of stdout."),
]
UseLlm = Annotated[
    bool, typer.Option("--use-llm", help="Use an LLM for semantic grouping (needs OPENAI_API_KEY).")
]
ModelArg = Annotated[str | None, typer.Option("--model", help="LLM model name (default from env).")]
BaseUrlArg = Annotated[
    str | None,
    typer.Option("--base-url", help="OpenAI-compatible API base URL (default from env)."),
]


def _load(config: Path, db: Path | None = None) -> tuple[AppConfig, sqlite3.Connection]:
    cfg = load_config(config)
    conn = connect(db or cfg.database_path)
    init_db(conn)
    return cfg, conn


@app.command("init-db")
def init_db_command(config: ConfigPath = Path("config.yaml")) -> None:
    cfg, conn = _load(config)
    conn.close()
    typer.echo(f"Initialized database at {cfg.database_path}")


@app.command("ingest-arxiv")
def ingest_arxiv(config: ConfigPath = Path("config.yaml"), db: DbPath = None) -> None:
    cfg, conn = _load(config, db)
    if not cfg.arxiv.enabled:
        typer.echo("arXiv collector disabled")
        return
    result = ingest_arxiv_feed(conn, cfg)
    conn.close()
    typer.echo(f"Fetched {result.fetched} arXiv papers; upserted {result.upserted}")


@app.command("ingest-github")
def ingest_github(config: ConfigPath = Path("config.yaml"), db: DbPath = None) -> None:
    cfg, conn = _load(config, db)
    if not cfg.github.enabled:
        typer.echo("GitHub collector disabled")
        return
    result = ingest_github_feed(conn, cfg)
    conn.close()
    typer.echo(
        f"Fetched {result.fetched} GitHub repositories; "
        f"upserted {result.upserted}; snapshots {result.snapshots}"
    )


@app.command("tag")
def tag(config: ConfigPath = Path("config.yaml"), db: DbPath = None) -> None:
    cfg, conn = _load(config, db)
    count = tag_database(conn, cfg)
    conn.close()
    typer.echo(f"Tagged {count} papers")


@app.command("tag-papers")
def tag_papers(config: ConfigPath = Path("config.yaml"), db: DbPath = None) -> None:
    cfg, conn = _load(config, db)
    count = tag_database(conn, cfg)
    conn.close()
    typer.echo(f"Tagged {count} papers")


@app.command("match")
def match(
    config: ConfigPath = Path("config.yaml"),
    db: DbPath = None,
    fetch_readme: FetchReadme = False,
) -> None:
    _, conn = _load(config, db)
    count = match_database(conn, fetch_readme=fetch_readme)
    conn.close()
    typer.echo(f"Upserted {count} paper/repo matches")


@app.command("match-repos")
def match_repos(
    db: SampleDbPath = Path("data/radar.db"),
    fetch_readme: FetchReadme = False,
) -> None:
    conn = connect(db)
    init_db(conn)
    count = match_database(conn, fetch_readme=fetch_readme)
    conn.close()
    typer.echo(f"Upserted {count} paper/repo matches")


@app.command("score")
def score(config: ConfigPath = Path("config.yaml"), db: DbPath = None) -> None:
    cfg, conn = _load(config, db)
    count = score_database(conn, cfg)
    conn.close()
    typer.echo(f"Computed {count} paper scores")


@app.command("digest")
def digest(
    config: ConfigPath = Path("config.yaml"),
    db: DbPath = None,
    digest_date: DigestDate = "today",
) -> None:
    cfg, conn = _load(config, db)
    path = write_digest(conn, cfg.reports_dir, _parse_digest_date(digest_date))
    conn.close()
    typer.echo(f"Wrote {path}")


@app.command("load-sample")
def load_sample(db: SampleDbPath = Path("data/sample/sample.db")) -> None:
    conn = connect(db)
    result = load_sample_data(conn)
    conn.close()
    typer.echo(
        f"Loaded sample data into {db}: "
        f"{result.papers} papers, {result.repos} repos, "
        f"{result.tags} tags, {result.scores} scores, "
        f"{result.snapshots} snapshots, {result.matches} matches"
    )
    typer.echo(
        f"Sample digest supported. Run: python -m radar.cli digest --db {db} --date 2026-07-09"
    )


@app.command("discover-topics")
def discover_topics(
    config: ConfigPath = Path("config.yaml"),
    db: DbPath = None,
    directions: DirectionsArg = 12,
    max_keywords: MaxKeywordsArg = 8,
    min_freq: MinFreqArg = 2,
    use_llm: UseLlm = False,
    model: ModelArg = None,
    base_url: BaseUrlArg = None,
    out: OutputPath = None,
) -> None:
    """Auto-discover research sub-directions from ingested papers.

    By default uses a deterministic statistical method (TF-IDF + co-occurrence).
    Pass --use-llm to ask an LLM (OpenAI-compatible) for semantic grouping;
    requires OPENAI_API_KEY (or LLM_API_KEY). Without a key it falls back to
    the statistical method automatically.
    """
    cfg, conn = _load(config, db)

    if use_llm:
        llm_cfg = load_llm_config(
            base_url=base_url,
            model=model,
            max_directions=directions,
            max_keywords=max_keywords,
        )
        if not llm_cfg.enabled:
            typer.echo("No API key found; falling back to statistical method.")
        papers = _fetch_papers(conn)
        conn.close()
        found = discover_with_llm(papers, llm_cfg)
        method = "LLM" if llm_cfg.enabled else "statistical (fallback)"
    else:
        params = DiscoveryParams(
            max_directions=directions,
            max_keywords=max_keywords,
            min_freq_phrase=min_freq,
            min_freq_unigram=max(min_freq, 4),
        )
        found = discover_from_db(conn, params)
        conn.close()
        method = "statistical"

    if not found:
        typer.echo("No sub-directions discovered; ingest more papers first.")
        raise typer.Exit(code=1)

    typer.echo(f"Discovered {len(found)} sub-directions ({method}):")
    for d in found:
        typer.echo(
            f"  {d.name:24s} papers={d.paper_count:3d}  weight={d.weight}  "
            f"keywords={', '.join(d.keywords[:3])}..."
        )
    yaml = render_config_yaml(
        database_path=str(cfg.database_path).replace(".db", "-discovered.sqlite"),
        reports_dir=str(cfg.reports_dir) + "/discovered",
        arxiv_queries=cfg.arxiv_queries,
        github_queries=cfg.github_queries,
        directions=found,
        max_results=cfg.arxiv.max_results,
    )
    if out:
        out.write_text(yaml, encoding="utf-8")
        typer.echo(f"Wrote config to {out}")
    else:
        typer.echo("---")
        typer.echo(yaml)


def _fetch_papers(conn: sqlite3.Connection) -> list:
    """Fetch all papers as radar.models.Paper objects."""
    from radar.models import Paper

    papers = []
    for row in conn.execute("SELECT arxiv_id, title, abstract, url FROM papers"):
        papers.append(
            Paper(
                arxiv_id=str(row["arxiv_id"]),
                title=str(row["title"]),
                abstract=str(row["abstract"]),
                url=str(row["url"]),
            )
        )
    return papers


@app.command("dashboard")
def dashboard(config: ConfigPath = Path("config.yaml")) -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--",
        "--config",
        str(config),
    ]
    raise typer.Exit(subprocess.call(command))


def main() -> None:
    app()


def _parse_digest_date(value: str) -> date:
    if value == "today":
        return date.today()
    return date.fromisoformat(value)


if __name__ == "__main__":
    main()
