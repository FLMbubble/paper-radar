from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from radar.config import AppConfig, GitHubConfig
from radar.db import insert_repo_snapshot, upsert_repo
from radar.models import Repo, RepoSnapshot

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitHubIngestResult:
    fetched: int
    upserted: int
    snapshots: int


GitHubFetcher = Callable[[str, int], dict[str, Any]]


def build_github_params(query: str, per_page: int) -> dict[str, str | int]:
    return {
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": per_page,
    }


def fetch_github_repos(
    query: str,
    per_page: int,
    token_env: str = "GITHUB_TOKEN",
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv(token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(
                GITHUB_SEARCH_URL,
                params=build_github_params(query, per_page),
                headers=headers,
                timeout=30.0,
            )
        except httpx.HTTPError as exc:
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
                continue
            logger.warning(
                "GitHub search network error for query %r after %s attempts: %s",
                query,
                max_attempts,
                exc,
            )
            return {"items": []}

        if response.status_code < 400:
            return response.json()
        if response.status_code in {403, 429}:
            logger.warning(
                "GitHub search rate limited or forbidden for query %r: HTTP %s",
                query,
                response.status_code,
            )
        if response.status_code in {403, 429} or response.status_code >= 500:
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
                continue
            if response.status_code in {403, 429}:
                return {"items": []}
        response.raise_for_status()

    return {"items": []}


def parse_github_repos(payload: dict[str, Any]) -> list[Repo]:
    return [_item_to_repo(item) for item in payload.get("items", [])]


def collect_github(config: AppConfig | GitHubConfig) -> list[Repo]:
    queries, per_page, token_env = _queries_limit_and_token(config)
    repos: dict[int, Repo] = {}
    for query in queries:
        payload = fetch_github_repos(query, per_page=per_page, token_env=token_env)
        for repo in parse_github_repos(payload):
            repos[repo.github_id] = repo
    return list(repos.values())


def ingest_github(
    conn: sqlite3.Connection,
    config: AppConfig | GitHubConfig,
    fetcher: GitHubFetcher | None = None,
    captured_at: date | None = None,
) -> GitHubIngestResult:
    queries, per_page, token_env = _queries_limit_and_token(config)
    repos: dict[int, Repo] = {}
    for query in queries:
        payload = (
            fetcher(query, per_page)
            if fetcher
            else fetch_github_repos(query, per_page=per_page, token_env=token_env)
        )
        for repo in parse_github_repos(payload):
            repos[repo.github_id] = repo

    snapshot_date = captured_at or date.today()
    snapshots = 0
    for repo in repos.values():
        repo_id = upsert_repo(conn, repo)
        insert_repo_snapshot(
            conn,
            RepoSnapshot(
                repo_id=repo_id,
                captured_at=snapshot_date,
                stars=repo.stars,
                forks=repo.forks,
                open_issues=repo.open_issues,
                pushed_at=repo.pushed_at,
            ),
        )
        snapshots += 1
    return GitHubIngestResult(fetched=len(repos), upserted=len(repos), snapshots=snapshots)


def _queries_limit_and_token(config: AppConfig | GitHubConfig) -> tuple[list[str], int, str]:
    if isinstance(config, AppConfig):
        return config.github_queries, config.github.max_results, config.github.token_env
    return list(dict.fromkeys(config.queries)), config.max_results, config.token_env


def _item_to_repo(item: dict[str, Any]) -> Repo:
    return Repo(
        github_id=int(item["id"]),
        full_name=str(item["full_name"]),
        description=str(item.get("description") or ""),
        url=str(item["html_url"]),
        stars=int(item.get("stargazers_count") or 0),
        forks=int(item.get("forks_count") or 0),
        open_issues=int(item.get("open_issues_count") or 0),
        language=item.get("language") if isinstance(item.get("language"), str) else None,
        created_at=_parse_github_datetime(item.get("created_at")),
        updated_at=_parse_github_datetime(item.get("updated_at")),
        pushed_at=_parse_github_datetime(item.get("pushed_at")),
    )


def _parse_github_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
