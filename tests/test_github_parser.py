from __future__ import annotations

from datetime import date

from radar.collectors.github_collector import ingest_github, parse_github_repos
from radar.config import AppConfig, GitHubConfig, TopicConfig
from radar.db import connect, fetch_all, init_db, insert_repo_snapshot, upsert_repo
from radar.models import RepoSnapshot

GITHUB_PAYLOAD = {
    "total_count": 1,
    "incomplete_results": False,
    "items": [
        {
            "id": 123,
            "full_name": "example/fast-inference",
            "html_url": "https://github.com/example/fast-inference",
            "description": "Fast inference server",
            "language": "Python",
            "stargazers_count": 42,
            "forks_count": 7,
            "open_issues_count": 3,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-03T00:00:00Z",
            "pushed_at": "2024-01-04T00:00:00Z",
        }
    ],
}


def test_parse_github_repos_parses_one_repository() -> None:
    repos = parse_github_repos(GITHUB_PAYLOAD)

    assert len(repos) == 1
    assert repos[0].github_id == 123
    assert repos[0].full_name == "example/fast-inference"
    assert repos[0].url == "https://github.com/example/fast-inference"
    assert repos[0].description == "Fast inference server"
    assert repos[0].language == "Python"
    assert repos[0].stars == 42
    assert repos[0].forks == 7
    assert repos[0].open_issues == 3
    assert repos[0].created_at is not None
    assert repos[0].updated_at is not None
    assert repos[0].pushed_at is not None


def test_parse_github_repos_handles_missing_description() -> None:
    payload = {
        "items": [
            {
                **GITHUB_PAYLOAD["items"][0],
                "id": 456,
                "full_name": "example/no-description",
                "description": None,
            }
        ]
    }

    repo = parse_github_repos(payload)[0]

    assert repo.description == ""


def test_upsert_repo_updates_stars() -> None:
    conn = connect(":memory:")
    init_db(conn)
    repo = parse_github_repos(GITHUB_PAYLOAD)[0]

    first_id = upsert_repo(conn, repo)
    second_id = upsert_repo(conn, repo.model_copy(update={"stars": 100}))
    rows = fetch_all(conn, "SELECT id, stars FROM repos")

    assert first_id == second_id
    assert len(rows) == 1
    assert rows[0]["stars"] == 100


def test_insert_repo_snapshot_works() -> None:
    conn = connect(":memory:")
    init_db(conn)
    repo = parse_github_repos(GITHUB_PAYLOAD)[0]
    repo_id = upsert_repo(conn, repo)

    snapshot_id = insert_repo_snapshot(
        conn,
        RepoSnapshot(
            repo_id=repo_id,
            captured_at=date(2026, 7, 9),
            stars=repo.stars,
            forks=repo.forks,
            open_issues=repo.open_issues,
            pushed_at=repo.pushed_at,
        ),
    )
    rows = fetch_all(conn, "SELECT id, captured_at, stars, open_issues FROM repo_snapshots")

    assert rows[0]["id"] == snapshot_id
    assert rows[0]["captured_at"] == "2026-07-09"
    assert rows[0]["stars"] == 42
    assert rows[0]["open_issues"] == 3


def test_ingest_github_upserts_repos_and_snapshots_without_network() -> None:
    conn = connect(":memory:")
    init_db(conn)
    config = AppConfig(
        github=GitHubConfig(max_results=5),
        topics={"inference": TopicConfig(github_queries=["inference server"])},
    )

    def fetcher(query: str, per_page: int) -> dict[str, object]:
        assert query == "inference server"
        assert per_page == 5
        return GITHUB_PAYLOAD

    result = ingest_github(
        conn,
        config,
        fetcher=fetcher,
        captured_at=date(2026, 7, 9),
    )
    repos = fetch_all(conn, "SELECT full_name, stars, open_issues FROM repos")
    snapshots = fetch_all(conn, "SELECT captured_at, stars, open_issues FROM repo_snapshots")

    assert result.fetched == 1
    assert result.upserted == 1
    assert result.snapshots == 1
    assert repos[0]["full_name"] == "example/fast-inference"
    assert repos[0]["stars"] == 42
    assert repos[0]["open_issues"] == 3
    assert snapshots[0]["captured_at"] == "2026-07-09"
    assert snapshots[0]["stars"] == 42


def test_fetch_github_repos_retries_on_timeout(monkeypatch) -> None:
    import httpx

    from radar.collectors import github_collector

    calls = {"n": 0}

    class FakeResponse:
        def __init__(self, payload) -> None:
            self._payload = payload

        @property
        def status_code(self) -> int:
            return 200

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("timed out")
        return FakeResponse(GITHUB_PAYLOAD)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(github_collector.time, "sleep", lambda s: None)

    payload = github_collector.fetch_github_repos(
        "inference server", per_page=5, backoff_seconds=0.01
    )

    assert calls["n"] == 3
    assert payload["total_count"] == 1


def test_fetch_github_repos_returns_empty_after_max_timeouts(monkeypatch) -> None:
    import httpx

    from radar.collectors import github_collector

    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(github_collector.time, "sleep", lambda s: None)

    payload = github_collector.fetch_github_repos(
        "inference server", per_page=5, max_attempts=3, backoff_seconds=0.01
    )

    assert calls["n"] == 3
    assert payload == {"items": []}
