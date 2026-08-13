from __future__ import annotations

import sqlite3

from radar.db import init_db, upsert_paper
from radar.discovery import (
    DiscoveryParams,
    SubDirection,
    discover_from_db,
    discover_subdirections,
    render_config_yaml,
    to_topic_configs,
)
from radar.models import Paper


def _paper(idx: int, title: str, abstract: str = "") -> Paper:
    return Paper(
        arxiv_id=f"2607.{idx:05d}",
        title=title,
        abstract=abstract,
        url=f"http://arxiv.org/abs/2607.{idx:05d}",
    )


def test_empty_corpus_returns_no_directions() -> None:
    assert discover_subdirections([]) == []


def test_discovers_clear_clusters() -> None:
    papers = [
        _paper(i, f"Reward Shaping for Robot Manipulation {i}", "reward model helps manipulation")
        for i in range(6)
    ] + [
        _paper(i, f"Video Generation World Model {i}", "video generation world model")
        for i in range(6, 12)
    ]
    directions = discover_subdirections(papers)
    # Two distinct clusters must emerge; reward/manipulation co-occur so they
    # merge into one cluster, video/world/generation into another.
    all_keywords = " ".join(kw for d in directions for kw in d.keywords)
    assert "reward" in all_keywords
    assert "video" in all_keywords or "generation" in all_keywords
    assert len(directions) >= 2


def test_subdirection_has_keywords_and_weight() -> None:
    papers = [
        _paper(i, f"Diffusion Policy for Action Generation {i}", "policy learning")
        for i in range(10)
    ]
    directions = discover_subdirections(papers)
    assert directions
    d = directions[0]
    assert isinstance(d, SubDirection)
    assert d.keywords
    assert d.weight >= 1.0
    assert d.paper_count > 0


def test_max_directions_limits_output() -> None:
    papers = [
        _paper(i, f"Topic{i} keyword{i} system {i}", f"keyword{i} domain{i}") for i in range(60)
    ]
    params = DiscoveryParams(
        min_freq_unigram=2,
        min_freq_phrase=2,
        max_directions=5,
        top_phrases=40,
        jaccard_threshold=0.9,
    )
    directions = discover_subdirections(papers, params)
    assert len(directions) <= 5


def test_render_config_yaml_is_parseable() -> None:
    papers = [
        _paper(i, f"Reward Shaping Manipulation {i}", "reward model manipulation") for i in range(8)
    ]
    directions = discover_subdirections(papers)
    yaml = render_config_yaml(
        database_path="data/test.sqlite",
        reports_dir="reports/test",
        arxiv_queries=['cat:cs.RO AND ("reward" OR manipulation)'],
        github_queries=["reward model language:Python"],
        directions=directions,
    )
    assert "database_path: data/test.sqlite" in yaml
    assert "topics:" in yaml
    assert "  weight:" in yaml
    assert "  keywords:" in yaml
    assert "scoring:" in yaml
    # round-trip through the real config loader
    import tempfile
    from pathlib import Path

    from radar.config import load_config

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml)
        path = Path(fh.name)
    cfg = load_config(path)
    assert cfg.topics
    for topic in cfg.topics.values():
        assert topic.keywords


def test_to_topic_configs_preserves_keywords() -> None:
    papers = [
        _paper(i, f"Diffusion Policy Action {i}", "policy learning action generation")
        for i in range(8)
    ]
    directions = discover_subdirections(papers)
    configs = to_topic_configs(directions)
    assert configs
    for _name, topic in configs.items():
        assert topic.keywords
        assert topic.weight >= 1.0


def test_discover_from_db_uses_ingested_papers() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    for i in range(8):
        upsert_paper(conn, _paper(i, f"Reward Shaping Manipulation {i}", "reward model"))
    directions = discover_from_db(conn, DiscoveryParams(min_freq_unigram=2, min_freq_phrase=2))
    conn.close()
    assert directions
    assert any("reward" in n or "manipulation" in n for n in [d.name for d in directions])
