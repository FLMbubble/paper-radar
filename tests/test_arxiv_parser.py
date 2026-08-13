from __future__ import annotations

import json

from radar.collectors.arxiv_collector import ingest_arxiv, parse_arxiv_feed
from radar.config import AppConfig, ArxivConfig, TopicConfig
from radar.db import connect, fetch_all, init_db

ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>arXiv Query Results</title>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <updated>2024-01-03T04:05:06Z</updated>
    <published>2024-01-02T03:04:05Z</published>
    <title>
      Fast
      Inference    Systems
    </title>
    <summary>
      A paper
      about   serving infrastructure.
    </summary>
    <author>
      <name>Alice Researcher</name>
    </author>
    <author>
      <name>Bob Engineer</name>
    </author>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.DC" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf"
          href="http://arxiv.org/pdf/2401.01234v2"
          rel="related"
          type="application/pdf"/>
  </entry>
</feed>
"""


def test_parse_arxiv_feed_parses_one_entry() -> None:
    papers = parse_arxiv_feed(ATOM_XML)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.01234"
    assert papers[0].url == "http://arxiv.org/abs/2401.01234v2"


def test_parse_arxiv_feed_normalizes_whitespace() -> None:
    paper = parse_arxiv_feed(ATOM_XML)[0]

    assert paper.title == "Fast Inference Systems"
    assert paper.abstract == "A paper about serving infrastructure."


def test_parse_arxiv_feed_extracts_authors() -> None:
    paper = parse_arxiv_feed(ATOM_XML)[0]

    assert paper.authors == ["Alice Researcher", "Bob Engineer"]


def test_parse_arxiv_feed_extracts_categories() -> None:
    paper = parse_arxiv_feed(ATOM_XML)[0]

    assert paper.primary_category == "cs.LG"
    assert paper.categories == ["cs.LG", "cs.DC"]


def test_parse_arxiv_feed_extracts_pdf_url() -> None:
    paper = parse_arxiv_feed(ATOM_XML)[0]

    assert paper.pdf_url == "http://arxiv.org/pdf/2401.01234v2"


def test_ingest_arxiv_upserts_parsed_papers_without_network() -> None:
    conn = connect(":memory:")
    init_db(conn)
    config = AppConfig(
        arxiv=ArxivConfig(max_results=7),
        topics={"systems": TopicConfig(arxiv_queries=["cat:cs.LG"])},
    )

    def fetcher(search_query: str, start: int, max_results: int) -> str:
        assert search_query == "cat:cs.LG"
        assert start == 0
        assert max_results == 7
        return ATOM_XML

    result = ingest_arxiv(conn, config, fetcher=fetcher)
    rows = fetch_all(
        conn,
        "SELECT arxiv_id, title, authors, primary_category, categories, pdf_url FROM papers",
    )

    assert result.fetched == 1
    assert result.upserted == 1
    assert len(rows) == 1
    assert rows[0]["arxiv_id"] == "2401.01234"
    assert rows[0]["title"] == "Fast Inference Systems"
    assert json.loads(rows[0]["authors"]) == ["Alice Researcher", "Bob Engineer"]
    assert rows[0]["primary_category"] == "cs.LG"
    assert json.loads(rows[0]["categories"]) == ["cs.LG", "cs.DC"]
    assert rows[0]["pdf_url"] == "http://arxiv.org/pdf/2401.01234v2"


def test_fetch_arxiv_feed_retries_on_timeout(monkeypatch) -> None:
    import httpx

    from radar.collectors import arxiv_collector

    calls = {"n": 0}

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("timed out")
        return FakeResponse(ATOM_XML)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(arxiv_collector.time, "sleep", lambda s: None)

    xml = arxiv_collector.fetch_arxiv_feed("cat:cs.LG", backoff_seconds=0.01)

    assert calls["n"] == 3
    assert "Fast" in xml and "Inference" in xml and "</feed>" in xml
