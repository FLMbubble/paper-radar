from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

import feedparser
import httpx

from radar.config import AppConfig, ArxivConfig
from radar.db import upsert_papers
from radar.models import Paper

ARXIV_API_URL = "https://export.arxiv.org/api/query"


@dataclass(frozen=True)
class ArxivIngestResult:
    fetched: int
    upserted: int


FeedFetcher = Callable[[str, int, int], str]


def build_arxiv_params(
    search_query: str,
    start: int = 0,
    max_results: int = 25,
) -> dict[str, str | int]:
    return {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }


def fetch_arxiv_feed(
    search_query: str,
    start: int = 0,
    max_results: int = 25,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(
                ARXIV_API_URL,
                params=build_arxiv_params(search_query, start, max_results),
                timeout=30.0,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
                continue
            raise


def parse_arxiv_feed(xml: str) -> list[Paper]:
    feed = feedparser.parse(xml)
    return [_entry_to_paper(entry) for entry in feed.entries]


def collect_arxiv(config: AppConfig | ArxivConfig) -> list[Paper]:
    queries, max_results = _queries_and_limit(config)
    papers: dict[str, Paper] = {}
    for query in queries:
        for paper in parse_arxiv_feed(fetch_arxiv_feed(query, start=0, max_results=max_results)):
            papers[paper.arxiv_id] = paper
    return list(papers.values())


def ingest_arxiv(
    conn: sqlite3.Connection,
    config: AppConfig | ArxivConfig,
    fetcher: FeedFetcher = fetch_arxiv_feed,
) -> ArxivIngestResult:
    queries, max_results = _queries_and_limit(config)
    papers: dict[str, Paper] = {}
    for query in queries:
        xml = fetcher(query, 0, max_results)
        for paper in parse_arxiv_feed(xml):
            papers[paper.arxiv_id] = paper
    upserted = upsert_papers(conn, papers.values())
    return ArxivIngestResult(fetched=len(papers), upserted=upserted)


def _queries_and_limit(config: AppConfig | ArxivConfig) -> tuple[list[str], int]:
    if isinstance(config, AppConfig):
        return config.arxiv_queries, config.arxiv.max_results
    return list(dict.fromkeys(config.queries)), config.max_results


def _entry_to_paper(entry: object) -> Paper:
    entry_id = str(_get(entry, "id", ""))
    categories = _categories(entry)
    primary_category = _primary_category(entry) or (categories[0] if categories else None)
    return Paper(
        arxiv_id=_extract_arxiv_id(entry_id),
        title=_normalize_whitespace(str(_get(entry, "title", ""))),
        abstract=_normalize_whitespace(str(_get(entry, "summary", ""))),
        authors=_authors(entry),
        primary_category=primary_category,
        categories=categories,
        published_at=_parse_datetime(_get(entry, "published")),
        updated_at=_parse_datetime(_get(entry, "updated")),
        url=entry_id,
        pdf_url=_pdf_url(entry),
    )


def _extract_arxiv_id(entry_id: str) -> str:
    path = urlsplit(entry_id).path
    raw_id = path.rstrip("/").rsplit("/", maxsplit=1)[-1] if path else entry_id
    return re.sub(r"v\d+$", "", raw_id)


def _authors(entry: object) -> list[str]:
    authors: list[str] = []
    for author in _get(entry, "authors", []):
        name = _get(author, "name")
        if name:
            authors.append(str(name))
    return authors


def _categories(entry: object) -> list[str]:
    categories: list[str] = []
    for tag in _get(entry, "tags", []):
        term = _get(tag, "term")
        if term:
            categories.append(str(term))
    primary = _primary_category(entry)
    if primary:
        categories.insert(0, primary)
    return list(dict.fromkeys(categories))


def _primary_category(entry: object) -> str | None:
    category = _get(entry, "arxiv_primary_category")
    term = _get(category, "term") if category else None
    return str(term) if term else None


def _pdf_url(entry: object) -> str | None:
    for link in _get(entry, "links", []):
        title = str(_get(link, "title", "")).lower()
        link_type = str(_get(link, "type", "")).lower()
        href = _get(link, "href")
        if href and (title == "pdf" or link_type == "application/pdf"):
            return str(href)
    return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    if value.endswith("Z"):
        value = value.removesuffix("Z") + "+00:00"
    return datetime.fromisoformat(value)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _get(item: object, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)
