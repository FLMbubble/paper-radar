"""Auto-discovery of research sub-directions from collected papers.

Given a corpus of paper titles/abstracts (typically already ingested for a
broad direction), this module extracts salient key phrases, clusters them by
co-occurrence into sub-directions, and renders a ready-to-use config file.

The implementation is intentionally local-first and deterministic: no external
NLP services, no randomness. It relies only on the Python standard library.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from radar.config import TopicConfig
from radar.models import Paper

# Generic English + academic filler words. These never start/end a kept phrase.
STOPWORDS = frozenset(
    """
    a an the of for and or but in on at to from by with as is are was were be been
    this that these those it its their our we they them he she his her you your
    not no nor so than then too very can could should would may might must shall
    do does did doing done have has had having i me my we us our you your he him
    his she her they them their what which who whom whose when where why how all
    any both each few more most other some such only own same s t can will just
    into over under again further once here there about above below up down out off
    via per among across within without toward towards upon amongst amidst
    based using use uses used via through towards also however thus therefore
    while although though whereas whether either neither nor yet
    new novel efficient effective simple unified general robust scalable
    approach method framework system model models task tasks problem solution
    results result performance evaluation study analysis work paper research
    propose proposed present show demonstrate achieve improve introduce
    """.split()
)

# Additional generic terms filtered even if they pass stopword checks.
GENERIC_TERMS = frozenset(
    """
    learning training method methods accuracy data base based two three first
    input output token tokens image images text video generation large scale
    high low real better best state art cross multi single end open source
    public current previous recent future past main key primary results result
    performance evaluation study analysis work paper research propose proposed
    present show demonstrate achieve improve introduce general robust scalable
    unified simple efficient effective novel approach framework system model
    models task tasks problem solution
    extensive often diverse available understanding dataset datasets
    comprehensive significant substantial challenging complex various
    numerous multiple existing prior recent advanced promising potential
    different similar consistent comparable overall furthermore moreover
    notably importantly specifically particularly namely respectively
    rather achieves achieves evidence capability capabilities context contexts
    inference infer reason reasoning fine-grained fine grained coarse
    leveraging leverage leverages exploit exploits exploit utilize utilizes
    employ employs employ enable enables enabling allow allows allowing
    require requires requiring needs need support supports supporting
    provide provides providing offer offers including include includes
    focus focuses focusing aim aims targeting target targets
    address between extensive experiments knowledge settings setting
    domain domains domains field fields area areas aspect aspects
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


@dataclass(frozen=True)
class DiscoveryParams:
    """Tunable knobs for phrase extraction and clustering."""

    min_freq_unigram: int = 4
    min_freq_phrase: int = 2
    max_ngram: int = 3
    title_weight: float = 3.0
    top_phrases: int = 80
    jaccard_threshold: float = 0.34
    max_directions: int = 12
    max_keywords: int = 8
    base_weight: float = 1.0


@dataclass(frozen=True)
class SubDirection:
    """A discovered research sub-direction."""

    name: str
    label: str
    keywords: list[str]
    weight: float
    paper_count: int
    phrase_count: int


@dataclass
class _Phrase:
    text: str
    score: float
    doc_count: int
    ngram: int


@dataclass
class _Cluster:
    seed: str
    members: list[_Phrase] = field(default_factory=list)
    papers: set[int] = field(default_factory=set)


def discover_subdirections(
    papers: list[Paper],
    params: DiscoveryParams | None = None,
) -> list[SubDirection]:
    """Extract and cluster sub-directions from a list of papers."""
    params = params or DiscoveryParams()
    if not papers:
        return []
    per_paper = [_paper_grams(p, params.max_ngram) for p in papers]
    phrases = _score_phrases(per_paper, params)
    if not phrases:
        return []
    clusters = _cluster_phrases(phrases, per_paper, params)
    return _clusters_to_directions(clusters, params)


def discover_from_db(
    conn: sqlite3.Connection,
    params: DiscoveryParams | None = None,
    limit: int = 0,
) -> list[SubDirection]:
    """Discover sub-directions from papers already stored in the database."""
    query = "SELECT id, arxiv_id, title, abstract, url FROM papers"
    if limit > 0:
        query += f" LIMIT {int(limit)}"
    papers: list[Paper] = []
    for row in conn.execute(query):
        papers.append(
            Paper(
                arxiv_id=str(row["arxiv_id"]),
                title=str(row["title"]),
                abstract=str(row["abstract"]),
                url=str(row["url"]),
            )
        )
    return discover_subdirections(papers, params)


def to_topic_configs(directions: list[SubDirection]) -> dict[str, TopicConfig]:
    """Convert discovered directions into TopicConfig objects.

    Queries are intentionally left empty: the parent config's
    ``arxiv.queries`` / ``github.queries`` act as a shared fallback (see
    ``AppConfig.arxiv_queries``), so every sub-direction reuses the broad
    collection scope while tagging with its own keywords.
    """
    return {
        d.name: TopicConfig(keywords=list(d.keywords), weight=d.weight)
        for d in directions
    }


def render_config_yaml(
    database_path: str,
    reports_dir: str,
    arxiv_queries: list[str],
    github_queries: list[str],
    directions: list[SubDirection],
    max_results: int = 40,
) -> str:
    """Render a complete, ready-to-use config file from discovered directions."""
    lines: list[str] = [
        "# Auto-discovered sub-directions.",
        f"# Generated {date.today().isoformat()} from collected papers.",
        "",
        f"database_path: {database_path}",
        f"reports_dir: {reports_dir}",
        "",
        "arxiv:",
        "  enabled: true",
        f"  max_results: {max_results}",
        "  queries:",
    ]
    for q in arxiv_queries:
        lines.append(f"    - {_yaml_scalar(q)}")
    lines += [
        "",
        "github:",
        "  enabled: true",
        f"  max_results: {max_results}",
        "  token_env: GITHUB_TOKEN",
        "  queries:",
    ]
    for q in github_queries:
        lines.append(f"    - {_yaml_scalar(q)}")
    lines += ["", "topics:"]
    for d in directions:
        lines.append(f"  {d.name}:")
        lines.append(f"    weight: {d.weight}")
        lines.append("    keywords:")
        for kw in d.keywords:
            lines.append(f"      - {_yaml_scalar(kw)}")
    lines += [
        "",
        "scoring:",
        "  paper_weight: 1.0",
        "  repo_weight: 1.0",
        "  star_weight: 0.05",
        "  match_weight: 2.0",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phrase extraction
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _paper_grams(paper: Paper, max_ngram: int) -> tuple[set[str], set[str]]:
    """Return (title_grams, abstract_grams) of kept phrases for one paper."""
    title_tokens = _tokenize(paper.title)
    abstract_tokens = _tokenize(paper.abstract)
    title_grams = _grams(title_tokens, max_ngram)
    abstract_grams = _grams(abstract_tokens, max_ngram)
    return title_grams, abstract_grams


def _grams(tokens: list[str], max_ngram: int) -> set[str]:
    grams: set[str] = set()
    n_tokens = len(tokens)
    for n in range(1, max_ngram + 1):
        for i in range(n_tokens - n + 1):
            window = tokens[i : i + n]
            if _keep_gram(window):
                grams.add(" ".join(window))
    return grams


def _keep_gram(window: list[str]) -> bool:
    if not window:
        return False
    if window[0] in STOPWORDS or window[-1] in STOPWORDS:
        return False
    if all(tok in STOPWORDS for tok in window):
        return False
    # Drop phrases whose meaningful tokens are all generic fillers.
    meaningful = [tok for tok in window if tok not in STOPWORDS]
    if not meaningful or all(tok in GENERIC_TERMS for tok in meaningful):
        return False
    # Drop pure numbers and very short standalone tokens.
    for tok in window:
        if tok.isdigit() and len(tok) <= 2:
            return False
    return True


def _score_phrases(
    per_paper: list[tuple[set[str], set[str]]],
    params: DiscoveryParams,
) -> list[_Phrase]:
    score: Counter[str] = Counter()
    doc_count: Counter[str] = Counter()
    ngram_of: dict[str, int] = {}
    for title_grams, abstract_grams in per_paper:
        present = title_grams | abstract_grams
        for g in present:
            weight = 0.0
            if g in title_grams:
                weight += params.title_weight
            if g in abstract_grams:
                weight += 1.0
            score[g] += weight
            doc_count[g] += 1
            ngram_of.setdefault(g, len(g.split()))
    phrases: list[_Phrase] = []
    for g, s in score.items():
        n = ngram_of[g]
        min_freq = params.min_freq_unigram if n == 1 else params.min_freq_phrase
        if doc_count[g] < min_freq:
            continue
        # Favor multi-word phrases: they carry more topical signal than
        # generic single words. A 2-gram gets a 1.6x lift, a 3-gram 2.2x.
        lift = 1.0 + 0.6 * (n - 1)
        phrases.append(_Phrase(text=g, score=s * lift, doc_count=doc_count[g], ngram=n))
    phrases.sort(key=lambda p: (-p.score, p.text))
    return phrases[: params.top_phrases]


# --------------------------------------------------------------------------- #
# Co-occurrence clustering
# --------------------------------------------------------------------------- #



def _stem(token: str) -> str:
    """Crude normalization: drop common English plural/verb suffixes."""
    for suffix in ("ies", "es", "s", "ing", "ed"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _phrase_stem(phrase: str) -> str:
    return " ".join(_stem(t) for t in phrase.split())


def _cluster_phrases(
    phrases: list[_Phrase],
    per_paper: list[tuple[set[str], set[str]]],
    params: DiscoveryParams,
) -> list[_Cluster]:
    phrase_set = {p.text for p in phrases}
    phrase_papers: dict[str, set[int]] = {}
    for idx, (title_grams, abstract_grams) in enumerate(per_paper):
        present = title_grams | abstract_grams
        for g in present:
            if g in phrase_set:
                phrase_papers.setdefault(g, set()).add(idx)

    # Reorder so multi-word phrases are considered as seeds first: they carry
    # more topical signal. Unigrams are subsumed when they are a token of any
    # already-chosen multi-word phrase, preventing generic single words from
    # spawning their own clusters.
    ordered = sorted(phrases, key=lambda p: (-(p.ngram >= 2), -p.score, p.text))
    chosen_multiword_tokens: set[str] = set()
    clusters: list[_Cluster] = []
    for phrase in ordered:
        pset = phrase_papers.get(phrase.text, set())
        # Subsume unigrams covered by an existing multi-word seed.
        if phrase.ngram == 1 and phrase.text in chosen_multiword_tokens:
            continue
        placed = False
        for cluster in clusters:
            seed_set = phrase_papers.get(cluster.seed, set())
            if (
                _jaccard(pset, seed_set) >= params.jaccard_threshold
                or _phrase_stem(phrase.text) == _phrase_stem(cluster.seed)
            ):
                cluster.members.append(phrase)
                cluster.papers |= pset
                placed = True
                break
        if not placed:
            clusters.append(
                _Cluster(seed=phrase.text, members=[phrase], papers=set(pset))
            )
            if phrase.ngram >= 2:
                chosen_multiword_tokens.update(phrase.text.split())
    clusters.sort(key=lambda c: (-len(c.papers), c.seed))
    return clusters


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _clusters_to_directions(
    clusters: list[_Cluster],
    params: DiscoveryParams,
) -> list[SubDirection]:
    directions: list[SubDirection] = []
    for cluster in clusters:
        members = sorted(cluster.members, key=lambda p: (-p.score, p.text))
        keywords = [m.text for m in members[: params.max_keywords]]
        if not keywords:
            continue
        paper_count = len(cluster.papers)
        weight = _compute_weight(paper_count, params.base_weight)
        label = cluster.seed
        name = _slugify(label)
        directions.append(
            SubDirection(
                name=name,
                label=label,
                keywords=keywords,
                weight=weight,
                paper_count=paper_count,
                phrase_count=len(cluster.members),
            )
        )
    # Deduplicate names (keep first occurrence) and cap the count.
    seen: set[str] = set()
    unique: list[SubDirection] = []
    for d in directions:
        if d.name in seen:
            continue
        seen.add(d.name)
        unique.append(d)
    return unique[: params.max_directions]


def _compute_weight(paper_count: int, base: float) -> float:
    growth = 0.1 * math.log1p(paper_count)
    return round(min(max(base + growth, base), base + 0.5), 1)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "topic"


def _yaml_scalar(value: str) -> str:
    """Render a string as a single-quoted YAML scalar when needed."""
    if value and re.fullmatch(r"[a-zA-Z0-9_.\-/ ]+", value):
        return f'"{value}"'
    escaped = value.replace("'", "''")
    return f"'{escaped}'"
