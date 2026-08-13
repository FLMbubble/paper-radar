"""LLM-powered discovery of research sub-directions.

A thin, dependency-free client (stdlib ``urllib`` only) that asks an
OpenAI-compatible Chat Completions API to归纳 research sub-directions from
collected paper titles/abstracts.

Design:
  * ``discovery.py`` first extracts salient candidate phrases (TF-IDF style)
    so we send a compact signal instead of full abstracts.
  * This module sends those candidates plus paper samples to an LLM, which
    performs *semantic* grouping into meaningful sub-directions.
  * Falls back gracefully when no API key is configured.

No new dependencies: only the Python standard library is used.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from radar.discovery import DiscoveryParams, SubDirection, discover_subdirections
from radar.models import Paper

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60.0
MAX_SAMPLE_PAPERS = 120
MAX_CANDIDATE_PHRASES = 60


@dataclass(frozen=True)
class LLMTopicsConfig:
    """Configuration for the LLM sub-direction discovery."""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_directions: int = 12
    max_keywords: int = 8
    extra_instructions: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def load_llm_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    max_directions: int = 12,
    max_keywords: int = 8,
    extra_instructions: str = "",
) -> LLMTopicsConfig:
    """Build config from explicit args, falling back to env vars.

    Env vars (in priority order):
      OPENAI_API_KEY / LLM_API_KEY  — API key
      OPENAI_BASE_URL / LLM_BASE_URL — API base (for compatible services)
      OPENAI_MODEL / LLM_MODEL       — model name
    """
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
    mdl = model or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    return LLMTopicsConfig(
        api_key=key,
        base_url=base.rstrip("/"),
        model=mdl,
        max_directions=max_directions,
        max_keywords=max_keywords,
        extra_instructions=extra_instructions,
    )


def discover_with_llm(
    papers: list[Paper],
    config: LLMTopicsConfig,
) -> list[SubDirection]:
    """Discover sub-directions using an LLM.

    If the LLM is disabled (no key) or the call fails, transparently falls
    back to the deterministic statistical method in ``discovery.py``.
    """
    if not papers:
        return []
    if not config.enabled:
        return discover_subdirections(papers, DiscoveryParams(max_directions=config.max_directions))

    candidates = _candidate_phrases(papers)
    sample = _sample_papers(papers)
    prompt = _build_prompt(candidates, sample, config)
    try:
        raw = _chat_complete(prompt, config)
        directions = _parse_response(raw, config)
    except (urllib.error.URLError, ValueError, RuntimeError, json.JSONDecodeError):
        # Network/API failure → deterministic fallback.
        return discover_subdirections(papers, DiscoveryParams(max_directions=config.max_directions))

    if not directions:
        return discover_subdirections(papers, DiscoveryParams(max_directions=config.max_directions))
    return directions


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


def _build_prompt(
    candidates: list[str],
    sample: list[Paper],
    config: LLMTopicsConfig,
) -> str:
    paper_lines = "\n".join(
        f"- {p.title}" + (f" | {p.abstract[:160]}" if p.abstract else "") for p in sample
    )
    candidate_lines = ", ".join(candidates)
    extra = (
        f"\n\nAdditional instructions: {config.extra_instructions}"
        if config.extra_instructions
        else ""
    )
    return f"""You are a research analyst tracking the AI infrastructure field.

Below are {len(sample)} recent paper titles (with short abstract excerpts) and a
list of {len(candidates)} high-frequency key phrases extracted from a larger
corpus in the same broad research direction.

Papers:
{paper_lines}

Extracted key phrases:
{candidate_lines}

Your task: identify {config.max_directions} or fewer meaningful RESEARCH
SUB-DIRECTIONS in this corpus. These should be specific research themes (e.g.
"reward shaping", "action chunking", "hallucination mitigation"), NOT generic
words like "method", "training", "results", "improves".

Respond with ONLY a JSON array (no prose, no markdown fences). Each element:
{{
  "name": "snake_case_identifier",
  "label": "Human Readable Label",
  "keywords": ["phrase1", "phrase2", ...],
  "rationale": "one short sentence"
}}

Constraints:
- "name": lowercase snake_case, 2-4 words.
- "keywords": {config.max_keywords} or fewer specific phrases that would appear
  in a paper title/abstract for this sub-direction.
- Keep only sub-directions well-supported by the papers above.
- Do NOT include generic terms.{extra}
"""


def _candidate_phrases(papers: list[Paper]) -> list[str]:
    """Use the statistical extractor to get compact signal phrases."""
    from radar.discovery import _paper_grams, _score_phrases  # type: ignore[attr-defined]

    per_paper = [_paper_grams(p, 3) for p in papers]
    phrases = _score_phrases(per_paper, DiscoveryParams(top_phrases=MAX_CANDIDATE_PHRASES))
    return [p.text for p in phrases]


def _sample_papers(papers: list[Paper]) -> list[Paper]:
    """Pick a representative, stable sample (most recent by arxiv_id)."""
    sorted_papers = sorted(papers, key=lambda p: p.arxiv_id, reverse=True)
    return sorted_papers[:MAX_SAMPLE_PAPERS]


# --------------------------------------------------------------------------- #
# API call
# --------------------------------------------------------------------------- #


def _chat_complete(prompt: str, config: LLMTopicsConfig) -> str:
    payload = json.dumps(
        {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise research analyst. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def _parse_response(raw: str, config: LLMTopicsConfig) -> list[SubDirection]:
    """Parse the LLM JSON response into SubDirection objects."""
    text = _strip_fences(raw).strip()
    # Tolerate leading/trailing prose by locating the first '[' ... last ']'.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    text = text[start : end + 1]
    items = json.loads(text)
    directions: list[SubDirection] = []
    for item in items:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        keywords = [str(k).strip() for k in item.get("keywords", []) if str(k).strip()]
        if not keywords:
            continue
        label = str(item.get("label", name)).strip()
        directions.append(
            SubDirection(
                name=name,
                label=label,
                keywords=keywords[: config.max_keywords],
                weight=1.2,
                paper_count=0,
                phrase_count=len(keywords),
            )
        )
    return directions[: config.max_directions]


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return stripped
