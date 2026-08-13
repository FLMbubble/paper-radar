from __future__ import annotations

from radar.llm_topics import (
    LLMTopicsConfig,
    _parse_response,
    _strip_fences,
    discover_with_llm,
    load_llm_config,
)
from radar.models import Paper


def _paper(idx: int, title: str, abstract: str = "") -> Paper:
    return Paper(
        arxiv_id=f"2607.{idx:05d}",
        title=title,
        abstract=abstract,
        url=f"http://arxiv.org/abs/2607.{idx:05d}",
    )


def test_disabled_falls_back_to_statistical() -> None:
    papers = [
        _paper(i, f"Reward Shaping Manipulation {i}", "reward model manipulation")
        for i in range(8)
    ]
    config = LLMTopicsConfig(api_key=None, max_directions=5)
    directions = discover_with_llm(papers, config)
    assert directions
    # Statistical fallback must return real sub-directions.
    assert all(d.keywords for d in directions)


def test_parse_response_extracts_directions() -> None:
    raw = """
    [
      {
        "name": "reward_shaping",
        "label": "Reward Shaping",
        "keywords": ["reward shaping", "reward model", "success detector"],
        "rationale": "papers about reward design"
      },
      {
        "name": "video_generation",
        "label": "Video Generation",
        "keywords": ["video generation", "text-to-video"],
        "rationale": "generating videos"
      }
    ]
    """
    config = LLMTopicsConfig(api_key="x", max_directions=10, max_keywords=5)
    directions = _parse_response(raw, config)
    assert len(directions) == 2
    assert directions[0].name == "reward_shaping"
    assert directions[0].label == "Reward Shaping"
    assert "reward shaping" in directions[0].keywords
    assert directions[1].name == "video_generation"


def test_parse_response_tolerates_markdown_fences() -> None:
    raw = (
        "```json\n"
        '    [{"name": "hallucination", "label": "Hallucination", '
        '"keywords": ["hallucination", "faithfulness"], "rationale": "x"}]\n'
        "    ```\n"
        "    "
    )
    config = LLMTopicsConfig(api_key="x")
    directions = _parse_response(raw, config)
    assert len(directions) == 1
    assert directions[0].name == "hallucination"


def test_parse_response_handles_surrounding_prose() -> None:
    raw = (
        "Here are the directions:\n"
        '[{"name": "a", "label": "A", "keywords": ["k"], "rationale": ""}]\n'
        "Hope this helps!"
    )
    config = LLMTopicsConfig(api_key="x")
    directions = _parse_response(raw, config)
    assert len(directions) == 1
    assert directions[0].name == "a"


def test_parse_response_empty_on_garbage() -> None:
    config = LLMTopicsConfig(api_key="x")
    assert _parse_response("not json at all", config) == []
    assert _parse_response("[]", config) == []


def test_parse_response_caps_keywords() -> None:
    raw = (
        '[{"name":"t","label":"T",'
        '"keywords":["a","b","c","d","e","f","g","h","i","j"],'
        '"rationale":""}]'
    )
    config = LLMTopicsConfig(api_key="x", max_keywords=3)
    directions = _parse_response(raw, config)
    assert len(directions[0].keywords) == 3


def test_strip_fences_removes_json_fences() -> None:
    assert (
        _strip_fences("```json\nhi\n```").strip() == "hi"
    )
    assert _strip_fences("plain text").strip() == "plain text"


def test_load_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1/")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    config = load_llm_config()
    assert config.api_key == "sk-test"
    assert config.base_url == "https://example.com/v1"  # trailing slash stripped
    assert config.model == "gpt-test"
    assert config.enabled


def test_load_config_prefers_explicit_args(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    config = load_llm_config(api_key="explicit-key", model="my-model")
    assert config.api_key == "explicit-key"
    assert config.model == "my-model"


def test_load_config_disabled_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = load_llm_config()
    assert not config.enabled
