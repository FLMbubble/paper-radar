# AI Infra Radar

Track which AI research ideas are turning into real code.

AI Infra Radar is a local-first dashboard for AI infrastructure trends across arXiv and GitHub. It collects public metadata, stores it in SQLite, applies rule-based topic tags, computes exploratory scores, generates Markdown digests, and renders a Streamlit dashboard.

## Live Demo

https://ai-infra-radar.streamlit.app

The hosted Streamlit demo reads from `data/sample/sample.db`. GitHub Actions refreshes that public metadata snapshot daily by running:

```text
ingest-arxiv -> ingest-github -> tag-papers -> match-repos -> score -> digest
```

The refreshed snapshot is committed back to the repository.

## What It Does

- Collects AI infrastructure papers from arXiv.
- Collects related open-source repositories from GitHub.
- Tags papers with a public v0.1 topic taxonomy.
- Scores papers with simple relevance and recency heuristics.
- Generates a daily Markdown digest.
- Shows everything in a local Streamlit dashboard.
- Links papers to likely GitHub implementations with explainable heuristics.

## Direction Radar

Track specific research directions — VLA, VLM, World Model — each with its own sub-topic tags. Switch directions by adding `DIR=...` to the standard make targets; no code changes needed.

```bash
source .venv/bin/activate
make pipeline DIR=vlm      # vla / vlm / wm
make dashboard DIR=vlm
```

Without `DIR`, `make pipeline` / `make dashboard` keep their original AI Infra behavior. Run `make help` for the full target list.

See [docs/directions.md](docs/directions.md) for the sub-topic list, manual CLI usage, and customization guide.

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
```

Quick demo with no API keys required:

```bash
make demo-data
make demo-dashboard
```

`make demo-dashboard` launches Streamlit with `RADAR_DB_PATH=data/sample/sample.db`, so the dashboard opens against the sample database automatically.

![AI Infra Radar Dashboard](docs/images/dashboard.png)

## One-Command Local Run

Run the full local pipeline against public APIs:

```bash
make pipeline
make dashboard
```

The dashboard uses `RADAR_DB_PATH` when set, otherwise `data/radar.db` when present, and falls back to the committed sample database at `data/sample/sample.db`.

## What the Dashboard Shows

- Papers collected: number of arXiv papers in local SQLite.
- Repos tracked: number of GitHub repos collected.
- Topic distribution: rule-based topic counts.
- Papers by publication date: publication date distribution of collected papers.
- Top papers by heuristic score: exploratory ranking based on tag relevance and recency.
- Daily digest: the latest generated Markdown report.

## Optional GitHub Token

Optional but recommended for better GitHub API coverage:

```bash
export GITHUB_TOKEN="..."
```

Without a token, the GitHub Search API may return partial results due to rate limits.

## Deploy

AI Infra Radar can run as a Streamlit Community Cloud demo with a committed public metadata snapshot.

- Main file path: `app/streamlit_app.py`
- Python version: 3.11+
- Demo DB: `data/sample/sample.db`
- Optional secret: `GITHUB_TOKEN` for live pipeline coverage, not needed for local demo runs

The app defaults to the sample DB on Streamlit Cloud when no live `data/radar.db` exists.
The hosted demo database refreshes daily through GitHub Actions and is committed back to `data/sample/sample.db`.

## Full Commands

```bash
python -m radar.cli ingest-arxiv --config config.example.yaml --db data/radar.db
python -m radar.cli ingest-github --config config.example.yaml --db data/radar.db
python -m radar.cli tag-papers --config config.example.yaml --db data/radar.db
python -m radar.cli score --config config.example.yaml --db data/radar.db
python -m radar.cli match-repos --db data/radar.db
python -m radar.cli digest --config config.example.yaml --db data/radar.db --date today
streamlit run app/streamlit_app.py
```

Useful Make targets:

```bash
make install
make pipeline
make dashboard
make digest
make test
make lint
```

## Reliability Note

arXiv and GitHub metadata comes from public APIs. Topic tags, paper-to-repo matching, and scores are heuristic in v0.1/v0.2; matching is intentionally conservative. Ranking is for exploration, not definitive research evaluation. GitHub results may be partial if rate limited. No paid LLM API is used.

## Roadmap

- Paper-to-repo matching.
- Repo star growth snapshots.
- Hugging Face model tracking.
- Anomaly detection.
- GitHub Actions daily run.
- Sample mode.
