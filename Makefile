.PHONY: install pipeline dashboard digest test lint demo-data demo-dashboard help

# Default database (used when DIR is not set; keeps original AI Infra behavior)
DB ?= data/radar.db
DEMO_DB ?= data/sample/sample.db

# Direction support: set DIR=vla|vlm|wm to switch radar direction.
# When DIR is set, CONFIG/DB/REPORTS_DIR auto-resolve to that direction's files.
# When unset, falls back to the original config.example.yaml / data/radar.db.
ifdef DIR
CONFIG := config.$(DIR).yaml
DB := data/radar-$(DIR).sqlite
DASHBOARD_DB := $(DB)
else
CONFIG := config.example.yaml
DASHBOARD_DB := $(DB)
endif

install:
	python -m pip install -e ".[dev]"

pipeline:
	python -m radar.cli ingest-arxiv --config $(CONFIG) --db $(DB)
	python -m radar.cli ingest-github --config $(CONFIG) --db $(DB)
	python -m radar.cli tag-papers --config $(CONFIG) --db $(DB)
	python -m radar.cli score --config $(CONFIG) --db $(DB)
	python -m radar.cli match-repos --db $(DB)
	python -m radar.cli digest --config $(CONFIG) --db $(DB) --date today

dashboard:
	RADAR_DB_PATH=$(DASHBOARD_DB) streamlit run app/streamlit_app.py

digest:
	python -m radar.cli digest --config $(CONFIG) --db $(DB) --date today

test:
	python -m pytest

lint:
	ruff check .

demo-data:
	python -m radar.cli load-sample --db $(DEMO_DB)

demo-dashboard:
	RADAR_DB_PATH=$(DEMO_DB) streamlit run app/streamlit_app.py

help:
	@echo "AI Infra Radar — Make targets"
	@echo ""
	@echo "  make install              Install dependencies"
	@echo "  make pipeline             Run full pipeline (default: AI Infra)"
	@echo "  make dashboard            Launch Streamlit dashboard"
	@echo "  make digest               Re-generate today's digest"
	@echo "  make demo-data            Load sample data"
	@echo "  make demo-dashboard       Launch dashboard with sample data"
	@echo "  make test                 Run tests"
	@echo "  make lint                 Run ruff"
	@echo ""
	@echo "Direction radar (set DIR=vla|vlm|wm):"
	@echo "  make pipeline DIR=vlm     Run pipeline for the VLM direction"
	@echo "  make dashboard DIR=vlm    Launch dashboard for the VLM direction"
	@echo "  make digest DIR=vlm       Re-generate digest for the VLM direction"
