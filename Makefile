.PHONY: help install fixtures test lint fmt eval sweep demo stats docs clean

help:
	@echo "install   install the package plus dev tooling"
	@echo "fixtures  regenerate the labelled fixture set"
	@echo "test      run the test suite"
	@echo "lint      ruff check + format check"
	@echo "fmt       apply ruff formatting"
	@echo "eval      score the pipeline against the old script"
	@echo "sweep     sweep the clustering threshold"
	@echo "demo      full offline dry run against fixtures"
	@echo "stats     summarise the local corpus"
	@echo "docs      render docs/*.md to PDF"

install:
	pip install -e ".[dev]"

fixtures:
	python scripts/generate_fixtures.py

test:
	pytest -q

lint:
	ruff check src evals tests scripts
	ruff format --check src evals tests scripts

fmt:
	ruff format src evals tests scripts
	ruff check --fix src evals tests scripts

eval:
	python -m evals.run_eval

sweep:
	python -m evals.run_eval --sweep

demo:
	rei run --from-file fixtures/slack_messages.json --offline --dry-run --db /tmp/rei-demo.db

stats:
	rei stats

docs:
	python scripts/build_docs.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache eval.json eval.txt
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
