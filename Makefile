PY ?= python

.PHONY: install data train serve test lint load pipeline bench

install:
	$(PY) -m pip install -e ".[train,dev]"

data:
	dvc repro build_dataset

train:
	dvc repro

serve:
	$(PY) -m uvicorn promptshield.serving.app:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

load:
	locust -f load_tests/locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000

bench:
	$(PY) -m promptshield.models.benchmark

pipeline:
	$(PY) -m promptshield.pipelines.flows daily_aggregates
