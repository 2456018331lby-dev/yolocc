.PHONY: install test lint format data train export benchmark demo serve clean

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[all]"

install-prod:
	pip install -e .

# ── Quality ──────────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/ --ignore-missing-imports

# ── Testing ──────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# ── Data ─────────────────────────────────────────────────────────────────────

data:
	python scripts/prepare_dataset.py generate --num 500

data-validate:
	python scripts/prepare_dataset.py validate

# ── Training ─────────────────────────────────────────────────────────────────

train:
	python -m src.train --cfg configs/train_cfg.yaml

train-resume:
	python -m src.train --cfg configs/train_cfg.yaml --resume

# ── Evaluation ───────────────────────────────────────────────────────────────

validate:
	python -m src.validate --weights results/runs/train/garbage_detect/weights/best.pt

# ── Export ───────────────────────────────────────────────────────────────────

export:
	python -m src.export_model --weights results/runs/train/garbage_detect/weights/best.pt --format onnx

export-int8:
	python -m src.export_model --weights results/runs/train/garbage_detect/weights/best.pt --format onnx --int8 --data configs/garbage.yaml

export-all:
	python -m src.export_model --weights results/runs/train/garbage_detect/weights/best.pt --format all

# ── Inference ────────────────────────────────────────────────────────────────

detect:
	python -m src.inference --source 0 --weights weights/best.onnx

detect-image:
	python -m src.inference --source test.jpg --weights weights/best.onnx --save results/detection.jpg

# ── Benchmark ────────────────────────────────────────────────────────────────

benchmark:
	python -m src.benchmark --weights results/runs/train/garbage_detect/weights/best.pt weights/best.onnx

# ── Visualization ────────────────────────────────────────────────────────────

viz-training:
	python -m src.visualize --type training --dir results/runs/train/garbage_detect

viz-benchmark:
	python -m src.visualize --type benchmark --json results/benchmark_results.json

viz-distribution:
	python -m src.visualize --type distribution --dir data/dataset/labels/train

# ── Server ───────────────────────────────────────────────────────────────────

serve:
	python -m src.api --weights weights/best.onnx --port 8000

demo:
	streamlit run deploy/app.py

# ── Pipeline ─────────────────────────────────────────────────────────────────

pipeline:
	python scripts/run_pipeline.py

pipeline-full:
	python scripts/run_pipeline.py --steps data validate train evaluate export export-int8 benchmark

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -t yolocc -f docker/Dockerfile .

docker-run:
	docker run -p 8000:8000 -v ./weights:/app/weights yolocc

docker-compose:
	docker compose -f docker/docker-compose.yml up

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache results/runs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
