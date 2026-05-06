"""
Full Pipeline Runner — Industrial Grade
One command to execute the entire workflow with error handling.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --steps data train export benchmark
    python scripts/run_pipeline.py --skip benchmark
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger, log_kv, log_section, log_success

log = get_logger("pipeline")
PROJECT_ROOT = Path(__file__).parent.parent

# Default experiment paths — matches train_cfg.yaml project/name
DEFAULT_EXPERIMENT = "results/runs/train/object_detect"
BEST_PT = f"{DEFAULT_EXPERIMENT}/weights/best.pt"
BEST_ONNX = f"{DEFAULT_EXPERIMENT}/weights/best.onnx"


def _run(cmd: str, desc: str) -> bool:
    """Run a command with logging."""
    log.info(f"\n{'─' * 60}")
    log.info(f"STEP: {desc}")
    log.info(f"CMD:  {cmd}")
    log.info("─" * 60)

    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error(f"FAILED: {desc} (exit {result.returncode}, {elapsed:.1f}s)")
        return False

    log_success(f"DONE: {desc} ({elapsed:.1f}s)")
    return True


def step_data() -> bool:
    return _run(f"{sys.executable} scripts/prepare_dataset.py generate --num 500", "Generate synthetic dataset")


def step_validate() -> bool:
    return _run(f"{sys.executable} scripts/prepare_dataset.py validate", "Validate dataset")


def step_train() -> bool:
    return _run(f"{sys.executable} -m src.train --cfg configs/train_cfg.yaml", "Train YOLOv8 model")


def step_evaluate() -> bool:
    return _run(
        f"{sys.executable} -m src.validate --weights {BEST_PT}",
        "Evaluate model",
    )


def step_export() -> bool:
    return _run(
        f"{sys.executable} -m src.export_model --weights {BEST_PT} --format onnx",
        "Export ONNX model",
    )


def step_export_int8() -> bool:
    return _run(
        f"{sys.executable} -m src.export_model "
        f"--weights {BEST_PT} "
        f"--format onnx --int8 --data configs/garbage.yaml",
        "Export INT8 model",
    )


def step_benchmark() -> bool:
    best_pt = PROJECT_ROOT / BEST_PT
    best_onnx = PROJECT_ROOT / BEST_ONNX
    weights = [str(p) for p in [best_pt, best_onnx] if p.exists()]
    if not weights:
        log.warning("No weights found for benchmarking, skipping")
        return True
    return _run(f"{sys.executable} -m src.benchmark --weights {' '.join(weights)}", "Benchmark models")


def step_demo() -> bool:
    log.info("\nLaunching Streamlit demo...")
    log.info("Open http://localhost:8501")
    subprocess.run("streamlit run deploy/app.py --server.headless true", shell=True, cwd=str(PROJECT_ROOT))
    return True


STEPS = {
    "data": step_data,
    "validate": step_validate,
    "train": step_train,
    "evaluate": step_evaluate,
    "export": step_export,
    "export-int8": step_export_int8,
    "benchmark": step_benchmark,
    "demo": step_demo,
}

DEFAULT_STEPS = ["data", "validate", "train", "evaluate", "export", "benchmark"]


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--steps", nargs="+", default=None, help=f"Steps: {list(STEPS.keys())}")
    parser.add_argument("--skip", nargs="+", default=[], help="Steps to skip")
    args = parser.parse_args()

    steps = args.steps or DEFAULT_STEPS
    steps = [s for s in steps if s not in args.skip]

    log_section("YOLO Garbage Detection — Full Pipeline", log)
    log_kv("Steps", " → ".join(steps), log)
    log_kv("Project", str(PROJECT_ROOT), log)

    t_total = time.time()
    results = {}

    for step in steps:
        if step not in STEPS:
            log.warning(f"Unknown step: {step}")
            continue

        t0 = time.time()
        ok = STEPS[step]()
        results[step] = {"ok": ok, "time": time.time() - t0}

        if not ok:
            log.error(f"Pipeline aborted at step: {step}")
            break

    total_time = time.time() - t_total

    log_section("Pipeline Summary", log)
    for step, r in results.items():
        status = "OK" if r["ok"] else "FAILED"
        log.info(f"  {step:15s}: {status} ({r['time']:.1f}s)")
    log.info(f"  {'TOTAL':15s}: {total_time:.1f}s")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
