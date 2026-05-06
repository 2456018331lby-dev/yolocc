"""
Allow running src modules directly:
    python -m src.train
    python -m src.validate
    python -m src.export_model
    python -m src.inference
    python -m src.benchmark
    python -m src.visualize
    python -m src.cli
    python -m src.api
"""

from src.cli import main

if __name__ == "__main__":
    main()
