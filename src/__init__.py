"""
YOLO Garbage Classification Detection System
=============================================

Industrial-grade object detection for garbage classification using YOLOv8.

Modules:
    config       — Pydantic-validated configuration
    logger       — Structured logging with rich
    train        — Training pipeline
    validate     — Model evaluation
    export_model — Model export (ONNX/INT8/TorchScript)
    inference    — Unified inference engine
    benchmark    — Performance benchmarking
    api          — REST API server
    visualize    — Plotting utilities
    cli          — CLI entry point
"""

__version__ = "2.0.0"
__author__ = "YOLOCC"
