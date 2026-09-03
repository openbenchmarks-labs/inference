"""Inference micro-single-task benchmark runner."""

from .config import BenchmarkConfig, load_config

WORKFLOW_NAME = "micro-single-task"

__all__ = ["BenchmarkConfig", "WORKFLOW_NAME", "load_config"]
