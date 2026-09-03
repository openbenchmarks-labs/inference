"""Lazy workflow registry used by the shared CLI."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

WORKFLOWS = {
    "micro-single-task": "openbenchmarks_inference.workflows.micro_single_task",
}


def get_workflow(name: str) -> ModuleType:
    try:
        module_name = WORKFLOWS[name]
    except KeyError as exc:
        raise ValueError(f"unknown workflow: {name}") from exc
    return import_module(module_name)

