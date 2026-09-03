"""Strict projection of the micro-single-task benchmark specification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = PACKAGE_ROOT / "assets/spec.yaml"
PROVIDER_COUNT = 10
TASK_FAMILIES = (
    "meeting-notes-lookup",
    "ticket-triage",
    "contract-terms-extraction",
)


def project_path(value: str, label: str) -> Path:
    """Resolve a user-owned path against the current project directory."""
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


@dataclass(frozen=True)
class TokenPricing:
    input: float
    output: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (input_tokens, output_tokens)):
            raise ValueError("token usage must be non-negative integers")
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000


@dataclass(frozen=True)
class BenchmarkConfig:
    spec_path: Path
    raw: dict[str, Any]
    code_root: Path
    run_root: Path
    dataset_root: Path
    manifest_path: Path
    benchmark_slug: str
    providers: tuple[str, ...]
    task_families: tuple[str, ...]
    pricing: dict[str, TokenPricing]
    pricing_hash: str

    @property
    def smoke_request_limit(self) -> int:
        return int(self.raw["smoke_run"]["budget"]["maximum_requests_per_attempt"])

    @property
    def smoke_cost_limit(self) -> float:
        return float(self.raw["smoke_run"]["budget"]["maximum_cost_usd_across_attempts"])

    @property
    def smoke_maximum_attempts(self) -> int:
        return int(self.raw["smoke_run"]["maximum_attempts"])

    @property
    def timeout_seconds(self) -> float:
        return float(self.raw["inference"]["timeout_seconds"])

    @property
    def full_rounds(self) -> int:
        return int(self.raw["execution_plan"]["question_rounds"])

    @property
    def selected_per_family(self) -> int:
        return int(self.raw["dataset"]["selection"]["selected_items_per_task_family"])

    def usage_cost(self, provider: str, usage: Any) -> float | None:
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (input_tokens, output_tokens)):
            return None
        return self.pricing[provider].cost(input_tokens, output_tokens)


def _validate(raw: dict[str, Any]) -> None:
    required = {
        "runner", "benchmark", "dataset", "providers", "inference", "execution_plan",
        "stream_timing", "rate_limit_policy", "evaluation", "metrics", "persistence",
        "smoke_run", "reporting", "operations",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"spec missing sections: {', '.join(sorted(missing))}")
    providers = raw["providers"]["vendors"]
    if len(providers) != PROVIDER_COUNT:
        raise ValueError("benchmark requires exactly ten provider arms")
    families = raw["dataset"]["task_families"]
    if tuple(families) != TASK_FAMILIES:
        raise ValueError("task-family order differs from the declared round-robin order")
    if raw["benchmark"].get("status") != "draft":
        raise ValueError("the bundled workflow contract must remain in draft status")
    inference = raw["inference"]
    expected = {
        "streaming": True, "temperature": 0, "top_p": 1, "top_k": "disabled",
        "presence_penalty": 0, "frequency_penalty": 0, "max_completion_tokens": 256,
        "timeout_seconds": 20, "max_attempts": 1, "turns": 1,
    }
    for key, value in expected.items():
        if inference.get(key) != value:
            raise ValueError(f"inference.{key} conflicts with the workflow contract")
    if inference.get("reasoning", {}).get("default_effort") != "low" or inference["reasoning"].get("provider_overrides") != {}:
        raise ValueError("all providers must use equivalent low reasoning without overrides")
    plan = raw["execution_plan"]
    required_plan = {
        "requests_per_provider": 600,
        "total_provider_requests": 6000,
        "question_rounds": 600,
        "per_provider_concurrency": 1,
        "maximum_global_concurrency": 10,
        "simultaneous_provider_calls_per_question": True,
        "wait_for_all_providers_before_next_question": True,
        "identical_schedule_across_providers": True,
        "failed_items_are_not_replaced": True,
    }
    for key, value in required_plan.items():
        if plan.get(key) != value:
            raise ValueError(f"execution_plan.{key} conflicts with the workflow contract")
    if plan.get("task_family_schedule", {}).get("order") != list(TASK_FAMILIES):
        raise ValueError("task-family round-robin order mismatch")
    smoke = raw["smoke_run"]
    resolved = len(providers) * len(families) * int(smoke.get("cases_per_task_family", 0))
    if smoke.get("maximum_attempts") != 2 or resolved != 30:
        raise ValueError("smoke must contain two permitted attempts of 30 requests each")
    if smoke.get("budget") != {"maximum_requests_per_attempt": 30, "maximum_cost_usd_across_attempts": 5}:
        raise ValueError("smoke ceilings differ from the workflow contract")
    if raw["operations"].get("full_run_requires_human_approval") is not True:
        raise ValueError("full execution must remain approval-guarded")
    if raw["operations"].get("database_writes_during_smoke") is not False:
        raise ValueError("smoke database writes must be forbidden")


def load_config(path: Path | str = DEFAULT_SPEC) -> BenchmarkConfig:
    spec_path = Path(path).resolve()
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("spec must contain a mapping")
    _validate(raw)
    runner = raw["runner"]
    prices: dict[str, TokenPricing] = {}
    for name, row in raw["providers"]["vendors"].items():
        price = row.get("pricing_usd_per_million_tokens")
        if not isinstance(price, dict) or set(price) != {"input", "output"}:
            raise ValueError(f"{name}: exact input/output pricing is required")
        values = (float(price["input"]), float(price["output"]))
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"{name}: pricing must be positive and finite")
        prices[name] = TokenPricing(*values)
    pricing_hash = hashlib.sha256(
        json.dumps({k: vars(v) for k, v in prices.items()}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BenchmarkConfig(
        spec_path=spec_path,
        raw=raw,
        code_root=PACKAGE_ROOT,
        run_root=project_path(runner["local_run_root"], "runner.local_run_root"),
        dataset_root=project_path(raw["dataset"]["source"], "dataset.source"),
        manifest_path=project_path(raw["dataset"]["manifest"], "dataset.manifest"),
        benchmark_slug=str(raw["benchmark"]["slug"]),
        providers=tuple(raw["providers"]["vendors"]),
        task_families=tuple(raw["dataset"]["task_families"]),
        pricing=prices,
        pricing_hash=pricing_hash,
    )
