"""Network-free validation, run locking, and execution environment handling."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence

from dotenv import dotenv_values

from .config import BenchmarkConfig, load_config
from .dataset import DatasetRelease, load_dataset
from .schedule import build_full_schedule, build_smoke_schedule, flatten, selection_artifact
from .storage import LocalRunStore, atomic_json, content_hash, now_iso, read_json


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def credential_names(config: BenchmarkConfig) -> tuple[str, ...]:
    names: list[str] = []
    for row in config.raw["providers"]["vendors"].values():
        if "api_key" in row:
            names.append(str(row["api_key"]))
        for key in ("proxy_token_id", "proxy_token_secret"):
            if key in row:
                names.append(str(row[key]))
    return tuple(names)


def load_provider_environment(
    config: BenchmarkConfig | None = None,
    *,
    dotenv_path: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    config = config or load_config()
    result = dict(os.environ if environment is None else environment)
    path = Path(dotenv_path).resolve() if dotenv_path is not None else Path.cwd() / ".env"
    if path.is_file():
        allowed = set(credential_names(config))
        for key, value in dotenv_values(path).items():
            if key in allowed and value and not result.get(key):
                result[key] = value
    return result


def missing_provider_credentials(config: BenchmarkConfig, environment: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(name for name in credential_names(config) if not environment.get(name))


def require_provider_credentials(config: BenchmarkConfig, environment: Mapping[str, str]) -> None:
    missing = missing_provider_credentials(config, environment)
    if missing:
        raise RuntimeError("provider execution refused before network calls; missing credentials: " + ", ".join(missing))


def tokenizer_identity() -> dict[str, Any]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required; install openbenchmarks-inference before materializing the run lock"
        ) from exc
    encoding = tiktoken.get_encoding("o200k_base")
    digest = hashlib.sha256()
    for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda row: (row[1], row[0])):
        digest.update(len(token).to_bytes(8, "big"))
        digest.update(token)
        digest.update(int(rank).to_bytes(8, "big"))
    for token, rank in sorted(encoding._special_tokens.items()):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(int(rank).to_bytes(8, "big"))
    return {
        "implementation": "tiktoken",
        "version": importlib.metadata.version("tiktoken"),
        "encoding": encoding.name,
        "vocabulary_sha256": digest.hexdigest(),
    }


def runtime_identity() -> dict[str, Any]:
    return {
        "execution_region": os.environ.get("BENCHMARK_EXECUTION_REGION", "local-unspecified"),
        "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:16],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "monotonic_clock": "time.monotonic_ns",
    }


def _input_hashes(config: BenchmarkConfig, dataset: DatasetRelease) -> dict[str, str]:
    return {_relative(config.spec_path): _sha256(config.spec_path), **dataset.file_hashes}


def _generated_hashes(config: BenchmarkConfig) -> dict[str, str]:
    paths = list(config.code_root.rglob("*.py")) + list(config.code_root.glob("requirements*.txt"))
    return {_relative(path): _sha256(path) for path in sorted(paths)}


def _evaluator_hashes(config: BenchmarkConfig) -> dict[str, str]:
    names = ("evaluation.py", "metrics.py", "reporting.py")
    return {
        _relative(path): _sha256(path)
        for name in names
        if (path := config.code_root / name).is_file()
    }


def _resolved_adapters(config: BenchmarkConfig) -> dict[str, Any]:
    from .providers import PROVIDERS, adapter_metadata, validate_against_spec

    validate_against_spec(config)
    return {
        name: {
            "endpoint": provider.endpoint,
            "model_key": provider.model,
            "credential_environment_names": list(provider.credential_env),
            "auth_style": provider.auth_style,
            "max_tokens_field": provider.max_tokens_field,
            "metadata": adapter_metadata(provider),
        }
        for name, provider in PROVIDERS.items()
    }


def _metric_calculation_identity() -> dict[str, Any]:
    from .metrics import METRICS_VERSION

    return {
        "metrics_implementation_version": METRICS_VERSION,
        "percentile_convention": "nearest-rank",
    }


def _resolved_execution_configuration(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        key: config.raw[key]
        for key in (
            "benchmark",
            "dataset",
            "providers",
            "inference",
            "execution_plan",
            "stream_timing",
            "rate_limit_policy",
            "evaluation",
            "metrics",
            "smoke_run",
            "reporting",
            "operations",
        )
    }


def _allocate_run_id(root: Path, requested: str | None) -> str:
    if requested is not None:
        if not re.fullmatch(r"run-\d{8}-\d{6}", requested):
            raise ValueError("run ID must use run-YYYYMMDD-HHMMSS")
        return requested
    candidate = datetime.now()
    for _ in range(60):
        run_id = candidate.strftime("run-%Y%m%d-%H%M%S")
        if not (root / run_id).exists():
            return run_id
        candidate += timedelta(seconds=1)
    raise RuntimeError("could not allocate a unique run ID")


def prepare_run(
    run_dir: Path | str,
    *,
    config: BenchmarkConfig | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Validate and lock an existing harness directory without making a network call."""
    config = config or load_config()
    dataset = load_dataset(config)
    root = Path(run_dir).resolve()
    try:
        root.relative_to(config.run_root.resolve())
    except ValueError as exc:
        raise ValueError("run directory is outside runner.local_run_root") from exc
    root.mkdir(parents=True, exist_ok=True)
    store = LocalRunStore(root)
    store.initialize()
    selection = selection_artifact(config, dataset)
    store.write_immutable_artifact("artifacts/selection.json", selection)
    full_schedule = build_full_schedule(config, dataset)
    store.write_immutable_artifact(
        "artifacts/full-plan.json",
        {"kind": "full", "expected_rounds": len(full_schedule), "expected_requests": len(flatten(full_schedule)), "rounds": [row.to_dict() for row in full_schedule]},
    )
    tokenizer = tokenizer_identity()
    resolved_adapters = _resolved_adapters(config)
    metric_calculation = _metric_calculation_identity()
    execution_configuration = _resolved_execution_configuration(config)
    lock = {
        "run_id": root.name,
        "benchmark_slug": config.benchmark_slug,
        "created_at": now_iso(),
        "human_input_hashes": _input_hashes(config, dataset),
        "dataset_release_hash": dataset.release_hash,
        "selection_hash": content_hash(selection),
        "generated_code_hashes": _generated_hashes(config),
        "evaluator_code_hashes": _evaluator_hashes(config),
        "resolved_adapters": resolved_adapters,
        "resolved_adapters_sha256": content_hash(resolved_adapters),
        "metric_calculation": metric_calculation,
        "resolved_execution_configuration": execution_configuration,
        "execution_configuration_hash": content_hash(execution_configuration),
        "tokenizer": tokenizer,
        "runtime": runtime_identity(),
        "credentials": {"required_names": list(credential_names(config)), "values_recorded": False},
    }
    existing_lock = read_json(root / "run-lock.json")
    if existing_lock is not None:
        comparable = {key: value for key, value in lock.items() if key != "created_at"}
        old_comparable = {key: value for key, value in existing_lock.items() if key != "created_at"}
        if content_hash(comparable) != content_hash(old_comparable):
            raise RuntimeError("run lock differs from current inputs/code; create a new run or approved amendment")
    else:
        atomic_json(root / "run-lock.json", lock)
    existing_plan = read_json(root / "plan.json", {})
    plan = {
        **existing_plan,
        "run_id": root.name,
        "benchmark": config.benchmark_slug,
        "full_run_authorized": False,
        "database_writes_authorized": False,
        "expected_units": {"smoke_requests_per_attempt": 30, "smoke_rounds_per_attempt": 3, "full_run_requests": 6000, "full_run_rounds": 600},
        "ceilings": {"requests_per_smoke_attempt": config.smoke_request_limit, "cost_usd_across_attempts": config.smoke_cost_limit, "request_timeout_seconds": config.timeout_seconds, "request_attempts": 1},
    }
    atomic_json(root / "plan.json", plan)
    store.event("run_locked", network_calls=0, database_writes=0)
    return root


def initialize_run(
    *,
    spec_path: Path | str | None = None,
    run_id: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    config = load_config(spec_path) if spec_path is not None else load_config()
    config.run_root.mkdir(parents=True, exist_ok=True)
    selected = _allocate_run_id(config.run_root, run_id)
    run_dir = config.run_root / selected
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    run_dir.mkdir()
    return prepare_run(run_dir, config=config, environment=environment)


def create_generated_code_retry_amendment(
    run_dir: Path | str,
    *,
    reason: str,
    source_diagnostics: Sequence[str],
    approver: str = "local-human-review",
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Authorize only exact generated-code deltas for smoke attempt two."""
    if not isinstance(approver, str) or not approver.strip():
        raise PermissionError("generated-code retry amendment requires a non-empty approver")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("generated-code retry amendment requires a reason")
    if not source_diagnostics or not all(isinstance(value, str) and value.strip() for value in source_diagnostics):
        raise ValueError("generated-code retry amendment requires source diagnostics")
    config = config or load_config()
    root = Path(run_dir).resolve()
    lock = read_json(root / "run-lock.json")
    if not isinstance(lock, dict):
        raise RuntimeError("cannot amend a run without run-lock.json")
    attempt_one = read_json(root / "artifacts/run-manifest-attempt-1.json")
    if not isinstance(attempt_one, dict) or attempt_one.get("status") != "smoke_attempt_failed":
        raise RuntimeError("generated-code amendment requires a failed attempt-1 manifest")
    if (root / "checkpoints/smoke-attempt-2.json").exists() or (root / "artifacts/run-manifest-attempt-2.json").exists():
        raise RuntimeError("generated-code amendment must precede smoke attempt 2")
    dataset = load_dataset(config)
    immutable_checks = {
        "human_input_hashes": _input_hashes(config, dataset),
        "dataset_release_hash": dataset.release_hash,
        "tokenizer": tokenizer_identity(),
        "resolved_adapters": _resolved_adapters(config),
        "resolved_adapters_sha256": content_hash(_resolved_adapters(config)),
        "metric_calculation": _metric_calculation_identity(),
        "resolved_execution_configuration": _resolved_execution_configuration(config),
        "execution_configuration_hash": content_hash(_resolved_execution_configuration(config)),
    }
    changed_immutable = [key for key, value in immutable_checks.items() if lock.get(key) != value]
    if changed_immutable:
        raise RuntimeError(
            "generated-code amendment refused; non-code lock fields changed: "
            + ", ".join(changed_immutable)
        )
    old_hashes = lock.get("generated_code_hashes")
    current_hashes = _generated_hashes(config)
    if not isinstance(old_hashes, dict):
        raise RuntimeError("run lock has no generated-code hash map")
    changed_paths = sorted(set(old_hashes) | set(current_hashes))
    changes = {
        path: {
            "change": "added" if path not in old_hashes else "deleted" if path not in current_hashes else "modified",
            "old_sha256": old_hashes.get(path),
            "new_sha256": current_hashes.get(path),
        }
        for path in changed_paths
        if old_hashes.get(path) != current_hashes.get(path)
    }
    if not changes:
        raise RuntimeError("generated-code amendment requires at least one code hash change")
    value = {
        "schema_version": "generated-code-retry-amendment-v1",
        "attempt": 2,
        "created_at": now_iso(),
        "approver": approver,
        "reason": reason.strip(),
        "source_diagnostics": list(source_diagnostics),
        "invalidated_stages": ["evaluation", "metrics", "paired_comparisons", "reporting", "smoke_attempt_2"],
        "preserved_run_lock_sha256": _sha256(root / "run-lock.json"),
        "old_generated_code_hashes": old_hashes,
        "new_generated_code_hashes": current_hashes,
        "generated_code_changes": changes,
        "unchanged_lock_fields": sorted(immutable_checks),
    }
    path = root / "artifacts/generated-code-amendment-attempt-2.json"
    existing = read_json(path)
    if existing is not None:
        stable_fields = {key: item for key, item in value.items() if key != "created_at"}
        existing_stable = {key: item for key, item in existing.items() if key != "created_at"}
        if existing_stable != stable_fields:
            raise RuntimeError("generated-code retry amendment is immutable")
        return existing
    atomic_json(path, value)
    return value


def validate_run_lock(
    run_dir: Path | str,
    config: BenchmarkConfig | None = None,
    *,
    attempt: int | None = None,
    allow_post_processing_code_change: bool = False,
) -> dict[str, Any]:
    config = config or load_config()
    lock = read_json(Path(run_dir) / "run-lock.json")
    if not isinstance(lock, dict):
        raise RuntimeError("missing run-lock.json; materialize the run before execution")
    dataset = load_dataset(config)
    current_generated_hashes = _generated_hashes(config)
    checks = {
        "human_input_hashes": _input_hashes(config, dataset),
        "dataset_release_hash": dataset.release_hash,
        "resolved_adapters": _resolved_adapters(config),
        "resolved_adapters_sha256": content_hash(_resolved_adapters(config)),
        "metric_calculation": _metric_calculation_identity(),
        "resolved_execution_configuration": _resolved_execution_configuration(config),
        "execution_configuration_hash": content_hash(_resolved_execution_configuration(config)),
        "tokenizer": tokenizer_identity(),
    }
    for key, value in checks.items():
        if lock.get(key) != value:
            raise RuntimeError(f"run lock validation failed: {key} changed")
    if (
        lock.get("generated_code_hashes") != current_generated_hashes
        and allow_post_processing_code_change
    ):
        return lock
    if lock.get("generated_code_hashes") != current_generated_hashes:
        amendment = read_json(Path(run_dir) / "artifacts/generated-code-amendment-attempt-2.json")
        if attempt != 2 or not isinstance(amendment, dict):
            raise RuntimeError("run lock validation failed: generated_code_hashes changed")
        if (
            amendment.get("schema_version") != "generated-code-retry-amendment-v1"
            or amendment.get("attempt") != 2
            or not isinstance(amendment.get("approver"), str)
            or not amendment.get("approver", "").strip()
            or amendment.get("old_generated_code_hashes") != lock.get("generated_code_hashes")
            or amendment.get("new_generated_code_hashes") != current_generated_hashes
            or amendment.get("preserved_run_lock_sha256") != _sha256(Path(run_dir) / "run-lock.json")
        ):
            raise RuntimeError("run lock validation failed: invalid generated-code retry amendment")
        evaluator_paths = set(lock.get("evaluator_code_hashes", {}))
        for path in evaluator_paths:
            if lock["generated_code_hashes"].get(path) != lock["evaluator_code_hashes"].get(path):
                raise RuntimeError("run lock evaluator hashes are internally inconsistent")
    elif lock.get("evaluator_code_hashes") != _evaluator_hashes(config):
        raise RuntimeError("run lock validation failed: evaluator_code_hashes changed")
    return lock


__all__ = [
    "create_generated_code_retry_amendment", "credential_names", "initialize_run", "load_provider_environment", "missing_provider_credentials",
    "prepare_run", "require_provider_credentials", "runtime_identity", "tokenizer_identity", "validate_run_lock",
]
