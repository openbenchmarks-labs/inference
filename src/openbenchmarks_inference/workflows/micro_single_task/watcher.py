"""Network-free watcher reconstructed exclusively from durable local state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .storage import read_json, read_jsonl


def snapshot(run_dir: Path | str) -> dict[str, Any]:
    root = Path(run_dir)
    plan = read_json(root / "plan.json", {})
    launcher = read_json(root / "launcher.json", {})
    states = [read_json(path, {}) for path in sorted((root / "checkpoints").glob("smoke-attempt-*.json"))]
    attempt = max((int(state["attempt"]) for state in states if "attempt" in state), default=None)
    full_state = read_json(root / "checkpoints/full-run.json", {})
    full_active = isinstance(full_state, dict) and full_state.get("state") in {"running", "complete"}
    if full_active:
        attempt = None
    suffix = f"-attempt-{attempt}" if attempt is not None else ""
    rows = read_jsonl(root / "raw" / f"results{suffix}.jsonl")
    if attempt is not None:
        atomic_rows = [read_json(path) for path in (root / "raw" / f"attempt-{attempt}").glob("*.json")]
        by_id = {row["request_id"]: row for row in rows if isinstance(row, dict) and row.get("request_id")}
        by_id.update({row["request_id"]: row for row in atomic_rows if isinstance(row, dict) and row.get("request_id")})
        rows = list(by_id.values())
    expected = 30 if attempt is not None else plan.get("expected_units", {}).get("full_run_requests", 6000)
    external_calls = sum(1 for event in read_jsonl(root / "events.jsonl") if event.get("kind") == "external_call_started")
    failures = [row for row in rows if row.get("terminal_outcome") != "completed"]
    elapsed = None
    current_state = next((state for state in states if state.get("attempt") == attempt), {})
    if current_state.get("state") == "complete":
        elapsed = current_state.get("duration_seconds")
    elif isinstance(current_state.get("active_segment_started_epoch_seconds"), (int, float)):
        elapsed = float(current_state.get("active_elapsed_seconds", 0)) + time.time() - float(current_state["active_segment_started_epoch_seconds"])
    elif full_active:
        elapsed_values = [
            row.get("transport_elapsed_ms")
            for row in rows
            if isinstance(row.get("transport_elapsed_ms"), (int, float))
        ]
        elapsed = sum(elapsed_values) / 1000 if elapsed_values else None
    durations = [row.get("e2e_latency_ms") for row in rows if isinstance(row.get("e2e_latency_ms"), (int, float))]
    remaining = max(0, int(expected) - len(rows))
    eta = (sum(durations) / len(durations) / 1000 * remaining) if durations else None
    budget = read_json(root / "artifacts/smoke-budget.json", {})
    invocations = launcher.get("invocations", []) if isinstance(launcher, dict) else []
    latest = invocations[-1] if invocations else {}
    manifest = read_json(root / "artifacts/run-manifest.json", {})
    if not manifest and attempt is not None:
        manifest = read_json(root / "artifacts" / f"run-manifest-attempt-{attempt}.json", {})
    attempt_summaries = [
        {
            "attempt": state.get("attempt"),
            "state": state.get("state"),
            "success": state.get("success"),
            "duration_seconds": state.get("duration_seconds"),
            "completed": len(read_jsonl(root / "raw" / f"results-attempt-{state.get('attempt')}.jsonl")),
            "expected": 30,
        }
        for state in states
    ]
    return {
        "parent": {"agent": latest.get("agent"), "model": latest.get("model"), "mode": latest.get("mode")},
        "active_subagents": [row.get("owner") for row in plan.get("task_assignments", []) if row.get("status") == "in_progress"],
        "stage": manifest.get("status") or ("full_run" if full_active else plan.get("stage", next((row.get("name") for row in plan.get("stages", []) if row.get("status") == "in_progress"), "unknown"))),
        "smoke_attempt": attempt,
        "smoke_attempts": attempt_summaries,
        "completed": len(rows),
        "remaining": remaining,
        "external_calls": external_calls,
        "local_writes": sum(1 for path in root.rglob("*") if path.is_file()),
        "elapsed_seconds": elapsed,
        "rolling_eta_seconds": eta,
        "spend": {"reserved_cost_usd": budget.get("reserved_cost_usd"), "actual_cost_usd": budget.get("actual_cost_usd"), "known_actual_cost_usd": budget.get("known_actual_cost_usd")},
        "recent_failures": [{key: row.get(key) for key in ("request_id", "provider", "task_family", "terminal_outcome", "http_status", "error")} for row in failures[-5:]],
        "latest_artifacts": sorted(str(path.relative_to(root)) for path in (root / "artifacts").rglob("*") if path.is_file()),
    }


def watch(run_dir: Path | str, *, interval: float = 2.0, once: bool = False) -> None:
    while True:
        print(json.dumps(snapshot(run_dir), indent=2, sort_keys=True), flush=True)
        if once:
            return
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    watch(args.run_dir, interval=args.interval, once=args.once)
    return 0
