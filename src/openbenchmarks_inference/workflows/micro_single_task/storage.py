"""Atomic local source-of-truth artifacts and resume checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable


TERMINAL_OUTCOMES = {"completed", "timeout", "http_error", "transport_error", "runner_error"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt JSONL {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL record {path}:{number}")
        rows.append(value)
    return rows


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def _known_failed_stream_shell(record: dict[str, Any]) -> bool:
    """Recognize the historical transport failure that lost embedded events."""

    error = record.get("error")
    return (
        record.get("terminal_outcome") == "runner_error"
        and record.get("http_status") is None
        and isinstance(error, dict)
        and error.get("type") == "AttributeError"
        and error.get("message") == "'NoneType' object has no attribute 'read'"
        and record.get("stream_events") == []
        and record.get("reconstructed_answer", "") == ""
    )


def _immutable(path: Path, value: Any) -> None:
    existing = read_json(path)
    if existing is not None:
        if content_hash(existing) != content_hash(value):
            raise RuntimeError(f"immutable artifact changed: {path}")
        return
    atomic_json(path, value)


@dataclass
class SmokeBudget:
    path: Path
    maximum_requests_per_attempt: int
    maximum_cost_usd_across_attempts: float
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def _state(self) -> dict[str, Any]:
        return read_json(self.path, {"attempts": {}, "reserved_cost_usd": 0.0, "known_actual_cost_usd": 0.0})

    def reserve(self, attempt: int, request_id: str, projected_cost_usd: float) -> None:
        if isinstance(projected_cost_usd, bool) or not isinstance(projected_cost_usd, (int, float)) or not math.isfinite(projected_cost_usd) or projected_cost_usd < 0:
            raise ValueError("projected cost must be finite and non-negative")
        with self._lock:
            state = self._state()
            reservations = state.setdefault("attempts", {}).setdefault(str(attempt), {}).setdefault("reservations", {})
            if request_id in reservations:
                return
            if len(reservations) >= self.maximum_requests_per_attempt:
                raise RuntimeError("smoke request ceiling reached")
            total = float(state.get("reserved_cost_usd", 0.0)) + float(projected_cost_usd)
            if total > self.maximum_cost_usd_across_attempts:
                raise RuntimeError("shared smoke cost ceiling would be exceeded")
            reservations[request_id] = float(projected_cost_usd)
            state["reserved_cost_usd"] = total
            atomic_json(self.path, state)

    def observe(self, attempt: int, request_id: str, cost_usd: float | None) -> None:
        if cost_usd is not None and (isinstance(cost_usd, bool) or not isinstance(cost_usd, (int, float)) or not math.isfinite(cost_usd) or cost_usd < 0):
            raise ValueError("observed cost must be finite, non-negative, or null")
        with self._lock:
            state = self._state()
            observations = state.setdefault("attempts", {}).setdefault(str(attempt), {}).setdefault("observations", {})
            observations[request_id] = cost_usd
            values = [v for attempt_state in state["attempts"].values() for v in attempt_state.get("observations", {}).values() if isinstance(v, (int, float))]
            state["known_actual_cost_usd"] = sum(values)
            state["actual_cost_complete"] = all(v is not None for attempt_state in state["attempts"].values() for v in attempt_state.get("observations", {}).values())
            state["actual_cost_usd"] = sum(values) if state["actual_cost_complete"] else None
            atomic_json(self.path, state)


class LocalRunStore:
    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir).resolve()
        self.raw_dir = self.run_dir / "raw"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.artifact_dir = self.run_dir / "artifacts"
        self._lock = threading.RLock()

    def initialize(self) -> None:
        for path in (self.raw_dir, self.checkpoint_dir, self.artifact_dir):
            path.mkdir(parents=True, exist_ok=True)
        if not (self.run_dir / "events.jsonl").exists():
            atomic_text(self.run_dir / "events.jsonl", "")
        if not (self.run_dir / "report.md").exists():
            atomic_text(self.run_dir / "report.md", "# Benchmark run\n\nInitialized; no benchmark requests have completed.\n")

    def event(self, kind: str, **fields: Any) -> None:
        row = {"at": now_iso(), "kind": kind, **fields}
        path = self.run_dir / "events.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_immutable_artifact(self, relative_path: str, value: Any) -> Path:
        path = self.run_dir / relative_path
        with self._lock:
            _immutable(path, value)
        return path

    def begin_attempt(self, attempt: int) -> dict[str, Any]:
        path = self.checkpoint_dir / f"smoke-attempt-{attempt}.json"
        with self._lock:
            state = read_json(path)
            if isinstance(state, dict) and state.get("state") == "complete":
                raise RuntimeError(f"smoke attempt {attempt} is already complete")
            now = time.time()
            if state is None:
                state = {"attempt": attempt, "state": "running", "started_at": now_iso(), "active_elapsed_seconds": 0.0}
            state["active_segment_started_epoch_seconds"] = now
            state["resumed_at"] = now_iso()
            atomic_json(path, state)
            return state

    def complete_attempt(self, attempt: int, success: bool) -> dict[str, Any]:
        path = self.checkpoint_dir / f"smoke-attempt-{attempt}.json"
        with self._lock:
            state = read_json(path)
            if not isinstance(state, dict) or state.get("state") != "running":
                raise RuntimeError(f"smoke attempt {attempt} is not running")
            started = float(state.pop("active_segment_started_epoch_seconds", time.time()))
            elapsed = float(state.get("active_elapsed_seconds", 0.0)) + max(0.0, time.time() - started)
            state.update({"state": "complete", "success": bool(success), "completed_at": now_iso(), "duration_seconds": elapsed})
            atomic_json(path, state)
            return state

    def _partition(self, attempt: int | None) -> str:
        return f"attempt-{attempt}" if attempt is not None else "full"

    def _raw_path(self, request_id: str, attempt: int | None) -> Path:
        return self.raw_dir / self._partition(attempt) / f"{request_id}.json"

    def write_terminal(self, record: dict[str, Any], *, attempt: int | None) -> None:
        request_id = record.get("request_id")
        outcome = record.get("terminal_outcome", record.get("outcome_category"))
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("terminal record requires request_id")
        if outcome not in TERMINAL_OUTCOMES:
            raise ValueError(f"record is not terminal: {outcome}")
        path = self._raw_path(request_id, attempt)
        checkpoint = self.checkpoint_dir / self._partition(attempt) / "requests" / f"{request_id}.json"
        with self._lock:
            _immutable(path, record)
            _immutable(checkpoint, {"request_id": request_id, "terminal": True, "terminal_outcome": outcome, "record_hash": content_hash(record)})

    def write_stream_event(
        self,
        request_id: str,
        event: dict[str, Any],
        *,
        attempt: int | None,
    ) -> None:
        """Durably record one parsed SSE event at its arrival boundary."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("stream event requires request_id")
        index = event.get("event_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("stream event requires a non-negative event_index")
        value = {
            **event,
            "request_id": request_id,
            "event_index": index,
            "durability": "arrival_atomic",
        }
        path = self.raw_dir / "stream-events" / self._partition(attempt) / request_id / f"{index:08d}.json"
        with self._lock:
            _immutable(path, value)

    def stream_events(self, request_id: str, *, attempt: int | None) -> list[dict[str, Any]]:
        path = self.raw_dir / "stream-events" / self._partition(attempt) / request_id
        return [row for item in sorted(path.glob("*.json")) if isinstance((row := read_json(item)), dict)]

    def terminal_records(self, attempt: int | None) -> list[dict[str, Any]]:
        path = self.raw_dir / self._partition(attempt)
        return [read_json(item) for item in sorted(path.glob("*.json")) if isinstance(read_json(item), dict)]

    def terminal_ids(self, attempt: int | None) -> set[str]:
        return {str(row["request_id"]) for row in self.terminal_records(attempt)}

    def write_evaluation(self, value: dict[str, Any], *, attempt: int | None) -> None:
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("evaluation requires request_id")
        with self._lock:
            _immutable(self.artifact_dir / "evaluations" / self._partition(attempt) / f"{request_id}.json", value)

    def materialize_attempt(self, *, attempt: int | None, ordered_request_ids: Iterable[str]) -> dict[str, Any]:
        partition = self._partition(attempt)
        by_id = {row["request_id"]: row for row in self.terminal_records(attempt)}
        order = list(ordered_request_ids)
        round_skew = {
            str(row["round_id"]): row.get("launch_skew_ms")
            for path in (self.artifact_dir / "rounds" / partition).glob("*.json")
            if isinstance((row := read_json(path)), dict) and row.get("round_id") is not None
        }
        records = [
            {
                **by_id[request_id],
                **(
                    {"launch_skew_ms": round_skew[str(by_id[request_id]["round_id"])]}
                    if str(by_id[request_id].get("round_id")) in round_skew
                    else {}
                ),
            }
            for request_id in order
            if request_id in by_id
        ]
        evaluations_by_id = {
            row["request_id"]: row
            for path in (self.artifact_dir / "evaluations" / partition).glob("*.json")
            if isinstance((row := read_json(path)), dict) and isinstance(row.get("request_id"), str)
        }
        evaluations = [evaluations_by_id[request_id] for request_id in order if request_id in evaluations_by_id]
        suffix = f"-attempt-{attempt}" if attempt is not None else ""
        atomic_jsonl(self.raw_dir / f"results{suffix}.jsonl", records)
        request_rows: list[dict[str, Any]] = []
        for row in records:
            request = row.get("request")
            request_rows.append(
                {
                    **(request if isinstance(request, dict) else {}),
                    **{key: row.get(key) for key in ("request_id", "provider", "task_family", "item_id", "round_id", "round_index", "launch_skew_ms")},
                }
            )
        atomic_jsonl(self.raw_dir / f"requests{suffix}.jsonl", request_rows)
        stream_rows: list[dict[str, Any]] = []
        reconstructed: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for row in records:
            durable_events = self.stream_events(row["request_id"], attempt=attempt)
            embedded_events = row.get("stream_events", [])
            if durable_events:
                without_marker = [
                    {key: value for key, value in event.items() if key != "durability"}
                    for event in durable_events
                ]
                normalized_embedded = [
                    {
                        **event,
                        "request_id": row["request_id"],
                        "event_index": event.get("event_index", index),
                    }
                    for index, event in enumerate(embedded_events)
                    if isinstance(event, dict)
                ]
                if (
                    without_marker != normalized_embedded
                    and not _known_failed_stream_shell(row)
                ):
                    raise RuntimeError(f"atomic stream events differ from terminal record: {row['request_id']}")
                stream_rows.extend(durable_events)
            else:
                for index, event in enumerate(embedded_events):
                    stream_rows.append({
                        **(event if isinstance(event, dict) else {"payload": event}),
                        "request_id": row["request_id"],
                        "event_index": event.get("event_index", index) if isinstance(event, dict) else index,
                        "durability": "terminal_embedded",
                    })
            reconstructed.append({
                "request_id": row["request_id"], "provider": row.get("provider"), "task_family": row.get("task_family"),
                "item_id": row.get("item_id"), "reasoning": row.get("reconstructed_reasoning", ""),
                "answer": row.get("reconstructed_answer", row.get("answer", "")),
            })
            if row.get("terminal_outcome") != "completed":
                failures.append(row)
        atomic_jsonl(self.raw_dir / f"stream-events{suffix}.jsonl", stream_rows)
        atomic_jsonl(self.raw_dir / f"reconstructed-responses{suffix}.jsonl", reconstructed)
        atomic_jsonl(self.artifact_dir / f"evaluations{suffix}.jsonl", evaluations)
        atomic_jsonl(self.artifact_dir / f"failures{suffix}.jsonl", failures)
        return {"records": records, "evaluations": evaluations, "stream_events": stream_rows, "reconstructed": reconstructed, "failures": failures}


def artifact_inventory(run_dir: Path | str) -> dict[str, dict[str, Any]]:
    root = Path(run_dir)
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and not candidate.name.startswith(".")):
        relative = str(path.relative_to(root))
        output[relative] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return output
