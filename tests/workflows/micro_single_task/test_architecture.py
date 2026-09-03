from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest

from openbenchmarks_inference.workflows.micro_single_task.config import load_config
from openbenchmarks_inference.workflows.micro_single_task.dataset import load_dataset
from openbenchmarks_inference.workflows.micro_single_task import runner as runner_module
from openbenchmarks_inference.workflows.micro_single_task import lifecycle as lifecycle_module
from openbenchmarks_inference.workflows.micro_single_task.runner import FULL_SUCCESS_STATUS, BenchmarkRunner, _configuration_evidence, _event_channel_error, _recompute_evidence, run_full
from openbenchmarks_inference.workflows.micro_single_task.lifecycle import create_generated_code_retry_amendment, prepare_run, validate_run_lock
from openbenchmarks_inference.workflows.micro_single_task.schedule import (
    SCHEDULE_VERSION,
    build_full_schedule,
    build_smoke_schedule,
    family_order,
    flatten,
    selection_artifact,
)
from openbenchmarks_inference.workflows.micro_single_task.storage import LocalRunStore, SmokeBudget, read_jsonl
from openbenchmarks_inference.workflows.micro_single_task.watcher import snapshot


@pytest.fixture(scope="module")
def config(synthetic_config):
    return synthetic_config


@pytest.fixture(scope="module")
def dataset(config):
    return load_dataset(config)


def test_dataset_and_selection_are_frozen_and_deterministic(config, dataset):
    assert dataset.total_items == 721
    assert {family: len(rows) for family, rows in dataset.by_family.items()} == {
        "meeting-notes-lookup": 237,
        "ticket-triage": 244,
        "contract-terms-extraction": 240,
    }
    selection = selection_artifact(config, dataset)
    assert selection["algorithm_version"] == SCHEDULE_VERSION
    assert selection["selected_items"] == 600
    for family in config.task_families:
        row = selection["families"][family]
        assert tuple(row["full_family_order"]) == family_order(config, dataset, family)
        assert len(row["selected_prefix"]) == len(row["selected_item_hashes"]) == 200


def test_full_schedule_is_600_paired_rounds_with_rotation(config, dataset):
    rounds = build_full_schedule(config, dataset)
    assert len(rounds) == 600
    assert len(flatten(rounds)) == 6000
    assert [row.task_family for row in rounds[:6]] == [*config.task_families, *config.task_families]
    assert all(len(row.units) == 10 for row in rounds)
    assert len({unit.request_id for unit in flatten(rounds)}) == 6000
    assert [unit.provider for unit in rounds[1].units] == [*config.providers[1:], config.providers[0]]
    assert all({unit.item_id for unit in row.units} == {row.item_id} for row in rounds)


def test_smoke_has_three_rounds_and_attempt_ids_do_not_collide(config, dataset):
    first = build_smoke_schedule(config, dataset, 1)
    second = build_smoke_schedule(config, dataset, 2)
    assert len(first) == len(second) == 3
    assert [(row.task_family, row.item_id) for row in first] == [(row.task_family, row.item_id) for row in second]
    assert {unit.request_id for unit in flatten(first)}.isdisjoint({unit.request_id for unit in flatten(second)})


def test_atomic_terminal_resume_and_materialization(tmp_path):
    store = LocalRunStore(tmp_path)
    store.initialize()
    record = {
        "request_id": "req-a",
        "provider": "provider",
        "task_family": "family",
        "item_id": "item",
        "round_id": "round",
        "terminal_outcome": "completed",
        "http_status": 200,
        "reconstructed_answer": "{}",
        "stream_events": [{"classification": "answer", "content": "{}", "monotonic_offset_ms": 1}],
    }
    store.write_terminal(record, attempt=1)
    store.write_terminal(record, attempt=1)
    store.write_evaluation({"request_id": "req-a", "schema_valid": True}, attempt=1)
    assert store.terminal_ids(1) == {"req-a"}
    materialized = store.materialize_attempt(attempt=1, ordered_request_ids=["req-a"])
    assert materialized["records"] == [record]
    assert len(read_jsonl(tmp_path / "raw/stream-events-attempt-1.jsonl")) == 1
    with pytest.raises(RuntimeError, match="immutable artifact changed"):
        store.write_terminal({**record, "http_status": 201}, attempt=1)


def test_budget_enforces_request_and_shared_cost_ceilings(tmp_path):
    budget = SmokeBudget(tmp_path / "budget.json", 1, 0.5)
    budget.reserve(1, "a", 0.25)
    budget.reserve(1, "a", 0.25)
    with pytest.raises(RuntimeError, match="request ceiling"):
        budget.reserve(1, "b", 0.1)
    with pytest.raises(RuntimeError, match="cost ceiling"):
        budget.reserve(2, "c", 0.3)


def test_shared_release_barrier_and_resume_skip(config, dataset, tmp_path):
    rounds = build_smoke_schedule(config, dataset, 1)
    active = 0
    peak = 0
    calls: list[str] = []

    class FakeTransport:
        async def execute_request(self, *, unit, item):
            nonlocal active, peak
            calls.append(unit.request_id)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1
            return {
                "terminal_outcome": "completed",
                "http_status": 200,
                "reconstructed_answer": "{}",
                "stream_events": [{"classification": "answer", "content": "{}", "monotonic_offset_ms": 1}],
            }

    store = LocalRunStore(tmp_path)
    store.initialize()
    first_unit = rounds[0].units[0]
    store.write_terminal({
        "request_id": first_unit.request_id,
        "round_id": first_unit.round_id,
        "round_index": first_unit.round_index,
        "provider": first_unit.provider,
        "provider_launch_index": first_unit.provider_launch_index,
        "task_family": first_unit.task_family,
        "item_id": first_unit.item_id,
        "terminal_outcome": "timeout",
        "http_status": None,
        "launch_skew_ms": 0.0,
        "reconstructed_answer": "",
        "stream_events": [],
    }, attempt=1)
    runner = BenchmarkRunner(tmp_path, config=config, dataset=dataset, transport=FakeTransport())
    records = asyncio.run(runner.execute_schedule(rounds[:1], attempt=1))
    assert len(records) == 10
    assert first_unit.request_id not in calls
    assert len(calls) == 9
    assert peak == 9
    assert all(
        "launch_skew_ms" not in row
        for row in records
        if row["request_id"] != first_unit.request_id
    )
    runner.store.materialize_attempt(
        attempt=1,
        ordered_request_ids=[unit.request_id for unit in rounds[0].units],
    )
    assert all(
        "launch_skew_ms" in row
        for row in read_jsonl(tmp_path / "raw/results-attempt-1.jsonl")
    )


def test_fast_terminal_is_checkpointed_while_round_peer_is_in_flight(config, dataset, tmp_path):
    question = build_smoke_schedule(config, dataset, 1)[0]
    fast_id = question.units[0].request_id
    slow_id = question.units[-1].request_id
    fast_done = asyncio.Event()
    release_slow = asyncio.Event()

    class UnevenTransport:
        async def execute_request(self, *, unit, item):
            if unit.request_id == slow_id:
                await release_slow.wait()
            elif unit.request_id == fast_id:
                fast_done.set()
            return {
                "terminal_outcome": "completed",
                "http_status": 200,
                "reconstructed_answer": "{}",
                "stream_events": [],
            }

    runner = BenchmarkRunner(
        tmp_path, config=config, dataset=dataset, transport=UnevenTransport()
    )

    async def scenario():
        execution = asyncio.create_task(
            runner.execute_schedule((question,), attempt=1)
        )
        await asyncio.wait_for(fast_done.wait(), timeout=1)
        checkpoint = (
            tmp_path
            / "checkpoints"
            / "attempt-1"
            / "requests"
            / f"{fast_id}.json"
        )
        for _ in range(100):
            if checkpoint.is_file():
                break
            await asyncio.sleep(0.001)
        assert checkpoint.is_file()
        assert not execution.done()
        release_slow.set()
        await execution

    asyncio.run(scenario())


def test_all_requests_prepared_before_release_and_events_are_arrival_atomic(config, dataset, tmp_path):
    question = build_smoke_schedule(config, dataset, 1)[0]
    prepared: list[str] = []
    executed: list[str] = []

    class PreparedTransport:
        def prepare_request(self, *, unit, item):
            assert not executed
            prepared.append(unit.request_id)
            return (unit, item)

        async def execute_prepared(self, *, prepared: tuple, event_sink):
            unit, _ = prepared
            assert len(prepared_ids) == 10
            event = {
                "event_index": 0,
                "classification": "answer",
                "answer_content": "{}",
                "content": "{}",
                "monotonic_offset_ms": 1.0,
            }
            event_sink(event)
            executed.append(unit.request_id)
            return {
                "terminal_outcome": "completed",
                "http_status": 200,
                "reconstructed_answer": "{}",
                "stream_events": [event],
            }

    # Keep a differently named alias so the execute argument named `prepared`
    # cannot obscure the outer evidence list.
    prepared_ids = prepared
    runner = BenchmarkRunner(tmp_path, config=config, dataset=dataset, transport=PreparedTransport())
    asyncio.run(runner.execute_schedule((question,), attempt=1))
    assert len(prepared_ids) == len(executed) == 10
    materialized = runner.store.materialize_attempt(
        attempt=1, ordered_request_ids=[unit.request_id for unit in question.units]
    )
    assert all(event["durability"] == "arrival_atomic" for event in materialized["stream_events"])


def test_pending_novita_mapping_needs_genuine_low_effort_evidence(config):
    expected = config.raw["providers"]["vendors"]["novita"]["model_key"]
    record = {
        "provider": "novita",
        "provider_model": expected,
        "model_identity": {"requested": expected, "reported": expected, "matches": True},
        "adapter_metadata": {
            "model_key": expected,
            "protocol": "openai-chat-completions",
            "endpoint_class": "serverless_or_shared",
            "streaming": True,
            "structured_output_mode": "json_object",
            "schema_enforcement": "local_strict",
            "max_attempts": 1,
            "reasoning_effort": "low",
            "reasoning_wire_mapping": {"reasoning_effort": "low"},
            "reasoning_docs_status": "pending_smoke",
        },
        "request": {"body": {
            "model": expected,
            "stream": True,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "max_completion_tokens": 256,
            "reasoning_effort": "low",
        }},
        "adapter_compatibility": {"status": "smoke_validated"},
    }
    assert _configuration_evidence(config, record)["compatible"] is False
    record["adapter_compatibility"]["exact_low_reasoning_evidenced"] = True
    assert _configuration_evidence(config, record)["compatible"] is True


def test_operational_failure_uses_request_configuration_evidence(config):
    expected = config.raw["providers"]["vendors"]["nebius"]["model_key"]
    record = {
        "provider": "nebius",
        "provider_model": expected,
        "terminal_outcome": "timeout",
        "http_status": None,
        "adapter_metadata": {
            "model_key": expected,
            "protocol": "openai-chat-completions",
            "endpoint_class": "serverless_or_shared",
            "streaming": True,
            "structured_output_mode": "json_object",
            "schema_enforcement": "local_strict",
            "max_attempts": 1,
            "reasoning_effort": "low",
            "reasoning_wire_mapping": {"reasoning_effort": "low"},
            "reasoning_docs_status": "documented",
        },
        "request": {"body": {
            "model": expected,
            "stream": True,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "max_completion_tokens": 256,
            "reasoning_effort": "low",
        }},
    }
    assert _configuration_evidence(config, record)["compatible"] is True
    record["request"]["body"]["model"] = "wrong-model"
    evidence = _configuration_evidence(config, record)
    assert evidence["compatible"] is False
    assert "request_model_matches_spec" in evidence["reasons"]


def test_recomputation_accepts_distinct_channels_in_one_sse_event(config, dataset):
    from openbenchmarks_inference.workflows.micro_single_task.evaluation import evaluate_record
    from openbenchmarks_inference.workflows.micro_single_task.metrics import compute_metrics, compute_paired_comparisons

    item = dataset.by_family["meeting-notes-lookup"][0]
    answer = '{"answer":"' + item.gold["value"] + '"}'
    event = {
        "request_id": "req-combined",
        "event_index": 0,
        "classification": "reasoning_and_answer",
        "reasoning_content": "r",
        "answer_content": answer,
        "content": "r",
        "content_presence": {"reasoning": True, "answer": True},
        "redacted_payload": {
            "choices": [{"delta": {"reasoning_content": "r", "content": answer}}]
        },
        "monotonic_offset_ms": 1.0,
    }
    import tiktoken
    count = len(tiktoken.get_encoding("o200k_base").encode(answer))
    record = {
        "request_id": "req-combined",
        "provider": "Z.AI",
        "task_family": item.task_family,
        "item_id": item.id,
        "round_id": "round-1",
        "round_index": 0,
        "terminal_outcome": "completed",
        "http_status": 200,
        "reconstructed_reasoning": "r",
        "reconstructed_answer": answer,
        "stream_events": [event],
        "timings": {
            "ttfa_ms": 1.0,
            "e2e_latency_ms": 1.0,
            "first_answer_token_ms": 1.0,
            "last_answer_token_ms": 1.0,
            "visible_answer_token_count": count,
            "output_tokens_per_second": None,
        },
    }
    evaluation = evaluate_record(item, record)
    compatibility = {"Z.AI": {"compatible": True}}
    completeness = {"Z.AI": {"complete": True}}
    metrics = compute_metrics(
        [record], [evaluation], benchmark_slug=config.benchmark_slug, run_id="run-test",
        provider_compatibility=compatibility, artifact_completeness=completeness,
        expected_task_families=config.task_families,
    )
    paired = compute_paired_comparisons(
        [record], benchmark_slug=config.benchmark_slug, run_id="run-test",
        provider_compatibility=compatibility, artifact_completeness=completeness,
    )
    evidence = _recompute_evidence(
        config=config,
        items={item.id: item},
        records=[record],
        evaluations=[evaluation],
        materialized_stream_events=[{**event, "durability": "arrival_atomic"}],
        metrics=metrics,
        paired=paired,
        run_id="run-test",
        provider_compatibility=compatibility,
        artifact_completeness=completeness,
    )
    assert evidence["channel_separation_valid"] is True
    assert evidence["evaluations_match"] is True
    assert evidence["metrics_match"] is True


def test_channel_validation_rejects_conflation_and_misclassification():
    payload = {"choices": [{"delta": {"reasoning_content": "think", "content": "answer"}}]}
    valid = {
        "classification": "reasoning_and_answer",
        "reasoning_content": "think",
        "answer_content": "answer",
        "content_presence": {"reasoning": True, "answer": True},
        "redacted_payload": payload,
    }
    assert _event_channel_error(valid) is None
    assert _event_channel_error({**valid, "classification": "answer"}) == "content-bearing event is misclassified"
    conflated = {
        **valid,
        "reasoning_content": "",
        "classification": "answer",
        "content_presence": {"reasoning": False, "answer": True},
    }
    assert _event_channel_error(conflated) == "payload channels were conflated or dropped"


@pytest.mark.parametrize(
    ("provider", "reasoning_field"),
    (("parasail", "reasoning"), ("deepinfra", "reasoning_content")),
)
def test_channel_validation_accepts_observed_provider_reasoning_aliases(provider, reasoning_field):
    event = {
        "request_id": f"req-{provider}",
        "event_index": 3,
        "classification": "reasoning_and_answer",
        "reasoning_content": "normalized thought",
        "answer_content": "{",
        "content": "normalized thought",
        "content_presence": {"reasoning": True, "answer": True},
        "redacted_payload": {
            "id": f"provider-response-{provider}",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        reasoning_field: "normalized thought",
                        "content": "{",
                    },
                }
            ],
        },
        "monotonic_offset_ms": 123.5,
    }
    assert _event_channel_error(event) is None


def test_full_path_is_explicitly_guarded(tmp_path):
    with pytest.raises(PermissionError, match="approve-full"):
        asyncio.run(run_full(tmp_path))


def test_approved_full_path_invokes_complete_derivation(config, dataset, tmp_path, monkeypatch):
    called: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def execute_schedule(self, rounds, *, attempt):
            called["execute"] = (len(rounds), attempt)

    async def fake_derive(runner, rounds, *, attempt):
        called["derive"] = (len(rounds), attempt)
        return {"success": True, "manifest": {"kind": "full"}}

    monkeypatch.setattr(runner_module, "load_config", lambda: config)
    monkeypatch.setattr(runner_module, "load_dataset", lambda _config: dataset)
    monkeypatch.setattr(runner_module, "validate_run_lock", lambda *_args: {})
    monkeypatch.setattr(runner_module, "require_provider_credentials", lambda *_args: None)
    monkeypatch.setattr(runner_module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(runner_module, "_derive_and_report", fake_derive)
    result = asyncio.run(run_full(
        tmp_path,
        approve_full=True,
        approval_id="human-approval-1",
        transport=object(),
        environment={},
    ))
    assert result["success"] is True
    assert called == {"execute": (600, None), "derive": (600, None)}
    assert FULL_SUCCESS_STATUS == "full_run_completed_awaiting_human_review"


def test_watcher_reads_only_local_state(tmp_path):
    store = LocalRunStore(tmp_path)
    store.initialize()
    (tmp_path / "plan.json").write_text('{"stage":"test","expected_units":{"full_run_requests":6000}}')
    value = snapshot(tmp_path)
    assert value["stage"] == "test"
    assert value["completed"] == 0
    assert value["remaining"] == 6000


def test_benchmark_local_requirements_pin_tokenizer():
    path = Path(__file__).parents[3] / "pyproject.toml"
    assert "tiktoken>=" in path.read_text(encoding="utf-8")


def test_run_lock_freezes_exact_adapters_metric_calculation_and_execution(config, tmp_path):
    local_config = replace(config, run_root=tmp_path)
    run_dir = tmp_path / "run-20260902-120000"
    prepare_run(run_dir, config=local_config)
    lock = json.loads((run_dir / "run-lock.json").read_text(encoding="utf-8"))
    assert lock["metric_calculation"] == {
        "metrics_implementation_version": "micro-single-task-metrics-v5",
        "percentile_convention": "nearest-rank",
    }
    assert set(lock["resolved_adapters"]) == set(config.providers)
    assert lock["resolved_adapters"]["novita"]["metadata"]["reasoning_docs_status"] == "pending_smoke"
    assert lock["resolved_adapters"]["Z.AI"]["metadata"]["reasoning_wire_mapping"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }
    assert lock["resolved_execution_configuration"]["stream_timing"] == config.raw["stream_timing"]
    assert lock["tokenizer"]["version"]
    assert lock["tokenizer"]["vocabulary_sha256"]
    assert len(lock["evaluator_code_hashes"]) == 3
    assert validate_run_lock(run_dir, local_config) == lock


def test_generated_code_retry_amendment_is_narrow_and_immutable(config, tmp_path, monkeypatch):
    local_config = replace(config, run_root=tmp_path)
    real_hashes = lifecycle_module._generated_hashes(local_config)
    current = {"hashes": dict(real_hashes)}
    monkeypatch.setattr(
        lifecycle_module,
        "_generated_hashes",
        lambda _config: dict(current["hashes"]),
    )
    run_dir = tmp_path / "run-20260902-120001"
    prepare_run(run_dir, config=local_config)
    locked_bytes = (run_dir / "run-lock.json").read_bytes()
    (run_dir / "artifacts/run-manifest-attempt-1.json").write_text(
        '{"status":"smoke_attempt_failed"}\n', encoding="utf-8"
    )
    changed_path = next(path for path in current["hashes"] if path.endswith("runner.py"))
    current["hashes"][changed_path] = "f" * 64
    with pytest.raises(RuntimeError, match="generated_code_hashes changed"):
        validate_run_lock(run_dir, local_config, attempt=2)
    amendment = create_generated_code_retry_amendment(
        run_dir,
        reason="attempt-1 stream-channel diagnostic fix",
        source_diagnostics=["artifacts/run-manifest-attempt-1.json"],
        config=local_config,
    )
    assert amendment["generated_code_changes"][changed_path] == {
        "change": "modified",
        "old_sha256": real_hashes[changed_path],
        "new_sha256": "f" * 64,
    }
    assert amendment["approver"] == "local-human-review"
    assert validate_run_lock(run_dir, local_config, attempt=2)["run_id"] == run_dir.name
    assert (run_dir / "run-lock.json").read_bytes() == locked_bytes
    assert create_generated_code_retry_amendment(
        run_dir,
        reason="attempt-1 stream-channel diagnostic fix",
        source_diagnostics=["artifacts/run-manifest-attempt-1.json"],
        config=local_config,
    ) == amendment
