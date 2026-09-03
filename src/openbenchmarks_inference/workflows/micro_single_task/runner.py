"""Paired-round local execution through the shared streaming transport."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
from pathlib import Path
import time
from typing import Any, Mapping

from .config import BenchmarkConfig, load_config
from .dataset import DatasetItem, DatasetRelease, load_dataset
from .lifecycle import (
    load_provider_environment,
    prepare_run,
    require_provider_credentials,
    runtime_identity,
    tokenizer_identity,
    validate_run_lock,
)
from .schedule import QuestionRound, WorkUnit, build_full_schedule, build_smoke_schedule, flatten
from .storage import (
    LocalRunStore,
    SmokeBudget,
    artifact_inventory,
    atomic_json,
    atomic_text,
    content_hash,
    now_iso,
    read_json,
)


FULL_SUCCESS_STATUS = "full_run_completed_awaiting_human_review"


def _projected_cost(config: BenchmarkConfig, unit: WorkUnit, item: DatasetItem) -> float:
    input_tokens = int(item.tokens.get("input_total", item.tokens.get("user", 0)))
    output_tokens = int(config.raw["inference"]["max_completion_tokens"])
    return config.pricing[unit.provider].cost(input_tokens, output_tokens)


def _cost(config: BenchmarkConfig, record: Mapping[str, Any]) -> float | None:
    value = record.get("cost_usd")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    cost_inputs = record.get("cost_inputs")
    if isinstance(cost_inputs, dict):
        value = cost_inputs.get("estimated_cost_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
    return config.usage_cost(str(record["provider"]), record.get("usage"))


def _content(record: Mapping[str, Any]) -> str:
    for name in ("reconstructed_answer", "assistant_content", "response_content", "answer"):
        value = record.get(name)
        if isinstance(value, str):
            return value
    return ""


def _http_status(record: Mapping[str, Any]) -> int | None:
    value = record.get("http_status", record.get("status_code"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _normalise_terminal(unit: WorkUnit, value: Any, handoff_ns: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("transport must return a terminal record mapping")
    record = {
        "request_id": unit.request_id,
        "round_id": unit.round_id,
        "round_index": unit.round_index,
        "provider": unit.provider,
        "provider_launch_index": unit.provider_launch_index,
        "task_family": unit.task_family,
        "item_id": unit.item_id,
        **value,
    }
    record.setdefault("request_start_monotonic_ns", handoff_ns)
    if "terminal_outcome" not in record:
        status = _http_status(record)
        if status is not None and 200 <= status < 300:
            record["terminal_outcome"] = "completed"
        elif status is not None:
            record["terminal_outcome"] = "http_error"
        else:
            record["terminal_outcome"] = "transport_error"
    record.setdefault("stream_events", [])
    record.setdefault("reconstructed_reasoning", "")
    record.setdefault("reconstructed_answer", _content(record))
    return record


async def _prepare_transport(transport: Any, unit: WorkUnit, item: DatasetItem) -> Any:
    method = getattr(transport, "prepare_request", None)
    if method is None:
        return None
    try:
        result = method(unit=unit, item=item)
    except TypeError as first:
        try:
            result = method(unit, item)
        except TypeError:
            raise first
    return await result if inspect.isawaitable(result) else result


async def _call_transport(
    transport: Any,
    unit: WorkUnit,
    item: DatasetItem,
    prepared: Any = None,
    event_sink: Any = None,
) -> Any:
    prepared_method = getattr(transport, "execute_prepared", None)
    if prepared_method is not None and prepared is not None:
        parameters = inspect.signature(prepared_method).parameters
        kwargs = {"prepared": prepared}
        if "event_sink" in parameters:
            kwargs["event_sink"] = event_sink
        result = prepared_method(**kwargs)
        return await result if inspect.isawaitable(result) else result
    method = getattr(transport, "execute_request", None) or getattr(transport, "execute", None)
    if method is None:
        if callable(transport):
            method = transport
        else:
            raise TypeError("transport needs execute_request or execute")
    # The generated transport's canonical API is keyword based. These two
    # compatibility paths keep fakes and older generated adapters usable.
    parameters = inspect.signature(method).parameters
    kwargs = {"unit": unit, "item": item}
    if "event_sink" in parameters:
        kwargs["event_sink"] = event_sink
    result = method(**kwargs)
    return await result if inspect.isawaitable(result) else result


@asynccontextmanager
async def _transport_scope(transport: Any):
    if hasattr(transport, "__aenter__"):
        async with transport as entered:
            yield entered
        return
    try:
        yield transport
    finally:
        close = getattr(transport, "close", None)
        if close is not None:
            value = close()
            if inspect.isawaitable(value):
                await value


class BenchmarkRunner:
    def __init__(
        self,
        run_dir: Path | str,
        *,
        config: BenchmarkConfig | None = None,
        dataset: DatasetRelease | None = None,
        transport: Any,
    ):
        self.config = config or load_config()
        self.dataset = dataset or load_dataset(self.config)
        self.store = LocalRunStore(run_dir)
        self.store.initialize()
        self.transport = transport
        self.items = {item.id: item for rows in self.dataset.by_family.values() for item in rows}

    async def _execute_round(
        self,
        question: QuestionRound,
        *,
        attempt: int | None,
        budget: SmokeBudget | None,
        terminal_ids: set[str],
    ) -> list[dict[str, Any]]:
        missing = [unit for unit in question.units if unit.request_id not in terminal_ids]
        if not missing:
            return []
        # Resolve every provider request before any provider is released. A
        # preparation failure is durable runner evidence, never a partial
        # pre-barrier network launch.
        prepared: dict[str, Any] = {}
        preparation_errors: dict[str, Exception] = {}
        for unit in missing:
            try:
                prepared[unit.request_id] = await _prepare_transport(
                    self.transport, unit, self.items[unit.item_id]
                )
            except Exception as exc:
                preparation_errors[unit.request_id] = exc
        gate = asyncio.Event()
        ready = 0
        ready_lock = asyncio.Lock()
        all_ready = asyncio.Event()

        async def execute(unit: WorkUnit) -> dict[str, Any]:
            nonlocal ready
            item = self.items[unit.item_id]
            if budget is not None and unit.request_id not in preparation_errors:
                budget.reserve(int(attempt), unit.request_id, _projected_cost(self.config, unit, item))
            async with ready_lock:
                ready += 1
                if ready == len(missing):
                    all_ready.set()
            await gate.wait()
            handoff_ns = time.monotonic_ns()
            try:
                if unit.request_id in preparation_errors:
                    raise preparation_errors[unit.request_id]
                self.store.event("external_call_started", request_id=unit.request_id, provider=unit.provider, round_id=unit.round_id)
                event_sink = lambda event: self.store.write_stream_event(
                    unit.request_id, event, attempt=attempt
                )
                value = await _call_transport(
                    self.transport,
                    unit,
                    item,
                    prepared.get(unit.request_id),
                    event_sink,
                )
                record = _normalise_terminal(unit, value, handoff_ns)
            except Exception as exc:
                record = _normalise_terminal(
                    unit,
                    {
                        "terminal_outcome": "runner_error",
                        "http_status": None,
                        "error": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                        "request_start_monotonic_ns": handoff_ns,
                    },
                    handoff_ns,
                )
            # Persist at the individual terminal boundary. A fast provider is
            # resumably complete even while another member of the paired round
            # remains in flight.
            self.store.write_terminal(record, attempt=attempt)
            if budget is not None and unit.request_id not in preparation_errors:
                budget.observe(int(attempt), unit.request_id, _cost(self.config, record))
            self.store.event(
                "external_call_terminal",
                request_id=unit.request_id,
                provider=unit.provider,
                round_id=question.round_id,
                terminal_outcome=record["terminal_outcome"],
            )
            return record

        tasks = [asyncio.create_task(execute(unit)) for unit in missing]
        await all_ready.wait()
        self.store.event("round_released", round_id=question.round_id, pending_requests=len(missing))
        gate.set()
        records = await asyncio.gather(*tasks)
        round_ids = {unit.request_id for unit in question.units}
        prior_records = [
            row
            for row in self.store.terminal_records(attempt)
            if row.get("request_id") in round_ids
        ]
        all_round_records = [*prior_records, *records]
        starts = [
            row.get("request_start_monotonic_ns")
            for row in all_round_records
            if isinstance(row.get("request_start_monotonic_ns"), int)
        ]
        skew_ms = (max(starts) - min(starts)) / 1_000_000 if starts else None
        durable_round_ids = self.store.terminal_ids(attempt) & round_ids
        if durable_round_ids != round_ids:
            raise RuntimeError(
                f"round barrier persistence incomplete: {question.round_id}"
            )
        self.store.write_immutable_artifact(
            f"artifacts/rounds/{self.store._partition(attempt)}/{question.round_id}.json",
            {
                "round_id": question.round_id,
                "terminal_request_ids": [
                    unit.request_id for unit in question.units
                ],
                "launch_skew_ms": skew_ms,
                "resumed_partial_round": len(missing) != len(question.units),
            },
        )
        return records

    async def execute_schedule(
        self,
        rounds: tuple[QuestionRound, ...],
        *,
        attempt: int | None,
        budget: SmokeBudget | None = None,
    ) -> list[dict[str, Any]]:
        terminal_ids = self.store.terminal_ids(attempt)
        for question in rounds:
            await self._execute_round(question, attempt=attempt, budget=budget, terminal_ids=terminal_ids)
            terminal_ids = self.store.terminal_ids(attempt)
        return self.store.terminal_records(attempt)


def _configuration_evidence(
    config: BenchmarkConfig, record: Mapping[str, Any]
) -> dict[str, Any]:
    provider = record.get("provider")
    if provider not in config.providers:
        return {"compatible": False, "reasons": ["unknown provider"]}
    expected_model = config.raw["providers"]["vendors"][provider]["model_key"]
    metadata = record.get("adapter_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    identity = record.get("model_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    request = record.get("request")
    request = request if isinstance(request, Mapping) else {}
    body = request.get("body")
    body = body if isinstance(body, Mapping) else {}
    reasoning_mapping = metadata.get("reasoning_wire_mapping")
    reasoning_mapping = reasoning_mapping if isinstance(reasoning_mapping, Mapping) else {}
    terminal_outcome = record.get("terminal_outcome")
    http_status = _http_status(record)
    completed_response = (
        terminal_outcome == "completed"
        and http_status is not None
        and 200 <= http_status < 300
    )
    # Some unit/legacy records predate terminal_outcome but contain explicit
    # response identity. Continue validating that evidence when it is present.
    response_evidence_required = completed_response or (
        terminal_outcome is None and bool(identity)
    )
    reasons: list[str] = []
    checks = {
        "provider_model_matches_spec": record.get("provider_model") == expected_model,
        "request_model_matches_spec": body.get("model") == expected_model,
        "reported_model_matches_intended": (
            not response_evidence_required
            or (
                identity.get("requested") == expected_model
                and identity.get("reported") == expected_model
                and identity.get("matches") is True
            )
        ),
        "adapter_model_matches_spec": metadata.get("model_key") == expected_model,
        "protocol_matches": metadata.get("protocol") == "openai-chat-completions",
        "endpoint_class_matches": metadata.get("endpoint_class") == "serverless_or_shared",
        "streaming_matches": metadata.get("streaming") is True and body.get("stream") is True,
        "structured_output_matches": (
            metadata.get("structured_output_mode") == "json_object"
            and metadata.get("schema_enforcement") == "local_strict"
            and body.get("response_format") == {"type": "json_object"}
        ),
        "single_attempt_matches": metadata.get("max_attempts") == 1,
        "sampling_matches": (
            body.get("temperature") == 0
            and body.get("top_p") == 1
            and body.get("presence_penalty") == 0
            and body.get("frequency_penalty") == 0
        ),
        "completion_ceiling_matches": any(
            body.get(name) == 256
            for name in ("max_completion_tokens", "max_tokens")
        ),
        "low_reasoning_mapped": (
            metadata.get("reasoning_effort") == "low"
            and reasoning_mapping.get("reasoning_effort") == "low"
            and body.get("reasoning_effort") == "low"
        ),
    }
    for label, passed in checks.items():
        if not passed:
            reasons.append(label)
    compatibility = record.get("adapter_compatibility")
    compatibility = compatibility if isinstance(compatibility, Mapping) else {}
    if metadata.get("reasoning_docs_status") == "pending_smoke":
        # A generic 2xx response only shows that the field was accepted; it
        # does not prove the provider interpreted it as exact low effort.
        exact_low = (
            not response_evidence_required
            or (
                compatibility.get("status") == "smoke_validated"
                and compatibility.get("exact_low_reasoning_evidenced") is True
            )
        )
        checks["pending_low_reasoning_genuinely_evidenced"] = exact_low
        if not exact_low:
            reasons.append("pending_low_reasoning_genuinely_evidenced")
    else:
        documented = metadata.get("reasoning_docs_status") == "documented"
        checks["low_reasoning_documented"] = documented
        if not documented:
            reasons.append("low_reasoning_documented")
    return {"compatible": not reasons, "checks": checks, "reasons": reasons}


def _recorded_timing(record: Mapping[str, Any], name: str) -> Any:
    timings = record.get("timings")
    if isinstance(timings, Mapping) and name in timings:
        return timings[name]
    return record.get(name)


def _event_channel_error(event: Mapping[str, Any]) -> str | None:
    """Validate channel preservation without forbidding a shared SSE arrival."""
    reasoning = event.get("reasoning_content")
    answer = event.get("answer_content")
    has_reasoning = isinstance(reasoning, str) and bool(reasoning)
    has_answer = isinstance(answer, str) and bool(answer)
    classification = event.get("classification")
    presence = event.get("content_presence")
    if not isinstance(presence, Mapping):
        return "missing content_presence"
    if presence.get("reasoning") is not has_reasoning or presence.get("answer") is not has_answer:
        return "content_presence does not match distinct channel fields"
    expected_classification = (
        "reasoning_and_answer"
        if has_reasoning and has_answer
        else "reasoning"
        if has_reasoning
        else "answer"
        if has_answer
        else None
    )
    if expected_classification is not None and classification != expected_classification:
        return "content-bearing event is misclassified"
    if classification == "reasoning_and_answer" and not (has_reasoning and has_answer):
        return "combined event does not preserve both explicit channels"
    payload = event.get("redacted_payload", event.get("payload"))
    if isinstance(payload, Mapping):
        try:
            delta = payload["choices"][0]["delta"]
        except (KeyError, IndexError, TypeError):
            delta = None
        if isinstance(delta, Mapping):
            payload_reasoning = next(
                (
                    delta.get(name)
                    for name in ("reasoning_content", "reasoning", "reasoning_details")
                    if delta.get(name) not in (None, "", [], {})
                ),
                None,
            )
            payload_answer = delta.get("content")
            payload_has_reasoning = (
                isinstance(payload_reasoning, str) and bool(payload_reasoning)
            ) or (
                isinstance(payload_reasoning, (list, Mapping)) and bool(payload_reasoning)
            )
            payload_has_answer = isinstance(payload_answer, str) and bool(payload_answer)
            if payload_has_reasoning != has_reasoning or payload_has_answer != has_answer:
                return "payload channels were conflated or dropped"
    return None


def _recompute_evidence(
    *,
    config: BenchmarkConfig,
    items: Mapping[str, DatasetItem],
    records: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    materialized_stream_events: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    paired: Mapping[str, Any],
    run_id: str,
    provider_compatibility: Mapping[str, Any],
    artifact_completeness: Mapping[str, Any],
) -> dict[str, Any]:
    from .evaluation import evaluate_record, is_known_transport_runner_error
    from .metrics import compute_metrics, compute_paired_comparisons, recompute_timing

    saved_evaluations = {row.get("request_id"): row for row in evaluations}
    evaluation_matches = True
    timing_matches = True
    stream_reconstructs = True
    channel_separation_valid = True
    channel_errors: list[str] = []
    errors: list[str] = []
    durable_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for event in materialized_stream_events:
        if isinstance(event, Mapping) and isinstance(event.get("request_id"), str):
            durable_by_id.setdefault(str(event["request_id"]), []).append(event)
    for record in records:
        request_id = record.get("request_id")
        normalized_transport_failure = is_known_transport_runner_error(record)
        try:
            recomputed_evaluation = evaluate_record(items[str(record["item_id"])], record)
            if saved_evaluations.get(request_id) != recomputed_evaluation:
                evaluation_matches = False
                errors.append(f"evaluation mismatch: {request_id}")
            events = record.get("stream_events")
            if record.get("terminal_outcome") == "completed" and not events:
                stream_reconstructs = False
                errors.append(f"completed record lacks stream events: {request_id}")
            durable_events = durable_by_id.get(str(request_id), [])
            if (
                len(durable_events) != len(events if isinstance(events, list) else [])
                and not normalized_transport_failure
            ):
                stream_reconstructs = False
                errors.append(f"durable stream event count mismatch: {request_id}")
            if any(event.get("durability") != "arrival_atomic" for event in durable_events):
                stream_reconstructs = False
                errors.append(f"stream event not persisted at arrival: {request_id}")
            events_to_validate = (
                durable_events
                if normalized_transport_failure and durable_events
                else events
            )
            if isinstance(events_to_validate, list):
                for event in events_to_validate:
                    if not isinstance(event, Mapping):
                        continue
                    channel_error = _event_channel_error(event)
                    if channel_error is not None:
                        channel_separation_valid = False
                        channel_errors.append(f"{request_id}: {channel_error}")
            recomputed_timing = recompute_timing(record)
            if recomputed_timing["reconstructed_answer"] != record.get("reconstructed_answer", ""):
                stream_reconstructs = False
                errors.append(f"answer reconstruction mismatch: {request_id}")
            if recomputed_timing["reconstructed_reasoning"] != record.get("reconstructed_reasoning", ""):
                stream_reconstructs = False
                errors.append(f"reasoning reconstruction mismatch: {request_id}")
            for name in (
                "ttfo_ms", "ttfa_ms",
                "pre_answer_reasoning_tokens", "reasoning_emitted_before_answer",
                "e2e_latency_ms", "first_answer_token_ms",
                "last_answer_token_ms", "visible_answer_token_count",
                "output_tokens_per_second",
            ):
                if recomputed_timing[name] != _recorded_timing(record, name):
                    timing_matches = False
                    errors.append(f"timing mismatch {name}: {request_id}")
        except Exception as exc:
            evaluation_matches = False
            timing_matches = False
            stream_reconstructs = False
            errors.append(f"recomputation error {request_id}: {type(exc).__name__}: {exc}")
    aggregate_matches = False
    paired_matches = False
    try:
        recomputed_metrics = compute_metrics(
            records,
            evaluations,
            benchmark_slug=config.benchmark_slug,
            run_id=run_id,
            provider_compatibility=provider_compatibility,
            artifact_completeness=artifact_completeness,
            expected_task_families=config.task_families,
        )
        aggregate_matches = recomputed_metrics == metrics
        recomputed_paired = compute_paired_comparisons(
            records,
            benchmark_slug=config.benchmark_slug,
            run_id=run_id,
            provider_compatibility=provider_compatibility,
            artifact_completeness=artifact_completeness,
        )
        paired_matches = recomputed_paired == paired
    except Exception as exc:
        errors.append(f"aggregate recomputation error: {type(exc).__name__}: {exc}")
    return {
        "evaluations_match": evaluation_matches,
        "timings_match": timing_matches,
        "stream_reconstructs": stream_reconstructs,
        "channel_separation_valid": channel_separation_valid,
        "channel_errors": channel_errors,
        "metrics_match": aggregate_matches,
        "paired_comparisons_match": paired_matches,
        "errors": errors,
    }


def _acceptance(
    config: BenchmarkConfig,
    items: Mapping[str, DatasetItem],
    records: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    stream_events: list[dict[str, Any]],
    reconstructed: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    paired: Mapping[str, Any],
    run_id: str,
    provider_compatibility: Mapping[str, Any],
    artifact_completeness: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    expected_pairs = {(provider, family) for provider in config.providers for family in config.task_families}
    by_pair = {(row.get("provider"), row.get("task_family")): row for row in records}
    eval_by_id = {row.get("request_id"): row for row in evaluations}
    completed = all(pair in by_pair and by_pair[pair].get("terminal_outcome") == "completed" for pair in expected_pairs)
    http_2xx = all(pair in by_pair and (_http_status(by_pair[pair]) or 0) in range(200, 300) for pair in expected_pairs)
    content = all(pair in by_pair and bool(_content(by_pair[pair]).strip()) for pair in expected_pairs)

    def schema_valid(row: dict[str, Any]) -> bool:
        evaluation = eval_by_id.get(row.get("request_id"), {})
        if evaluation.get("schema_valid") is not None:
            return evaluation["schema_valid"] is True
        return evaluation.get("classification") not in {"malformed_json", "schema_invalid", "missing_required_field", "invalid_result", "runner_error"}

    schemas = len(records) == 30 and all(schema_valid(row) for row in records)
    compatibility = all(
        provider_compatibility.get(provider, {}).get("compatible") is True
        for provider in config.providers
    )
    recomputation = _recompute_evidence(
        config=config,
        items=items,
        records=records,
        evaluations=evaluations,
        materialized_stream_events=stream_events,
        metrics=metrics,
        paired=paired,
        run_id=run_id,
        provider_compatibility=provider_compatibility,
        artifact_completeness=artifact_completeness,
    )
    checks = {
        "runner_errors": sum(row.get("terminal_outcome") == "runner_error" for row in records),
        "local_artifact_errors": 0,
        "every_provider_completes_at_least_one_request_per_task_family": completed,
        "every_provider_returns_http_2xx_per_task_family": http_2xx,
        "every_provider_returns_nonempty_answer_content_per_task_family": content,
        "every_provider_returns_schema_valid_result_per_task_family": schemas,
        "every_provider_uses_intended_model_and_compatible_configuration": compatibility,
        "every_observed_reasoning_stream_is_timed_separately_from_answer_content": recomputation["channel_separation_valid"],
        "raw_result_count_matches_submitted_request_count": len(records) == 30,
        "stream_events_reconstruct_final_answers": recomputation["stream_reconstructs"],
        "evaluations_recompute_from_raw_results": recomputation["evaluations_match"],
        "timing_metrics_recompute_from_raw_stream_events": recomputation["timings_match"],
        "aggregate_metrics_recompute_exactly": recomputation["metrics_match"],
        "paired_comparisons_recompute_exactly": recomputation["paired_comparisons_match"],
        "database_writes": 0,
        "recomputation": recomputation,
    }
    success = checks["runner_errors"] == 0 and checks["local_artifact_errors"] == 0 and checks["database_writes"] == 0 and all(value is True for key, value in checks.items() if key not in {"runner_errors", "local_artifact_errors", "database_writes", "recomputation"})
    return success, checks


async def _derive_and_report(
    runner: BenchmarkRunner,
    rounds: tuple[QuestionRound, ...],
    *,
    attempt: int | None,
) -> dict[str, Any]:
    from .evaluation import evaluate_record, is_known_transport_runner_error
    from .metrics import compute_metrics, compute_paired_comparisons
    from .reporting import render_report

    ordered_ids = [unit.request_id for unit in flatten(rounds)]
    records = runner.store.terminal_records(attempt)
    record_by_id = {row["request_id"]: row for row in records}
    for request_id in ordered_ids:
        record = record_by_id.get(request_id)
        if record is not None:
            runner.store.write_evaluation(evaluate_record(runner.items[record["item_id"]], record), attempt=attempt)
    materialized = runner.store.materialize_attempt(attempt=attempt, ordered_request_ids=ordered_ids)
    records = materialized["records"]
    evaluations = materialized["evaluations"]
    expected_per_provider = len(rounds)
    provider_records = {
        provider: [
            record for record in records if record.get("provider") == provider
        ]
        for provider in runner.config.providers
    }
    compatibility_rows = {
        provider: [
            _configuration_evidence(runner.config, record)
            for record in provider_records[provider]
            if not is_known_transport_runner_error(record)
        ]
        for provider in runner.config.providers
    }
    provider_compatibility = {
        provider: {
            "compatible": (
                len(provider_records[provider]) == expected_per_provider
                and bool(rows)
                and all(row["compatible"] for row in rows)
            ),
            "records": len(provider_records[provider]),
            "expected_records": expected_per_provider,
            "configuration_evidence_records": len(rows),
            "normalized_transport_failure_records": sum(
                is_known_transport_runner_error(record)
                for record in provider_records[provider]
            ),
            "reasons": sorted(
                {reason for row in rows for reason in row.get("reasons", [])}
            ),
        }
        for provider, rows in compatibility_rows.items()
    }
    artifact_completeness = {
        provider: {
            "complete": (
                sum(record.get("provider") == provider for record in records)
                == expected_per_provider
                and sum(evaluation.get("provider") == provider for evaluation in evaluations)
                == expected_per_provider
            )
        }
        for provider in runner.config.providers
    }
    completeness = (
        len(records) == len(ordered_ids)
        and len(evaluations) == len(records)
        and all(row["complete"] for row in artifact_completeness.values())
    )
    metrics = compute_metrics(
        records,
        evaluations,
        benchmark_slug=runner.config.benchmark_slug,
        run_id=runner.store.run_dir.name,
        provider_compatibility=provider_compatibility,
        artifact_completeness=artifact_completeness,
        expected_task_families=runner.config.task_families,
    )
    paired = compute_paired_comparisons(
        records,
        benchmark_slug=runner.config.benchmark_slug,
        run_id=runner.store.run_dir.name,
        provider_compatibility=provider_compatibility,
        artifact_completeness=artifact_completeness,
    )
    artifact_suffix = f"-attempt-{attempt}" if attempt is not None else ""
    atomic_json(runner.store.artifact_dir / f"metrics{artifact_suffix}.json", metrics)
    atomic_json(runner.store.artifact_dir / f"paired-comparisons{artifact_suffix}.json", paired)
    if attempt is not None:
        success, checks = _acceptance(
            runner.config,
            runner.items,
            records,
            evaluations,
            materialized["stream_events"],
            materialized["reconstructed"],
            metrics,
            paired,
            runner.store.run_dir.name,
            provider_compatibility,
            artifact_completeness,
        )
        status = (
            "smoke_passed_awaiting_human_review"
            if success
            else ("smoke_failed_after_retry" if attempt == 2 else "smoke_attempt_failed")
        )
    else:
        recomputation = _recompute_evidence(
            config=runner.config,
            items=runner.items,
            records=records,
            evaluations=evaluations,
            materialized_stream_events=materialized["stream_events"],
            metrics=metrics,
            paired=paired,
            run_id=runner.store.run_dir.name,
            provider_compatibility=provider_compatibility,
            artifact_completeness=artifact_completeness,
        )
        configuration_complete = all(
            row["compatible"] for row in provider_compatibility.values()
        )
        success = (
            completeness
            and configuration_complete
            and recomputation["evaluations_match"]
            and recomputation["timings_match"]
            and recomputation["stream_reconstructs"]
            and recomputation["channel_separation_valid"]
            and recomputation["metrics_match"]
            and recomputation["paired_comparisons_match"]
        )
        checks = {
            "artifact_graph_complete": completeness,
            "provider_configuration_compatible": configuration_complete,
            "recomputation": recomputation,
            "database_writes": 0,
        }
        status = FULL_SUCCESS_STATUS if success else "full_run_invalid"
    if attempt is not None:
        runner.store.complete_attempt(attempt, success)
    else:
        atomic_json(
            runner.store.checkpoint_dir / "full-run.json",
            {
                "state": "complete",
                "success": success,
                "completed_at": now_iso(),
                "terminal_requests": len(records),
            },
        )
    round_partition = runner.store._partition(attempt)
    round_history = [
        read_json(path, {})
        for path in sorted(
            (runner.store.artifact_dir / "rounds" / round_partition).glob("*.json")
        )
    ]
    resume_history = {
        "resumed_partial_rounds": [
            row.get("round_id")
            for row in round_history
            if row.get("resumed_partial_round") is True
        ],
        "round_diagnostic_count": len(round_history),
    }
    started_values = sorted(
        str(row["started_at"])
        for row in records
        if isinstance(row.get("started_at"), str)
    )
    completed_values = sorted(
        str(row["completed_at"])
        for row in records
        if isinstance(row.get("completed_at"), str)
    )
    execution_metadata = runtime_identity()
    execution_metadata["wall_clock_period"] = {
        "start": started_values[0] if started_values else None,
        "end": completed_values[-1] if completed_values else None,
    }
    manifest = {
        "run_id": runner.store.run_dir.name,
        "benchmark_slug": runner.config.benchmark_slug,
        "kind": "smoke" if attempt is not None else "full",
        "attempt": attempt,
        "status": status,
        "created_at": now_iso(),
        "expected_rounds": len(rounds),
        "expected_requests": len(ordered_ids),
        "submitted_requests": len(records),
        "raw_records": len(records),
        "stream_event_records": len(materialized["stream_events"]),
        "reconstructed_records": len(materialized["reconstructed"]),
        "evaluation_records": len(evaluations),
        "database_writes": 0,
        "selection_sha256": read_json(runner.store.run_dir / "run-lock.json", {}).get("selection_hash"),
        "schedule_sha256": content_hash([row.to_dict() for row in rounds]),
        "acceptance": checks,
        "artifact_complete": success,
        "complete": success,
        "provider_compatibility": provider_compatibility,
        "artifact_completeness_by_provider": artifact_completeness,
        "provider_adapter_wire_mappings": {},
        "execution_metadata": execution_metadata,
        "resume_history": resume_history,
    }
    manifest_path = (
        runner.store.artifact_dir / f"run-manifest-attempt-{attempt}.json"
        if attempt is not None
        else runner.store.artifact_dir / "run-manifest.json"
    )
    reasoning_mappings: dict[str, Any] = {}
    for provider in runner.config.providers:
        provider_record = next(
            (row for row in records if row.get("provider") == provider), {}
        )
        metadata = provider_record.get("adapter_metadata")
        reasoning_mappings[provider] = (
            dict(metadata)
            if isinstance(metadata, Mapping)
            else "unavailable"
        )
    manifest["provider_adapter_wire_mappings"] = reasoning_mappings
    if success and attempt is not None:
        atomic_json(runner.store.artifact_dir / "metrics.json", metrics)
        atomic_json(runner.store.artifact_dir / "paired-comparisons.json", paired)
    report = render_report(
        benchmark_name=runner.config.raw["benchmark"]["name"],
        run_id=runner.store.run_dir.name,
        status=manifest["status"],
        metrics=metrics,
        paired_comparisons=paired,
        manifest=manifest,
        task_families=runner.config.task_families,
        tokenizer=tokenizer_identity(),
        execution_metadata=execution_metadata,
        provider_reasoning_mappings=reasoning_mappings,
        selection_hash=manifest["selection_sha256"],
        raw_event_completeness={"events": len(materialized["stream_events"]), "requests": len(records)},
        resume_history=resume_history,
        run_dir=runner.store.run_dir,
        smoke_attempt=attempt,
        warnings=("Full execution requires explicit approval flags.",),
    )
    atomic_text(runner.store.run_dir / "report.md", report)
    inventory = artifact_inventory(runner.store.run_dir)
    manifest["artifacts"] = {
        path: value
        for path, value in inventory.items()
        if not path.startswith("artifacts/run-manifest")
    }
    atomic_json(manifest_path, manifest)
    if success and attempt is not None:
        atomic_json(runner.store.artifact_dir / "run-manifest.json", manifest)
    return {"success": success, "manifest": manifest, "metrics": metrics, "paired_comparisons": paired}


def _default_transport(config: BenchmarkConfig, environment: Mapping[str, str]) -> Any:
    from .transport import StreamingTransport

    return StreamingTransport(config=config, environment=environment)


async def run_smoke(
    run_dir: Path | str,
    attempt: int = 1,
    *,
    transport: Any | None = None,
    environment: Mapping[str, str] | None = None,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    root = Path(run_dir).resolve()
    if not (root / "run-lock.json").is_file():
        prepare_run(root, config=config)
    validate_run_lock(root, config, attempt=attempt)
    prior = read_json(root / "artifacts" / f"run-manifest-attempt-{attempt}.json")
    if isinstance(prior, dict):
        return {"success": prior.get("status") == "smoke_passed_awaiting_human_review", "manifest": prior}
    if attempt == 2:
        first = read_json(root / "artifacts/run-manifest-attempt-1.json")
        if not isinstance(first, dict) or first.get("status") != "smoke_attempt_failed":
            raise RuntimeError("attempt 2 requires one preserved failed attempt 1")
    env = dict(environment) if environment is not None else load_provider_environment(config)
    require_provider_credentials(config, env)
    rounds = build_smoke_schedule(config, load_dataset(config), attempt)
    store = LocalRunStore(root)
    store.initialize()
    store.write_immutable_artifact(
        f"artifacts/smoke-plan-attempt-{attempt}.json",
        {"kind": "smoke", "attempt": attempt, "expected_rounds": 3, "expected_requests": 30, "rounds": [row.to_dict() for row in rounds]},
    )
    budget = SmokeBudget(store.artifact_dir / "smoke-budget.json", config.smoke_request_limit, config.smoke_cost_limit)
    store.begin_attempt(attempt)
    actual_transport = transport if transport is not None else _default_transport(config, env)
    async with _transport_scope(actual_transport) as active_transport:
        runner = BenchmarkRunner(root, config=config, dataset=load_dataset(config), transport=active_transport)
        await runner.execute_schedule(rounds, attempt=attempt, budget=budget)
        result = await _derive_and_report(runner, rounds, attempt=attempt)
    return result


async def run_full(
    run_dir: Path | str,
    *,
    approve_full: bool = False,
    approval_id: str | None = None,
    transport: Any | None = None,
    environment: Mapping[str, str] | None = None,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    if not approve_full or not isinstance(approval_id, str) or not approval_id.strip():
        raise PermissionError("full run refused: --approve-full and a non-empty --approval-id are required")
    # This path is generated for later human use. Runner generation never calls it.
    config = config or load_config()
    root = Path(run_dir).resolve()
    validate_run_lock(root, config)
    existing = read_json(root / "artifacts/run-manifest.json")
    if isinstance(existing, dict) and existing.get("kind") == "full":
        raise RuntimeError("completed historical full run is immutable; create a new run")
    full_state = read_json(root / "checkpoints/full-run.json")
    if full_state is None:
        atomic_json(
            root / "checkpoints/full-run.json",
            {"state": "running", "approval_id": approval_id, "started_at": now_iso()},
        )
    elif full_state.get("state") != "running" or full_state.get("approval_id") != approval_id:
        raise RuntimeError("full-run checkpoint approval/state cannot change on resume")
    approval_path = root / "artifacts/full-run-approval.json"
    approval = read_json(approval_path)
    if approval is None:
        atomic_json(
            approval_path,
            {"approval_id": approval_id, "approve_full": True, "recorded_at": now_iso()},
        )
    elif approval.get("approval_id") != approval_id or approval.get("approve_full") is not True:
        raise RuntimeError("full-run approval artifact cannot change on resume")
    env = dict(environment) if environment is not None else load_provider_environment(config)
    require_provider_credentials(config, env)
    rounds = build_full_schedule(config, load_dataset(config))
    actual_transport = transport if transport is not None else _default_transport(config, env)
    async with _transport_scope(actual_transport) as active_transport:
        runner = BenchmarkRunner(root, config=config, dataset=load_dataset(config), transport=active_transport)
        await runner.execute_schedule(rounds, attempt=None)
        result = await _derive_and_report(runner, rounds, attempt=None)
    return {**result, "approval_id": approval_id}


async def derive_full_run(
    run_dir: Path | str,
    *,
    approval_id: str,
    replace_invalid: bool = False,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Build full-run artifacts from a complete raw run without network access."""

    if not isinstance(approval_id, str) or not approval_id.strip():
        raise PermissionError("offline full-run derivation requires a non-empty approval-id")
    config = config or load_config()
    root = Path(run_dir).resolve()
    lock = validate_run_lock(
        root,
        config,
        allow_post_processing_code_change=True,
    )
    existing = read_json(root / "artifacts/run-manifest.json")
    if isinstance(existing, dict) and existing.get("kind") == "full":
        if not (replace_invalid and existing.get("status") == "full_run_invalid"):
            return {
                "success": existing.get("status") == FULL_SUCCESS_STATUS,
                "manifest": existing,
            }
    full_state = read_json(root / "checkpoints/full-run.json")
    checkpoint_matches = (
        isinstance(full_state, dict)
        and full_state.get("approval_id", approval_id) == approval_id
        and (
            full_state.get("state") == "running"
            or (
                replace_invalid
                and full_state.get("state") == "complete"
                and full_state.get("success") is False
            )
        )
    )
    if not checkpoint_matches:
        raise RuntimeError("offline derivation requires the matching running full-run checkpoint")
    approval = read_json(root / "artifacts/full-run-approval.json")
    if (
        not isinstance(approval, dict)
        or approval.get("approval_id") != approval_id
        or approval.get("approve_full") is not True
    ):
        raise RuntimeError("offline derivation requires the matching full-run approval artifact")

    dataset = load_dataset(config)
    rounds = build_full_schedule(config, dataset)
    ordered_ids = [unit.request_id for unit in flatten(rounds)]
    store = LocalRunStore(root)
    store.initialize()
    from .lifecycle import _evaluator_hashes, _generated_hashes

    post_processing_provenance = {
        "schema_version": "micro-single-task-post-processing-v1",
        "approval_id": approval_id,
        "preserved_run_lock_hash": content_hash(lock),
        "execution_generated_code_hashes": lock.get("generated_code_hashes"),
        "post_processing_generated_code_hashes": _generated_hashes(config),
        "execution_evaluator_code_hashes": lock.get("evaluator_code_hashes"),
        "post_processing_evaluator_code_hashes": _evaluator_hashes(config),
        "normalizations": [],
        "network_calls_permitted": False,
    }
    provenance_hash = content_hash(
        post_processing_provenance["post_processing_generated_code_hashes"]
    )[:12]
    store.write_immutable_artifact(
        f"artifacts/post-processing-provenance-{provenance_hash}.json",
        post_processing_provenance,
    )
    terminal_ids = store.terminal_ids(None)
    expected_ids = set(ordered_ids)
    if len(ordered_ids) != len(expected_ids):
        raise RuntimeError("full schedule contains duplicate request ids")
    if terminal_ids != expected_ids:
        raise RuntimeError(
            "offline derivation requires an exact terminal record set: "
            f"missing={len(expected_ids - terminal_ids)}, extra={len(terminal_ids - expected_ids)}"
        )

    # The transport is deliberately absent: this path never executes the
    # schedule and therefore cannot issue or retry a provider request.
    runner = BenchmarkRunner(
        root,
        config=config,
        dataset=dataset,
        transport=None,
    )
    result = await _derive_and_report(runner, rounds, attempt=None)
    return {**result, "approval_id": approval_id}
