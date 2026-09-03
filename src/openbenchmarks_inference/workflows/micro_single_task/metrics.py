"""Recomputable metrics for the paired single-user streaming benchmark."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .evaluation import (
    CONTRACT_FIELDS,
    CONTRACT_TASK,
    EvaluationContractError,
    TASK_FAMILIES,
    terminal_outcome,
)


METRICS_VERSION = "micro-single-task-metrics-v5"
FAILURE_OUTCOMES = ("timeout", "429", "other_4xx", "5xx", "transport_error")
PERCENTILES = {
    "ttfo_ms": ("p50", "p95", "p99"),
    "ttfa_ms": ("p50", "p95", "p99"),
    "pre_answer_reasoning_tokens": ("p50", "p95", "p99"),
    "e2e_latency_ms": ("p50", "p95", "p99"),
    "output_tokens_per_second": ("p50", "p95", "p99"),
}


class MetricsContractError(EvaluationContractError):
    """Raw/evaluation evidence cannot produce an unambiguous aggregate."""


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MetricsContractError(f"{field} must be a finite number")
    result = float(value)
    if result < 0 or (positive and result <= 0):
        raise MetricsContractError(f"{field} must be {'positive' if positive else 'non-negative'}")
    return result


def nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    """Return the declared nearest-rank percentile (ceil(p*n), one based)."""

    if not values:
        return None
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _metric_quantile(metric: str, label: str) -> float:
    """Map public coverage labels to the underlying distribution quantile."""

    coverage = int(label[1:]) / 100
    if metric == "output_tokens_per_second" and label != "p50":
        return 1 - coverage
    return coverage


def _token_count(text: str) -> int:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - pinned runtime dependency
        raise MetricsContractError("tiktoken is required to recompute visible token counts") from exc
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def _event_offset(event: Mapping[str, Any]) -> float:
    for key in ("monotonic_offset_ms", "offset_ms", "elapsed_ms", "arrival_offset_ms"):
        if key in event:
            return _number(event[key], f"stream event {key}")
    raise MetricsContractError("stream event is missing monotonic offset milliseconds")


def _delta(event: Mapping[str, Any]) -> tuple[str, str]:
    """Extract distinct reasoning and visible content from a persisted event."""

    reasoning = event.get("reasoning_content")
    answer = event.get("answer_content")
    classification = event.get("classification", event.get("event_type", event.get("kind")))
    content = event.get("content")
    if classification in {"reasoning", "reasoning_content", "reasoning_delta"} and reasoning is None:
        reasoning = content
    if classification in {"answer", "answer_content", "answer_delta", "visible_answer"} and answer is None:
        answer = content
    payload = event.get("redacted_payload", event.get("payload"))
    if isinstance(payload, Mapping):
        try:
            delta = payload["choices"][0]["delta"]
        except (KeyError, IndexError, TypeError):
            delta = None
        if isinstance(delta, Mapping):
            if reasoning is None:
                reasoning = delta.get("reasoning_content")
            if answer is None:
                answer = delta.get("content")
    return (
        reasoning if isinstance(reasoning, str) else "",
        answer if isinstance(answer, str) else "",
    )


def _fallback_timing(record: Mapping[str, Any]) -> dict[str, Any]:
    timings = record.get("timings") if isinstance(record.get("timings"), Mapping) else {}

    def value(name: str) -> float | None:
        candidate = timings.get(name, record.get(name))
        return None if candidate is None else _number(candidate, name)

    answer = record.get("reconstructed_answer", record.get("answer", ""))
    answer = answer if isinstance(answer, str) else ""
    tokens = record.get("visible_answer_token_count", timings.get("visible_answer_token_count"))
    if tokens is None and answer:
        tokens = _token_count(answer)
    if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
        raise MetricsContractError("visible_answer_token_count must be a non-negative integer")
    first = value("first_answer_token_ms")
    if first is None:
        first = value("ttfa_ms")
    last = value("last_answer_token_ms")
    if last is None:
        last = value("e2e_latency_ms")
    speed = value("output_tokens_per_second")
    if speed is None and isinstance(tokens, int) and tokens >= 2 and first is not None and last is not None and last > first:
        speed = (tokens - 1) / ((last - first) / 1000.0)
    outcome = terminal_outcome(record)
    complete = outcome == "2xx" and bool(answer)
    ttfo = value("ttfo_ms")
    if ttfo is None:
        ttfo = first
    reasoning_tokens = record.get(
        "pre_answer_reasoning_tokens", timings.get("pre_answer_reasoning_tokens")
    )
    if reasoning_tokens is None and first is not None:
        reasoning = record.get("reconstructed_reasoning", "")
        reasoning_tokens = _token_count(reasoning) if isinstance(reasoning, str) and reasoning else 0
    if reasoning_tokens is not None and (
        isinstance(reasoning_tokens, bool)
        or not isinstance(reasoning_tokens, int)
        or reasoning_tokens < 0
    ):
        raise MetricsContractError("pre_answer_reasoning_tokens must be a non-negative integer")
    return {
        "source": "persisted_timings",
        "ttfo_ms": ttfo if first is not None else None,
        "ttfa_ms": first,
        "pre_answer_reasoning_tokens": reasoning_tokens,
        # A request that fails before its first output has no observable answer
        # boundary. Preserve that as unavailable instead of converting it to a
        # measured ``False`` value during artifact recomputation.
        "reasoning_emitted_before_answer": (
            None if first is None else bool(reasoning_tokens)
        ),
        "e2e_latency_ms": value("e2e_latency_ms") if complete else None,
        "first_answer_token_ms": first,
        "last_answer_token_ms": last,
        "visible_answer_token_count": tokens,
        "output_tokens_per_second": speed if complete else None,
        "reconstructed_reasoning": record.get("reconstructed_reasoning", ""),
        "reconstructed_answer": answer,
        "complete_visible_answer": complete,
    }


def recompute_timing(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute first-output, answer, E2E, and output-speed diagnostics."""

    events = record.get("stream_events")
    if not isinstance(events, list) or not events:
        return _fallback_timing(record)
    ordered: list[tuple[float, int, str, str]] = []
    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            raise MetricsContractError("stream_events must contain objects")
        reasoning, answer = _delta(raw_event)
        ordered.append((_event_offset(raw_event), index, reasoning, answer))
    ordered.sort(key=lambda row: (row[0], row[1]))
    reasoning_text = ""
    pre_answer_reasoning_text = ""
    answer_text = ""
    ttfo: float | None = None
    ttfa: float | None = None
    last_answer: float | None = None
    for offset, _, reasoning, answer in ordered:
        if reasoning:
            reasoning_text += reasoning
            if ttfa is None:
                pre_answer_reasoning_text += reasoning
            if ttfo is None and _token_count(reasoning_text) >= 1:
                ttfo = offset
        if answer:
            answer_text += answer
            if ttfa is None and _token_count(answer_text) >= 1:
                ttfa = offset
            if ttfo is None and _token_count(answer_text) >= 1:
                ttfo = offset
            last_answer = offset
    recorded_answer = record.get("reconstructed_answer")
    if isinstance(recorded_answer, str) and recorded_answer != answer_text:
        raise MetricsContractError(
            f"stream events do not reconstruct saved answer for {record.get('request_id')!r}"
        )
    recorded_reasoning = record.get("reconstructed_reasoning")
    if isinstance(recorded_reasoning, str) and recorded_reasoning != reasoning_text:
        raise MetricsContractError(
            f"stream events do not reconstruct saved reasoning for {record.get('request_id')!r}"
        )
    token_count = _token_count(answer_text) if answer_text else 0
    complete = terminal_outcome(record) == "2xx" and bool(answer_text)
    e2e = last_answer if complete else None
    speed = None
    if complete and token_count >= 2 and ttfa is not None and last_answer is not None and last_answer > ttfa:
        speed = (token_count - 1) / ((last_answer - ttfa) / 1000.0)
    reasoning_token_count = _token_count(pre_answer_reasoning_text) if pre_answer_reasoning_text else 0
    return {
        "source": "stream_events",
        "ttfo_ms": ttfo if ttfa is not None else None,
        "ttfa_ms": ttfa,
        "pre_answer_reasoning_tokens": reasoning_token_count if ttfa is not None else None,
        "reasoning_emitted_before_answer": reasoning_token_count > 0 if ttfa is not None else None,
        "e2e_latency_ms": e2e,
        "first_answer_token_ms": ttfa,
        "last_answer_token_ms": last_answer,
        "visible_answer_token_count": token_count,
        "output_tokens_per_second": speed,
        "reconstructed_reasoning": reasoning_text,
        "reconstructed_answer": answer_text,
        "complete_visible_answer": complete,
    }


def _index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise MetricsContractError(f"every {label} requires a request_id")
        if request_id in output:
            raise MetricsContractError(f"duplicate {label} request_id {request_id!r}")
        output[request_id] = row
    return output


def _percentile_summary(
    metric: str, values: Sequence[float], submitted: int, labels: Sequence[str]
) -> dict[str, Any]:
    result = {
        "population_count": len(values),
        "submitted_count": submitted,
        "missing_sample_count": submitted - len(values),
        "population_rate": len(values) / submitted if submitted else None,
        "percentiles": {},
    }
    for label in labels:
        quantile = _metric_quantile(metric, label)
        result["percentiles"][label] = {
            "value": nearest_rank(values, quantile),
        }
    return result


def _usage_cost(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_total = output_total = 0
    usage_count = cost_count = 0
    known_cost = 0.0
    for record in records:
        usage = record.get("usage")
        if isinstance(usage, Mapping):
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
            if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (input_tokens, output_tokens)):
                input_total += input_tokens
                output_total += output_tokens
                usage_count += 1
        cost = record.get("cost_usd")
        if cost is not None:
            known_cost += _number(cost, "cost_usd")
            cost_count += 1
    submitted = len(records)
    return (
        {
            "reported_count": usage_count,
            "submitted_count": submitted,
            "missing_count": submitted - usage_count,
            "coverage_rate": usage_count / submitted,
            "input_tokens": input_total,
            "output_tokens": output_total,
            "total_tokens": input_total + output_total,
        },
        {
            "reported_count": cost_count,
            "submitted_count": submitted,
            "missing_count": submitted - cost_count,
            "coverage_rate": cost_count / submitted,
            "known_cost_usd": known_cost,
            "total_cost_usd": known_cost if cost_count == submitted else None,
        },
    )


def _family_metric(
    provider: str,
    family: str,
    records: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    *,
    slug: str,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluated = {row["request_id"]: row for row in evaluations}
    timing = {row["request_id"]: recompute_timing(row) for row in records}
    submitted = len(records)
    outcomes = [terminal_outcome(row) for row in records]
    correct = [1.0 if evaluated[row["request_id"]].get("correct") is True else 0.0 for row in records]
    failures = [1.0 if outcome in FAILURE_OUTCOMES else 0.0 for outcome in outcomes]
    series = {
        name: [sample[name] for sample in timing.values() if sample[name] is not None]
        for name in PERCENTILES
    }
    answer_samples = [sample for sample in timing.values() if sample["ttfa_ms"] is not None]
    reasoning_count = sum(sample["reasoning_emitted_before_answer"] is True for sample in answer_samples)
    row: dict[str, Any] = {
        "provider": provider,
        "task_family": family,
        "submitted_count": submitted,
        "correct_count": int(sum(correct)),
        "failed_request_count": int(sum(failures)),
        "invalid_result_count": sum(evaluated[r["request_id"]].get("classification") == "invalid_result" for r in records),
        "incorrect_result_count": sum(evaluated[r["request_id"]].get("classification") == "incorrect_result" for r in records),
        "task_success_rate": sum(correct) / submitted,
        "task_success_rate_detail": {
            "value": sum(correct) / submitted,
            "numerator": int(sum(correct)),
            "denominator": submitted,
            "missing_sample_count": 0,
        },
        "failure_rate": sum(failures) / submitted,
        "failure_rate_detail": {
            "value": sum(failures) / submitted,
            "numerator": int(sum(failures)),
            "denominator": submitted,
            "missing_sample_count": 0,
        },
        "reasoning_emission_rate": reasoning_count / len(answer_samples) if answer_samples else None,
        "reasoning_emission_detail": {
            "numerator": reasoning_count,
            "denominator": len(answer_samples),
            "missing_sample_count": submitted - len(answer_samples),
        },
    }
    names = {"timeout": "timeout", "429": "http_429", "other_4xx": "other_http_4xx", "5xx": "http_5xx", "transport_error": "transport_error"}
    for outcome, label in names.items():
        count = outcomes.count(outcome)
        row[f"{label}_rate"] = count / submitted
        row[f"{label}_count"] = count
    for name, labels in PERCENTILES.items():
        summary = _percentile_summary(name, series[name], submitted, labels)
        row[name] = summary
        prefix = name.removesuffix("_ms")
        for label, value in summary["percentiles"].items():
            row[f"{prefix}_{label}"] = value["value"]
    usage, cost = _usage_cost(records)
    row["usage"] = usage
    row["cost"] = cost
    if family == CONTRACT_TASK:
        counts = [
            value
            for evaluation in evaluations
            if isinstance((value := evaluation.get("correct_field_count")), int) and not isinstance(value, bool)
        ]
        row["contract_field_diagnostics"] = {
            "field_count_total": len(CONTRACT_FIELDS),
            "returned_object_count": len(counts),
            "median_correct_fields_returned": median(counts) if counts else None,
        }
    return row, {
        "series": series,
        "correct": correct,
        "failures": failures,
        "reasoning_count": reasoning_count,
        "reasoning_denominator": len(answer_samples),
    }


def _flag(source: Mapping[str, Any] | None, provider: str, key: str, default: bool) -> bool:
    if source is None or provider not in source:
        return default
    value = source[provider]
    if isinstance(value, Mapping):
        return value.get(key, value.get("complete", value.get("compatible", default))) is True
    return value is True


def _record_compatibility(record: Mapping[str, Any]) -> bool:
    """Honor explicit saved compatibility evidence when caller maps are absent."""

    for key in ("configuration_compatible", "provider_configuration_compatible"):
        if key in record:
            return record.get(key) is True
    identity = record.get("model_identity")
    if isinstance(identity, Mapping) and "matches" in identity and identity.get("matches") is not True:
        return False
    adapter = record.get("adapter_compatibility")
    if isinstance(adapter, Mapping) and adapter.get("status") in {
        "unsupported", "request_failed", "incompatible"
    }:
        return False
    checks = record.get("configuration_checks")
    if isinstance(checks, Mapping) and any(value is not True for value in checks.values()):
        return False
    return True


def _record_artifact_complete(record: Mapping[str, Any]) -> bool:
    for key in (
        "artifact_graph_complete",
        "local_artifact_graph_complete",
        "artifact_complete",
    ):
        if key in record:
            return record.get(key) is True
    return True


def _paired_provider_eligibility(
    records: Sequence[Mapping[str, Any]],
    providers: Sequence[str],
    compatibility: Mapping[str, Any] | None,
    artifact_completeness: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    observed_rounds = {_round_key(row) for row in records}
    for provider in providers:
        provider_records = [row for row in records if row.get("provider") == provider]
        compatible = all(_record_compatibility(row) for row in provider_records)
        if compatibility is not None and provider in compatibility:
            compatible = compatible and _flag(compatibility, provider, "compatible", False)
        complete = (
            all(_record_artifact_complete(row) for row in provider_records)
            and {_round_key(row) for row in provider_records} == observed_rounds
        )
        if artifact_completeness is not None and provider in artifact_completeness:
            complete = complete and _flag(
                artifact_completeness, provider, "complete", False
            )
        reasons: list[str] = []
        if not compatible:
            reasons.append("incompatible model or request configuration")
        if not complete:
            reasons.append("incomplete or internally inconsistent local artifact graph")
        output[provider] = {
            "configuration_compatible": compatible,
            "artifact_graph_complete": complete,
            "eligible": not reasons,
            "exclusion_reasons": reasons,
        }
    return output


def compute_metrics(
    records: Iterable[Mapping[str, Any]],
    evaluations: Iterable[Mapping[str, Any]],
    *,
    benchmark_slug: str,
    run_id: str,
    provider_compatibility: Mapping[str, Any] | None = None,
    artifact_completeness: Mapping[str, Any] | None = None,
    expected_task_families: Sequence[str] = TASK_FAMILIES,
) -> dict[str, Any]:
    """Recompute task-family metrics and pooled balanced provider aggregates."""

    raw_by_id = _index(records, "raw record")
    evaluation_by_id = _index(evaluations, "evaluation")
    if set(raw_by_id) != set(evaluation_by_id):
        raise MetricsContractError(
            f"raw/evaluation identity mismatch: missing={sorted(set(raw_by_id)-set(evaluation_by_id))}, extra={sorted(set(evaluation_by_id)-set(raw_by_id))}"
        )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for request_id, record in raw_by_id.items():
        provider, family = record.get("provider"), record.get("task_family")
        if not isinstance(provider, str) or not provider or family not in expected_task_families:
            raise MetricsContractError(f"raw record {request_id!r} has invalid provider/task family")
        evaluation = evaluation_by_id[request_id]
        if evaluation.get("provider") != provider or evaluation.get("task_family") != family:
            raise MetricsContractError(f"evaluation identity mismatch for {request_id!r}")
        grouped.setdefault((provider, family), []).append(record)
    if not grouped:
        raise MetricsContractError("at least one terminal record is required")

    family_rows: list[dict[str, Any]] = []
    private: dict[tuple[str, str], dict[str, Any]] = {}
    for (provider, family), rows in sorted(grouped.items()):
        evals = [evaluation_by_id[row["request_id"]] for row in rows]
        metric, samples = _family_metric(provider, family, rows, evals, slug=benchmark_slug, run_id=run_id)
        family_rows.append(metric)
        private[(provider, family)] = samples

    providers = sorted({provider for provider, _ in grouped})
    combined: list[dict[str, Any]] = []
    for provider in providers:
        rows = {row["task_family"]: row for row in family_rows if row["provider"] == provider}
        missing_families = [family for family in expected_task_families if family not in rows]
        available = [rows[family] for family in expected_task_families if family in rows]
        row: dict[str, Any] = {
            "provider": provider,
            "weighting": "pooled_balanced_requests",
            "task_family_count": len(available),
            "expected_task_family_count": len(expected_task_families),
            "missing_task_families": missing_families,
            "submitted_count": sum(value["submitted_count"] for value in available),
        }
        submitted = row["submitted_count"]
        row["task_success_rate"] = (
            sum(value["correct_count"] for value in available) / submitted if submitted else None
        )
        row["failure_rate"] = (
            sum(value["failed_request_count"] for value in available) / submitted if submitted else None
        )
        for label in ("timeout", "http_429", "other_http_4xx", "http_5xx", "transport_error"):
            count = sum(value[f"{label}_count"] for value in available)
            row[f"{label}_count"] = count
            row[f"{label}_rate"] = count / submitted if submitted else None
        reasoning_count = sum(private[(provider, family)]["reasoning_count"] for family in rows)
        reasoning_denominator = sum(
            private[(provider, family)]["reasoning_denominator"] for family in rows
        )
        row["reasoning_emission_rate"] = (
            reasoning_count / reasoning_denominator if reasoning_denominator else None
        )
        row["reasoning_emission_detail"] = {
            "numerator": reasoning_count,
            "denominator": reasoning_denominator,
            "missing_sample_count": submitted - reasoning_denominator,
        }
        for name, labels in PERCENTILES.items():
            prefix = name.removesuffix("_ms")
            pooled = [
                sample
                for family in rows
                for sample in private[(provider, family)]["series"][name]
            ]
            for label in labels:
                row[f"{prefix}_{label}"] = nearest_rank(pooled, _metric_quantile(name, label))
            row[name] = _percentile_summary(name, pooled, submitted, labels)
        row["usage"] = {
            "reported_count": sum(value["usage"]["reported_count"] for value in available),
            "missing_count": sum(value["usage"]["missing_count"] for value in available),
            "input_tokens": sum(value["usage"]["input_tokens"] for value in available),
            "output_tokens": sum(value["usage"]["output_tokens"] for value in available),
        }
        row["usage"]["coverage_rate"] = row["usage"]["reported_count"] / row["submitted_count"] if row["submitted_count"] else None
        row["cost"] = {
            "reported_count": sum(value["cost"]["reported_count"] for value in available),
            "missing_count": sum(value["cost"]["missing_count"] for value in available),
            "known_cost_usd": sum(value["cost"]["known_cost_usd"] for value in available),
        }
        row["cost"]["coverage_rate"] = row["cost"]["reported_count"] / row["submitted_count"] if row["submitted_count"] else None
        compatible = _flag(provider_compatibility, provider, "compatible", True)
        family_counts = [value["submitted_count"] for value in available]
        balanced = bool(family_counts) and len(set(family_counts)) == 1 and not missing_families
        complete = (
            _flag(artifact_completeness, provider, "complete", True)
            and not missing_families
            and balanced
        )
        reasons: list[str] = []
        if row["task_success_rate"] is None or row["task_success_rate"] < 0.95:
            reasons.append("task success rate below 95%")
        if row["failure_rate"] is None or row["failure_rate"] > 0.01:
            reasons.append("operational failure rate above 1%")
        if not compatible:
            reasons.append("incompatible model or request configuration")
        if not complete:
            reasons.append("incomplete, unbalanced, or internally inconsistent local artifact graph")
        row.update(configuration_compatible=compatible, artifact_graph_complete=complete, eligible=not reasons, ineligibility_reasons=reasons, rank=None)
        combined.append(row)
    eligible = sorted((row for row in combined if row["eligible"]), key=lambda value: (value["e2e_latency_p99"] is None, value["e2e_latency_p99"] or math.inf, value["provider"]))
    for rank, row in enumerate(eligible, 1):
        if row["e2e_latency_p99"] is not None:
            row["rank"] = rank
    combined.sort(key=lambda value: (value["rank"] is None, value["rank"] or math.inf, value["provider"]))
    return {
        "metrics_version": METRICS_VERSION,
        "benchmark_slug": benchmark_slug,
        "run_id": run_id,
        "task_family_metrics": family_rows,
        "combined_provider_metrics": combined,
        "ranking": [row["provider"] for row in combined if row["rank"] is not None],
    }


def _round_key(record: Mapping[str, Any]) -> str:
    for key in ("round_id", "round_index", "item_id"):
        value = record.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return str(value)
    raise MetricsContractError("paired record requires round_id, round_index, or item_id")


def compute_paired_comparisons(
    records: Iterable[Mapping[str, Any]],
    *,
    benchmark_slug: str,
    run_id: str,
    provider_compatibility: Mapping[str, Any] | None = None,
    artifact_completeness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute paired E2E-latency analysis using only eligible provider samples.

    Existing callers need not pass the optional maps. In that case explicit
    record-level compatibility and artifact evidence is honored.
    """

    rows = list(records)
    providers = sorted(
        {str(row.get("provider")) for row in rows if isinstance(row.get("provider"), str)}
    )
    eligibility = _paired_provider_eligibility(
        rows, providers, provider_compatibility, artifact_completeness
    )
    by_round: dict[str, dict[str, float | None]] = {}
    for row in rows:
        provider = row.get("provider")
        if provider not in providers:
            raise MetricsContractError("paired record has invalid provider")
        key = _round_key(row)
        if provider in by_round.setdefault(key, {}):
            raise MetricsContractError(f"duplicate provider {provider!r} in round {key!r}")
        by_round[key][provider] = (
            recompute_timing(row)["e2e_latency_ms"] if eligibility[provider]["eligible"] else None
        )

    wins = {provider: 0.0 for provider in providers}
    outright = {provider: 0 for provider in providers}
    ties = {provider: 0 for provider in providers}
    eligible_rounds = 0
    for values in by_round.values():
        present = {provider: value for provider, value in values.items() if value is not None}
        if not present:
            continue
        eligible_rounds += 1
        best = min(present.values())
        winners = [provider for provider, value in present.items() if value == best]
        for provider in winners:
            wins[provider] += 1 / len(winners)
            if len(winners) == 1:
                outright[provider] += 1
            else:
                ties[provider] += 1

    win_rows: list[dict[str, Any]] = []
    for provider in providers:
        sample_count = sum(values.get(provider) is not None for values in by_round.values())
        provider_eligible = eligibility[provider]["eligible"]
        win_rows.append(
            {
                "provider": provider,
                **eligibility[provider],
                "credited_wins": wins[provider],
                "outright_wins": outright[provider],
                "tied_fastest_rounds": ties[provider],
                "eligible_round_count": eligible_rounds,
                "eligible_e2e_sample_count": sample_count,
                "excluded_or_missing_sample_count": len(by_round) - sample_count,
                "total_round_count": len(by_round),
                "win_rate": (
                    wins[provider] / eligible_rounds
                    if provider_eligible and eligible_rounds
                    else None
                ),
            }
        )

    pairwise: list[dict[str, Any]] = []
    for left in providers:
        for right in providers:
            if left >= right:
                continue
            pair_eligible = eligibility[left]["eligible"] and eligibility[right]["eligible"]
            shared = [
                values[left] - values[right]  # type: ignore[operator]
                for values in by_round.values()
                if pair_eligible
                and values.get(left) is not None
                and values.get(right) is not None
            ]
            exclusion_reasons = sorted(
                {
                    f"{provider}: {reason}"
                    for provider in (left, right)
                    for reason in eligibility[provider]["exclusion_reasons"]
                }
            )
            row: dict[str, Any] = {
                "provider_a": left,
                "provider_b": right,
                "difference_definition": "provider_a_e2e_latency_ms - provider_b_e2e_latency_ms",
                "eligible_comparison": pair_eligible,
                "exclusion_reasons": exclusion_reasons,
                "paired_count": len(shared),
                "total_round_count": len(by_round),
                "missing_pair_count": len(by_round) - len(shared),
                "coverage_rate": len(shared) / len(by_round) if by_round else None,
            }
            for label in ("p50", "p95", "p99"):
                quantile = int(label[1:]) / 100
                row[f"difference_{label}_ms"] = nearest_rank(shared, quantile)
            pairwise.append(row)
    return {
        "metrics_version": METRICS_VERSION,
        "benchmark_slug": benchmark_slug,
        "run_id": run_id,
        "provider_eligibility": [
            {"provider": provider, **eligibility[provider]} for provider in providers
        ],
        "fastest_provider_wins": win_rows,
        "pairwise_e2e_latency_differences": pairwise,
    }


__all__ = [
    "METRICS_VERSION",
    "MetricsContractError",
    "compute_metrics",
    "compute_paired_comparisons",
    "nearest_rank",
    "recompute_timing",
]
