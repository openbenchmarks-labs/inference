"""Human-readable reporting for micro-single-task local artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import TASK_FAMILIES


REQUIRED_METRIC_COLUMNS = (
    "ttfo_p50", "ttfo_p95", "ttfo_p99",
    "ttfa_p50", "ttfa_p95", "ttfa_p99",
    "pre_answer_reasoning_tokens_p50", "pre_answer_reasoning_tokens_p95",
    "pre_answer_reasoning_tokens_p99", "reasoning_emission_rate",
    "e2e_latency_p50", "e2e_latency_p95", "e2e_latency_p99",
    "output_tokens_per_second_p50", "output_tokens_per_second_p95", "output_tokens_per_second_p99",
    "task_success_rate", "failure_rate", "timeout_rate", "http_429_rate",
    "other_http_4xx_rate", "http_5xx_rate", "transport_error_rate",
)


def _value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rate(value: Any) -> str:
    return "unavailable" if value is None else f"{100 * float(value):.2f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def _metric_row(row: Mapping[str, Any], *, combined: bool = False) -> list[str]:
    prefix = [str(row.get("rank") or "—"), str(row.get("provider", "unknown"))] if combined else [str(row.get("provider", "unknown"))]
    values: list[str] = []
    for column in REQUIRED_METRIC_COLUMNS:
        value = row.get(column)
        if column.endswith("_rate"):
            values.append(_rate(value))
        elif column.endswith(("_p50", "_p95", "_p99")):
            values.append(_value(value))
        else:
            values.append(_value(value))
    return prefix + values


def render_report(
    *,
    benchmark_name: str,
    run_id: str,
    status: str,
    metrics: Mapping[str, Any],
    paired_comparisons: Mapping[str, Any],
    manifest: Mapping[str, Any],
    task_families: Sequence[str] = TASK_FAMILIES,
    tokenizer: str | Mapping[str, Any] = "tiktoken o200k_base",
    execution_metadata: Mapping[str, Any] | None = None,
    provider_reasoning_mappings: Mapping[str, Any] | None = None,
    selection_hash: str | None = None,
    raw_event_completeness: Mapping[str, Any] | bool | None = None,
    resume_history: Sequence[Any] | Mapping[str, Any] | None = None,
    run_dir: Path | str | None = None,
    smoke_attempt: int | None = None,
    warnings: Sequence[str] = (),
) -> str:
    """Render the complete local smoke/full-run report as Markdown."""

    execution_metadata = execution_metadata or {}
    provider_reasoning_mappings = provider_reasoning_mappings or {}
    combined = list(metrics.get("combined_provider_metrics", ()))
    family_rows = list(metrics.get("task_family_metrics", ()))
    lines = [
        f"# {benchmark_name}",
        "",
        f"Run `{run_id}` — **{status}**",
        "",
        "This report was derived from local run artifacts.",
        "",
        "## E2E latency comparison",
        "",
        "E2E values are milliseconds and are computed directly from each provider's balanced pooled requests. Only eligible providers receive ranks.",
        "",
    ]
    e2e_rows = [
        [
            row.get("rank") or "—",
            row.get("provider"),
            _value(row.get("e2e_latency_p99")),
            "eligible" if row.get("eligible") else "ineligible",
            "; ".join(row.get("ineligibility_reasons", ())) or "none",
            f"{row.get('e2e_latency_ms', {}).get('population_count', 0)}/{row.get('submitted_count', 0)}",
        ]
        for row in combined
    ]
    lines.extend(_table(("Rank", "Provider", "E2E latency p99", "Eligibility", "Reasons", "E2E coverage"), e2e_rows or [["—", "No provider rows", "—", "—", "—", "0/0"]]))
    lines += ["", "## Balanced pooled provider summary", ""]
    headers = ("Rank", "Provider", *REQUIRED_METRIC_COLUMNS)
    lines.extend(_table(headers, [_metric_row(row, combined=True) for row in combined] or [["—"] * len(headers)]))
    lines += [
        "",
        "Combined percentiles are computed directly over pooled request samples. Pooling is accepted only when every task family contributes the same required request count.",
        "",
        "## Task-family results",
        "",
    ]
    for family in task_families:
        lines += [f"### {family}", ""]
        rows = [row for row in family_rows if row.get("task_family") == family]
        rows.sort(key=lambda row: (row.get("e2e_latency_p99") is None, row.get("e2e_latency_p99") or float("inf"), str(row.get("provider"))))
        family_headers = ("Provider", *REQUIRED_METRIC_COLUMNS)
        lines.extend(_table(family_headers, [_metric_row(row) for row in rows] or [["—"] * len(family_headers)]))
        lines.append("")
        lines.append("Coverage and cost:")
        lines.append("")
        coverage_rows = []
        for row in rows:
            coverage_rows.append([
                row.get("provider"),
                f"{row.get('ttfa_ms', {}).get('population_count', 0)}/{row.get('submitted_count', 0)}",
                f"{row.get('e2e_latency_ms', {}).get('population_count', 0)}/{row.get('submitted_count', 0)}",
                f"{row.get('output_tokens_per_second', {}).get('population_count', 0)}/{row.get('submitted_count', 0)}",
                f"{row.get('usage', {}).get('reported_count', 0)}/{row.get('submitted_count', 0)}",
                f"{row.get('cost', {}).get('reported_count', 0)}/{row.get('submitted_count', 0)}",
                _value(row.get("cost", {}).get("known_cost_usd"), 6),
            ])
        lines.extend(_table(("Provider", "TTFA", "E2E", "Output speed", "Usage", "Cost", "Known USD"), coverage_rows or [["—"] * 7]))
        diagnostics = [row for row in rows if row.get("contract_field_diagnostics")]
        if diagnostics:
            lines += ["", "Contract extraction field diagnostics:", ""]
            lines.extend(_table(("Provider", "Returned objects", "Median correct fields", "Total fields"), [[row.get("provider"), row["contract_field_diagnostics"].get("returned_object_count"), _value(row["contract_field_diagnostics"].get("median_correct_fields_returned")), row["contract_field_diagnostics"].get("field_count_total")] for row in diagnostics]))
        lines.append("")

    wins = list(paired_comparisons.get("fastest_provider_wins", ()))
    pairs = list(paired_comparisons.get("pairwise_e2e_latency_differences", ()))
    lines += ["## Paired temporal analysis", "", "Fastest-provider wins split credit equally across exact ties.", ""]
    lines.extend(_table(("Provider", "Win rate", "Credited wins", "Outright", "Tied", "Covered rounds"), [[row.get("provider"), _rate(row.get("win_rate")), _value(row.get("credited_wins")), row.get("outright_wins"), row.get("tied_fastest_rounds"), f"{row.get('eligible_round_count', 0)}/{row.get('total_round_count', 0)}"] for row in wins] or [["—"] * 6]))
    lines += ["", "Pairwise differences are `provider A E2E latency - provider B E2E latency`; negative values favor provider A.", ""]
    lines.extend(_table(("Provider A", "Provider B", "Median diff", "p95 diff", "p99 diff", "Paired coverage"), [[row.get("provider_a"), row.get("provider_b"), _value(row.get("difference_p50_ms")), _value(row.get("difference_p95_ms")), _value(row.get("difference_p99_ms")), f"{row.get('paired_count', 0)}/{row.get('total_round_count', 0)} ({_rate(row.get('coverage_rate'))})"] for row in pairs] or [["—"] * 6]))

    location = execution_metadata.get("execution_region", execution_metadata.get("region", "unavailable"))
    period = execution_metadata.get("wall_clock_period", execution_metadata.get("period", "unavailable"))
    lines += [
        "",
        "## Provenance and interpretation",
        "",
        f"- Local tokenizer: `{tokenizer}`.",
        f"- Execution location: `{location}`.",
        f"- Execution period: `{period}`.",
        "- Connection policy: one persistent connection pool per provider; no hidden warmup.",
        "- Scheduling: paired question rounds, shared release barrier, per-provider concurrency one.",
        "- First visible answer content may be JSON punctuation because JSON-object mode is used.",
        f"- Selected-item hash: `{selection_hash or manifest.get('selection_hash', 'unavailable')}`.",
        f"- Raw-event completeness: `{raw_event_completeness if raw_event_completeness is not None else manifest.get('raw_event_completeness', 'unavailable')}`.",
        f"- Resume history: `{resume_history if resume_history is not None else manifest.get('resume_history', 'none')}`.",
        "- Percentiles use nearest-rank; confidence-interval bounds are not calculated or reported.",
        "- Failed and invalid requests remain in submitted denominators and are disclosed in coverage counts.",
        "",
        "Reasoning-effort wire mappings:",
        "",
    ]
    if provider_reasoning_mappings:
        lines.extend(_table(("Provider", "Wire mapping"), [[provider, f"`{mapping}`"] for provider, mapping in sorted(provider_reasoning_mappings.items())]))
    else:
        lines.append("No provider mapping evidence was supplied to the report renderer.")

    planned = manifest.get("planned_requests", manifest.get("expected_requests", "unavailable"))
    executed = manifest.get("executed_requests", manifest.get("submitted_requests", len(manifest.get("request_ids", ())) or "unavailable"))
    completed = manifest.get("completed_requests", manifest.get("raw_records", "unavailable"))
    valid = manifest.get("valid_results", sum(row.get("correct_count", 0) for row in family_rows))
    invalid = manifest.get("invalid_results", sum(row.get("invalid_result_count", 0) for row in family_rows))
    failed = manifest.get("failed_requests", sum(row.get("failed_request_count", 0) for row in family_rows))
    lines += [
        "",
        "## Local artifact and smoke status",
        "",
        f"- Smoke attempt: `{smoke_attempt if smoke_attempt is not None else 'not specified'}`.",
        f"- Planned / executed / completed: `{planned}` / `{executed}` / `{completed}`.",
        f"- Valid exact-correct / invalid / operationally failed: `{valid}` / `{invalid}` / `{failed}`.",
        f"- Manifest complete: `{manifest.get('complete', False)}`.",
        f"- Local run directory: `{run_dir or 'unavailable'}`.",
        "- This OSS runner performs no database writes or publication steps.",
    ]
    all_warnings = list(warnings) + list(manifest.get("warnings", ()))
    if all_warnings:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in all_warnings]
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["REQUIRED_METRIC_COLUMNS", "render_report"]
