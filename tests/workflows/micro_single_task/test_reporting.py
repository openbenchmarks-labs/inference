from pathlib import Path

from openbenchmarks_inference.workflows.micro_single_task.reporting import (
    REQUIRED_METRIC_COLUMNS,
    render_report,
)


def _row(provider="fast", family=None):
    row = {
        "provider": provider,
        "rank": 1,
        "eligible": True,
        "ineligibility_reasons": [],
        "submitted_count": 3,
        "ttfa_ms": {"population_count": 3},
        "e2e_latency_ms": {"population_count": 3},
        "output_tokens_per_second": {"population_count": 3},
        "usage": {"reported_count": 3},
        "cost": {"reported_count": 3, "known_cost_usd": 0.003},
    }
    for column in REQUIRED_METRIC_COLUMNS:
        row[column] = 0.0 if column.endswith("_rate") else 100.0
    row["task_success_rate"] = 1.0
    row["reasoning_emission_rate"] = 2 / 3
    if family:
        row["task_family"] = family
    return row


def test_report_contains_all_tables_and_provenance():
    families = ("meeting-notes-lookup", "ticket-triage", "contract-terms-extraction")
    report = render_report(
        benchmark_name="Inference Micro Single-Task Benchmark",
        run_id="run-test",
        status="smoke_passed_awaiting_human_review",
        metrics={
            "combined_provider_metrics": [_row()],
            "task_family_metrics": [_row(family=family) for family in families],
        },
        paired_comparisons={
            "fastest_provider_wins": [{"provider": "fast", "win_rate": 1.0, "credited_wins": 3, "outright_wins": 3, "tied_fastest_rounds": 0, "eligible_round_count": 3, "total_round_count": 3}],
            "pairwise_e2e_latency_differences": [{"provider_a": "fast", "provider_b": "slow", "difference_p50_ms": -10, "difference_p95_ms": -5, "difference_p99_ms": -5, "paired_count": 3, "total_round_count": 3, "coverage_rate": 1.0}],
        },
        manifest={"expected_requests": 30, "raw_records": 30, "complete": True},
        provider_reasoning_mappings={"fast": {"reasoning_effort": "low"}},
        execution_metadata={"execution_region": "local", "period": "2026-09-02"},
        selection_hash="abc123",
        raw_event_completeness={"complete": True},
        resume_history=[],
        run_dir=Path("/tmp/run-test"),
        smoke_attempt=1,
        warnings=("sample warning",),
    )
    assert report.index("## E2E latency comparison") < report.index("## Balanced pooled")
    assert "headline" not in report.lower()
    for family in families:
        assert f"### {family}" in report
    for column in REQUIRED_METRIC_COLUMNS:
        assert column in report
    assert "E2E latency p99" in report
    assert "[95% CI]" not in report
    assert "TTFR" not in report
    assert "Fastest-provider wins" in report
    assert "provider A E2E latency - provider B E2E latency" in report
    assert "tiktoken o200k_base" in report
    assert "performs no database writes or publication steps" in report
    assert "sample warning" in report


def test_ineligible_provider_remains_visible_without_rank_and_with_reason():
    row = _row("failed")
    row.update(rank=None, eligible=False, ineligibility_reasons=["task success rate below 95%"])
    report = render_report(
        benchmark_name="Benchmark",
        run_id="run",
        status="smoke_failed_after_retry",
        metrics={"combined_provider_metrics": [row], "task_family_metrics": []},
        paired_comparisons={},
        manifest={"complete": False},
    )
    assert "| — | failed |" in report
    assert "task success rate below 95%" in report
    assert "performs no database writes or publication steps" in report
