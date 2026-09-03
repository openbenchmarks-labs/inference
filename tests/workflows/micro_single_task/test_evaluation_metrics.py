from __future__ import annotations

import json

import pytest

from openbenchmarks_inference.workflows.micro_single_task.evaluation import (
    CONTRACT_FIELDS,
    EvaluationContractError,
    MalformedJSONError,
    evaluate,
    evaluate_record,
    parse_exact_json,
    terminal_outcome,
)
from openbenchmarks_inference.workflows.micro_single_task.metrics import (
    MetricsContractError,
    compute_metrics,
    compute_paired_comparisons,
    nearest_rank,
    recompute_timing,
)


CONTRACT_GOLD = {
    "agreement_type": "saas_subscription",
    "effective_date": "2026-01-01",
    "initial_term_end_date": "2026-12-31",
    "auto_renews": True,
    "renewal_term_months": 12,
    "non_renewal_notice_days": 30,
    "payment_terms_days": None,
    "termination_for_convenience": False,
    "termination_notice_days": None,
    "security_incident_notice_hours": 24,
}


def test_exact_json_and_all_three_task_schemas_are_strict():
    assert parse_exact_json(' {"answer":"Ada"}\n') == {"answer": "Ada"}
    for malformed in ("", "```json\n{}\n```", "{} {}", '{"x":NaN}', '{"x":1,"x":2}'):
        with pytest.raises(MalformedJSONError):
            parse_exact_json(malformed)

    meeting = evaluate(
        "meeting-notes-lookup",
        '{"answer":"  ADA   LOVELACE "}',
        {"type": "person", "value": "Augusta King"},
        ["Ada Lovelace"],
    )
    assert meeting["classification"] == "valid_result"
    assert evaluate("meeting-notes-lookup", '{"answer":1}', {"value": "1"})["error_type"] == "schema_invalid"

    ticket_gold = {"category": "bug", "priority": "p1"}
    wrong = evaluate("ticket-triage", '{"category":"bug","priority":"p2"}', ticket_gold)
    assert wrong["classification"] == "incorrect_result" and wrong["schema_valid"] is True
    assert evaluate("ticket-triage", '{"category":"Bug","priority":"p1"}', ticket_gold)["classification"] == "invalid_result"

    changed = dict(CONTRACT_GOLD)
    changed["renewal_term_months"] = True
    contract = evaluate("contract-terms-extraction", json.dumps(changed), CONTRACT_GOLD)
    assert contract["classification"] == "invalid_result"
    assert contract["correct_field_count"] == len(CONTRACT_FIELDS) - 1
    assert contract["field_correct"]["renewal_term_months"] is False


def test_operational_outcomes_and_record_evaluation_are_mutually_exclusive():
    assert terminal_outcome({"terminal_outcome": "completed", "http_status": 200}) == "2xx"
    assert terminal_outcome({"terminal_outcome": "http_error", "http_status": 429}) == "429"
    assert terminal_outcome({"terminal_outcome": "http_error", "http_status": 404}) == "other_4xx"
    assert terminal_outcome({"terminal_outcome": "http_error", "http_status": 503}) == "5xx"
    assert terminal_outcome({"terminal_outcome": "timeout"}) == "timeout"
    assert terminal_outcome({"terminal_outcome": "transport_error"}) == "transport_error"
    with pytest.raises(EvaluationContractError):
        terminal_outcome({"terminal_outcome": "runner_error"})
    assert terminal_outcome({
        "terminal_outcome": "runner_error",
        "http_status": None,
        "error": {
            "type": "AttributeError",
            "message": "'NoneType' object has no attribute 'read'",
        },
    }) == "transport_error"

    item = {"use_case": "ticket-triage", "gold": {"category": "bug", "priority": "p1"}, "aliases": {}}
    base = {"request_id": "r", "provider": "p", "task_family": "ticket-triage"}
    failed = evaluate_record(item, {**base, "terminal_outcome": "http_error", "http_status": 500})
    assert failed["classification"] == "failed_request"
    assert failed["error_type"] == "http_5xx"
    invalid = evaluate_record(item, {**base, "terminal_outcome": "completed", "http_status": 200, "reconstructed_answer": "oops"})
    assert invalid["classification"] == "invalid_result"
    assert invalid["error_type"] == "malformed_json"


def _events(first: float, last: float, answer: str = '{"answer":"Ada"}'):
    return [
        {"monotonic_offset_ms": first - 10, "classification": "reasoning_delta", "content": "think"},
        {"monotonic_offset_ms": first, "classification": "answer_delta", "content": answer[:1]},
        {"monotonic_offset_ms": last, "classification": "answer_delta", "content": answer[1:]},
        {"monotonic_offset_ms": last + 1, "classification": "finish"},
    ]


def _record(provider: str, family: str, round_id: int, ttfa: float, *, outcome="completed", status=200):
    answer = '{"answer":"Ada"}'
    return {
        "request_id": f"{provider}-{family}-{round_id}",
        "provider": provider,
        "task_family": family,
        "round_id": str(round_id),
        "terminal_outcome": outcome,
        "http_status": status,
        "stream_events": _events(ttfa, ttfa + 100, answer) if outcome == "completed" else [],
        "reconstructed_reasoning": "think" if outcome == "completed" else "",
        "reconstructed_answer": answer if outcome == "completed" else "",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5} if outcome == "completed" else None,
        "cost_usd": 0.01 if outcome == "completed" else None,
    }


def _evaluation(record, *, correct=True, classification=None):
    classification = classification or ("valid_result" if correct else "incorrect_result")
    return {
        "request_id": record["request_id"],
        "provider": record["provider"],
        "task_family": record["task_family"],
        "classification": classification,
        "schema_valid": classification in {"valid_result", "incorrect_result"},
        "correct": classification == "valid_result",
    }


def test_stream_evidence_recomputes_boundaries_and_output_speed():
    record = _record("p", "meeting-notes-lookup", 1, 125)
    timing = recompute_timing(record)
    assert timing["source"] == "stream_events"
    assert "ttfr_ms" not in timing
    assert timing["ttfo_ms"] == 115
    assert timing["ttfa_ms"] == 125
    assert timing["pre_answer_reasoning_tokens"] >= 1
    assert timing["reasoning_emitted_before_answer"] is True
    assert timing["e2e_latency_ms"] == 225
    assert timing["visible_answer_token_count"] >= 2
    assert timing["output_tokens_per_second"] > 0
    contradictory = dict(record, reconstructed_answer="patched")
    with pytest.raises(MetricsContractError, match="do not reconstruct"):
        recompute_timing(contradictory)


def test_no_reasoning_makes_first_output_equal_first_answer():
    record = _record("p", "meeting-notes-lookup", 1, 125)
    record["stream_events"] = [
        event for event in record["stream_events"] if event["classification"] != "reasoning_delta"
    ]
    record["reconstructed_reasoning"] = ""
    timing = recompute_timing(record)
    assert timing["ttfo_ms"] == timing["ttfa_ms"] == 125
    assert timing["pre_answer_reasoning_tokens"] == 0
    assert timing["reasoning_emitted_before_answer"] is False


def test_pre_output_failure_keeps_reasoning_emission_unavailable():
    record = _record(
        "p", "meeting-notes-lookup", 1, 0, outcome="timeout", status=None
    )
    timing = recompute_timing(record)
    assert timing["ttfo_ms"] is None
    assert timing["ttfa_ms"] is None
    assert timing["pre_answer_reasoning_tokens"] is None
    assert timing["reasoning_emitted_before_answer"] is None


def test_nearest_rank_family_metrics_pooled_eligibility_and_coverage_are_deterministic():
    assert nearest_rank(list(range(1, 101)), 0.99) == 99
    families = ("meeting-notes-lookup", "ticket-triage", "contract-terms-extraction")
    records = []
    evaluations = []
    for provider, base in (("fast", 100), ("slow", 200)):
        for index, family in enumerate(families):
            record = _record(provider, family, index, base + index)
            records.append(record)
            evaluations.append(_evaluation(record))
    first = compute_metrics(records, evaluations, benchmark_slug="slug", run_id="run")
    second = compute_metrics(records, evaluations, benchmark_slug="slug", run_id="run")
    assert first == second
    assert first["ranking"] == ["fast", "slow"]
    assert len(first["task_family_metrics"]) == 6
    fast = next(row for row in first["combined_provider_metrics"] if row["provider"] == "fast")
    assert fast["ttfa_p99"] == pytest.approx(102)
    assert fast["e2e_latency_p99"] == pytest.approx(202)
    assert fast["weighting"] == "pooled_balanced_requests"
    assert fast["reasoning_emission_rate"] == 1
    assert fast["task_success_rate"] == 1
    assert fast["eligible"] is True
    assert "output_tokens_per_second_p95" in fast
    assert "output_tokens_per_second_p99" in fast
    family = next(row for row in first["task_family_metrics"] if row["provider"] == "fast")
    assert family["ttfa_ms"]["population_count"] == 1
    assert "ttfr_population_rate" not in family
    assert family["usage"]["coverage_rate"] == 1


def test_failures_lower_success_and_eligibility_but_not_latency_population():
    families = ("meeting-notes-lookup", "ticket-triage", "contract-terms-extraction")
    records = []
    evaluations = []
    for index, family in enumerate(families):
        record = _record("p", family, index, 100)
        if index == 1:
            record = _record("p", family, index, 100, outcome="http_error", status=429)
            evaluation = _evaluation(record, classification="failed_request")
        else:
            evaluation = _evaluation(record)
        records.append(record)
        evaluations.append(evaluation)
    result = compute_metrics(records, evaluations, benchmark_slug="slug", run_id="run")
    combined = result["combined_provider_metrics"][0]
    assert combined["task_success_rate"] == pytest.approx(2 / 3)
    assert combined["failure_rate"] == pytest.approx(1 / 3)
    assert combined["http_429_rate"] == pytest.approx(1 / 3)
    assert combined["eligible"] is False
    assert "operational failure rate above 1%" in combined["ineligibility_reasons"]
    ticket = next(row for row in result["task_family_metrics"] if row["task_family"] == "ticket-triage")
    assert ticket["ttfa_ms"]["population_count"] == 0
    assert ticket["ttfa_ms"]["missing_sample_count"] == 1


def test_pairwise_differences_and_fastest_wins_report_missing_pair_coverage():
    records = [
        _record("a", "meeting-notes-lookup", 1, 100),
        _record("b", "meeting-notes-lookup", 1, 120),
        _record("a", "ticket-triage", 2, 150),
        _record("b", "ticket-triage", 2, 150),
        _record("a", "contract-terms-extraction", 3, 200),
        _record("b", "contract-terms-extraction", 3, 0, outcome="timeout", status=None),
    ]
    result = compute_paired_comparisons(records, benchmark_slug="slug", run_id="run")
    pair = result["pairwise_e2e_latency_differences"][0]
    assert pair["paired_count"] == 2
    assert pair["missing_pair_count"] == 1
    assert pair["coverage_rate"] == pytest.approx(2 / 3)
    assert pair["difference_p50_ms"] == -20
    wins = {row["provider"]: row for row in result["fastest_provider_wins"]}
    assert wins["a"]["credited_wins"] == 2.5
    assert wins["b"]["credited_wins"] == 0.5


def test_paired_analysis_excludes_incompatible_and_incomplete_providers_with_coverage():
    records = [
        _record(provider, "meeting-notes-lookup", round_id, ttfa)
        for round_id, values in enumerate(((100, 90, 80), (110, 95, 85)), 1)
        for provider, ttfa in zip(("eligible", "incompatible", "incomplete"), values)
    ]
    result = compute_paired_comparisons(
        records,
        benchmark_slug="slug",
        run_id="run",
        provider_compatibility={
            "eligible": {"compatible": True},
            "incompatible": {"compatible": False},
            "incomplete": {"compatible": True},
        },
        artifact_completeness={
            "eligible": {"complete": True},
            "incompatible": {"complete": True},
            "incomplete": {"complete": False},
        },
    )
    eligibility = {row["provider"]: row for row in result["provider_eligibility"]}
    assert eligibility["eligible"]["eligible"] is True
    assert eligibility["incompatible"]["eligible"] is False
    assert eligibility["incomplete"]["eligible"] is False
    wins = {row["provider"]: row for row in result["fastest_provider_wins"]}
    assert wins["eligible"]["credited_wins"] == 2
    assert wins["eligible"]["eligible_e2e_sample_count"] == 2
    assert wins["incompatible"]["credited_wins"] == 0
    assert wins["incompatible"]["win_rate"] is None
    assert wins["incompatible"]["excluded_or_missing_sample_count"] == 2
    for pair in result["pairwise_e2e_latency_differences"]:
        assert pair["eligible_comparison"] is False
        assert pair["paired_count"] == 0
        assert pair["missing_pair_count"] == 2
        assert pair["coverage_rate"] == 0
        assert pair["exclusion_reasons"]


def test_paired_analysis_derives_model_mismatch_from_raw_records_without_new_arguments():
    eligible = _record("eligible", "meeting-notes-lookup", 1, 100)
    eligible_second = _record("eligible", "ticket-triage", 2, 110)
    mismatched = _record("mismatched", "meeting-notes-lookup", 1, 50)
    mismatched_second = _record("mismatched", "ticket-triage", 2, 55)
    incomplete = _record("incomplete", "meeting-notes-lookup", 1, 40)
    eligible["model_identity"] = {"matches": True}
    eligible_second["model_identity"] = {"matches": True}
    mismatched["model_identity"] = {"matches": False}
    mismatched_second["model_identity"] = {"matches": False}
    incomplete["model_identity"] = {"matches": True}
    result = compute_paired_comparisons(
        [eligible, eligible_second, mismatched, mismatched_second, incomplete],
        benchmark_slug="slug",
        run_id="run",
    )
    wins = {row["provider"]: row for row in result["fastest_provider_wins"]}
    assert wins["eligible"]["win_rate"] == 1
    assert wins["mismatched"]["eligible"] is False
    assert wins["mismatched"]["eligible_e2e_sample_count"] == 0
    assert wins["incomplete"]["eligible"] is False
    assert wins["incomplete"]["artifact_graph_complete"] is False
    pairs = result["pairwise_e2e_latency_differences"]
    assert all(pair["paired_count"] == 0 and pair["coverage_rate"] == 0 for pair in pairs)
    assert any(
        "mismatched: incompatible model or request configuration" in pair["exclusion_reasons"]
        for pair in pairs
    )
    assert any(
        "incomplete: incomplete or internally inconsistent local artifact graph"
        in pair["exclusion_reasons"]
        for pair in pairs
    )
