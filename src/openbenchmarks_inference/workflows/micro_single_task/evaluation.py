"""Strict deterministic evaluation for the single-user streaming benchmark.

The functions in this module are deliberately free of filesystem, provider,
and database dependencies.  A persisted terminal record and its frozen dataset
item are sufficient to reproduce an evaluation.
"""

from __future__ import annotations

from datetime import date
import json
from typing import Any, Mapping, Sequence


EVALUATOR_VERSION = "micro-single-task-evaluator-v1"
MEETING_TASK = "meeting-notes-lookup"
TICKET_TASK = "ticket-triage"
CONTRACT_TASK = "contract-terms-extraction"
TASK_FAMILIES = (MEETING_TASK, TICKET_TASK, CONTRACT_TASK)

TICKET_CATEGORIES = frozenset({"bug", "feature", "improvement", "customer_support"})
TICKET_PRIORITIES = frozenset({"p0", "p1", "p2", "p3"})
AGREEMENT_TYPES = frozenset(
    {"saas_subscription", "software_license", "support_maintenance", "technology_services"}
)
CONTRACT_FIELDS = (
    "agreement_type",
    "effective_date",
    "initial_term_end_date",
    "auto_renews",
    "renewal_term_months",
    "non_renewal_notice_days",
    "payment_terms_days",
    "termination_for_convenience",
    "termination_notice_days",
    "security_incident_notice_hours",
)


class EvaluationContractError(RuntimeError):
    """Local evidence is contradictory or unusable and invalidates the run."""


class MalformedJSONError(ValueError):
    """Visible answer content is not exactly one standards-compliant JSON value."""


class _DuplicateKey(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise _DuplicateKey(f"duplicate object key {key!r}")
        output[key] = value
    return output


_DECODER = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
_JSON_WHITESPACE = " \t\r\n"


def parse_exact_json(text: str) -> Any:
    """Parse one JSON value without markdown stripping, coercion, or repair."""

    if not isinstance(text, str):
        raise MalformedJSONError("assistant content must be a string")
    start = len(text) - len(text.lstrip(_JSON_WHITESPACE))
    if start == len(text):
        raise MalformedJSONError("assistant content is empty")
    try:
        value, end = _DECODER.raw_decode(text, start)
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise MalformedJSONError(str(exc)) from exc
    if text[end:].strip(_JSON_WHITESPACE):
        raise MalformedJSONError("assistant content contains more than one JSON value")
    return value


def normalize_meeting_answer(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _base(task_family: str, request_id: str | None) -> dict[str, Any]:
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "request_id": request_id,
        "task_family": task_family,
        "classification": None,
        "error_type": None,
        "parsed_output": None,
        "schema_valid": False,
        "correct": False,
        "correct_field_count": None,
        "field_count_total": None,
        "field_correct": None,
    }


def _invalid(
    task_family: str,
    request_id: str | None,
    error_type: str,
    parsed_output: Any = None,
) -> dict[str, Any]:
    result = _base(task_family, request_id)
    result.update(
        classification="invalid_result", error_type=error_type, parsed_output=parsed_output
    )
    return result


def _json_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _iso_date(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 10:
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _contract_value_valid(field: str, value: Any) -> bool:
    if field == "agreement_type":
        return isinstance(value, str) and value in AGREEMENT_TYPES
    if field in {"effective_date", "initial_term_end_date"}:
        return _iso_date(value)
    if field in {"auto_renews", "termination_for_convenience"}:
        return isinstance(value, bool)
    return value is None or _json_integer(value)


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "invalid"


def _json_equal(actual: Any, expected: Any) -> bool:
    return _json_kind(actual) == _json_kind(expected) and actual == expected


def _meeting_gold(gold: Any) -> str:
    if isinstance(gold, str):
        return gold
    if isinstance(gold, Mapping):
        for key in ("value", "answer"):
            if isinstance(gold.get(key), str):
                return gold[key]
    raise EvaluationContractError("meeting gold must contain a string value")


def _validate_gold(task_family: str, gold: Any) -> None:
    if task_family == MEETING_TASK:
        _meeting_gold(gold)
        return
    if task_family == TICKET_TASK:
        if (
            not isinstance(gold, Mapping)
            or set(gold) != {"category", "priority"}
            or gold.get("category") not in TICKET_CATEGORIES
            or gold.get("priority") not in TICKET_PRIORITIES
        ):
            raise EvaluationContractError("ticket gold must be an exact valid enum pair")
        return
    if task_family != CONTRACT_TASK:
        raise EvaluationContractError(f"unknown task family {task_family!r}")
    if (
        not isinstance(gold, Mapping)
        or set(gold) != set(CONTRACT_FIELDS)
        or any(not _contract_value_valid(field, gold[field]) for field in CONTRACT_FIELDS)
    ):
        raise EvaluationContractError("contract gold must exactly match the declared schema")


def evaluate(
    task_family: str,
    assistant_content: str,
    gold: Any,
    aliases: Sequence[str] | Mapping[str, Any] | None = (),
    request_id: str | None = None,
) -> dict[str, Any]:
    """Strictly evaluate one HTTP-successful reconstructed visible answer."""

    _validate_gold(task_family, gold)
    try:
        parsed = parse_exact_json(assistant_content)
    except MalformedJSONError:
        return _invalid(task_family, request_id, "malformed_json")

    if task_family == MEETING_TASK:
        if not isinstance(parsed, dict):
            return _invalid(task_family, request_id, "schema_invalid", parsed)
        if "answer" not in parsed:
            return _invalid(task_family, request_id, "missing_required_field", parsed)
        if set(parsed) != {"answer"} or not isinstance(parsed["answer"], str):
            return _invalid(task_family, request_id, "schema_invalid", parsed)
        candidates = [_meeting_gold(gold)]
        if isinstance(aliases, Mapping):
            aliases = aliases.get("answer", ())
        if isinstance(aliases, str):
            candidates.append(aliases)
        elif isinstance(aliases, Sequence):
            candidates.extend(value for value in aliases if isinstance(value, str))
        correct = normalize_meeting_answer(parsed["answer"]) in {
            normalize_meeting_answer(value) for value in candidates
        }
        result = _base(task_family, request_id)
        result.update(
            classification="valid_result" if correct else "incorrect_result",
            parsed_output=parsed,
            schema_valid=True,
            correct=correct,
        )
        return result

    if task_family == TICKET_TASK:
        if not isinstance(parsed, dict):
            return _invalid(task_family, request_id, "schema_invalid", parsed)
        missing = {"category", "priority"} - set(parsed)
        if missing:
            return _invalid(task_family, request_id, "missing_required_field", parsed)
        if (
            set(parsed) != {"category", "priority"}
            or parsed["category"] not in TICKET_CATEGORIES
            or parsed["priority"] not in TICKET_PRIORITIES
        ):
            return _invalid(task_family, request_id, "schema_invalid", parsed)
        correct = parsed["category"] == gold["category"] and parsed["priority"] == gold["priority"]
        result = _base(task_family, request_id)
        result.update(
            classification="valid_result" if correct else "incorrect_result",
            parsed_output=parsed,
            schema_valid=True,
            correct=correct,
        )
        return result

    if not isinstance(parsed, dict):
        return _invalid(task_family, request_id, "schema_invalid", parsed)
    field_correct = {
        field: field in parsed and _json_equal(parsed[field], gold[field])
        for field in CONTRACT_FIELDS
    }
    missing = [field for field in CONTRACT_FIELDS if field not in parsed]
    invalid = [
        field
        for field in CONTRACT_FIELDS
        if field in parsed and not _contract_value_valid(field, parsed[field])
    ]
    schema_valid = not missing and not invalid and set(parsed) == set(CONTRACT_FIELDS)
    correct_count = sum(field_correct.values())
    correct = schema_valid and correct_count == len(CONTRACT_FIELDS)
    result = _base(task_family, request_id)
    result.update(
        classification=(
            "valid_result" if correct else "incorrect_result" if schema_valid else "invalid_result"
        ),
        error_type=(None if schema_valid else "missing_required_field" if missing else "schema_invalid"),
        parsed_output=parsed,
        schema_valid=schema_valid,
        correct=correct,
        correct_field_count=correct_count,
        field_count_total=len(CONTRACT_FIELDS),
        field_correct=field_correct,
    )
    return result


def is_known_transport_runner_error(record: Mapping[str, Any]) -> bool:
    """Identify the historical requests/urllib3 stream failure signature."""

    status = record.get("http_status", record.get("status_code"))
    terminal = record.get("terminal_outcome", record.get("terminal_status"))
    error = record.get("error")
    return (
        terminal == "runner_error"
        and isinstance(error, Mapping)
        and error.get("type") == "AttributeError"
        and error.get("message") == "'NoneType' object has no attribute 'read'"
        and status is None
    )


def terminal_outcome(record: Mapping[str, Any]) -> str:
    """Return one mutually-exclusive operational category from saved evidence."""

    status = record.get("http_status", record.get("status_code"))
    if isinstance(status, bool) or (status is not None and not isinstance(status, int)):
        raise EvaluationContractError("http status must be an integer or null")
    terminal = record.get("terminal_outcome", record.get("terminal_status"))
    # requests/urllib3 can surface this response-stream failure as a bare
    # AttributeError outside RequestException. Historical records captured it
    # at the runner boundary. It is still an observed transport failure, so
    # classify this exact signature without mutating the immutable raw record.
    if is_known_transport_runner_error(record):
        return "transport_error"
    if terminal in {"runner_error", "evaluator_error"}:
        raise EvaluationContractError(f"run-invalidating terminal outcome {terminal!r}")
    if terminal in {"timeout", "timed_out"}:
        return "timeout"
    if terminal in {"transport_error", "transport-error"}:
        return "transport_error"
    if status is not None:
        if 200 <= status <= 299:
            if terminal == "http_error":
                raise EvaluationContractError("http_error record has a 2xx status")
            return "2xx"
        if status == 429:
            return "429"
        if 400 <= status <= 499:
            return "other_4xx"
        if 500 <= status <= 599:
            return "5xx"
        raise EvaluationContractError(f"unclassifiable HTTP status {status}")
    explicit = {
        "http_2xx": "2xx",
        "http_429": "429",
        "http_other_4xx": "other_4xx",
        "http_5xx": "5xx",
    }
    if terminal in explicit:
        return explicit[terminal]
    if terminal == "completed":
        return "2xx"
    if terminal == "http_error":
        raise EvaluationContractError("http_error record requires an HTTP status")
    raise EvaluationContractError("terminal record has no classifiable outcome")


def _assistant_content(record: Mapping[str, Any]) -> Any:
    for key in ("reconstructed_answer", "assistant_content", "response_content", "answer"):
        if key in record:
            return record[key]
    response = record.get("response")
    if isinstance(response, Mapping) and isinstance(response.get("body"), Mapping):
        response = response["body"]
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def evaluate_record(item: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one terminal raw record against a frozen dataset item."""

    task_family = item.get("task_family", item.get("use_case"))
    if task_family not in TASK_FAMILIES:
        raise EvaluationContractError("dataset item has an unknown or missing task family")
    record_family = record.get("task_family")
    if record_family != task_family:
        raise EvaluationContractError("terminal record task_family does not match dataset item")
    provider = record.get("provider")
    if not isinstance(provider, str) or not provider:
        raise EvaluationContractError("terminal record requires a non-empty provider")
    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise EvaluationContractError("terminal record requires a non-empty request_id")

    identity = {
        key: record.get(key)
        for key in ("provider", "task_family", "item_id", "round_id", "round_index")
        if record.get(key) is not None
    }
    outcome = terminal_outcome(record)
    if outcome != "2xx":
        error_type = {
            "timeout": "timeout",
            "429": "rate_limited_http_429",
            "other_4xx": "other_http_4xx",
            "5xx": "http_5xx",
            "transport_error": "transport_error",
        }[outcome]
        result = _base(task_family, request_id)
        result.update(classification="failed_request", error_type=error_type, **identity)
        return result

    result = evaluate(
        task_family,
        _assistant_content(record),
        item.get("gold"),
        item.get("aliases") or (),
        request_id,
    )
    result.update(identity)
    return result


def compatibility_evidence(
    record: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, bool]:
    content = _assistant_content(record)
    evidence = {
        "terminal_record": True,
        "http_2xx": terminal_outcome(record) == "2xx",
        "nonempty_answer_content": isinstance(content, str) and bool(content.strip()),
        "schema_valid": evaluation.get("schema_valid") is True,
    }
    evidence["compatible"] = all(evidence.values())
    return evidence


__all__ = [
    "AGREEMENT_TYPES",
    "CONTRACT_FIELDS",
    "CONTRACT_TASK",
    "EVALUATOR_VERSION",
    "EvaluationContractError",
    "MEETING_TASK",
    "MalformedJSONError",
    "TASK_FAMILIES",
    "TICKET_CATEGORIES",
    "TICKET_PRIORITIES",
    "TICKET_TASK",
    "compatibility_evidence",
    "evaluate",
    "evaluate_record",
    "is_known_transport_runner_error",
    "normalize_meeting_answer",
    "parse_exact_json",
    "terminal_outcome",
]
