from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import time

import pytest
import requests

from openbenchmarks_inference.workflows.micro_single_task.config import load_config
from openbenchmarks_inference.workflows.micro_single_task.providers import (
    LOGICAL_INFERENCE_SETTINGS,
    PROVIDERS,
    authentication_headers,
    build_logical_request,
    provider_compatibility,
    resolve_provider_request,
)
from openbenchmarks_inference.workflows.micro_single_task.transport import (
    StreamingTransport,
    capture_rate_limit_headers,
    failure_classification,
    terminal_classification,
)


MESSAGES = (
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "The owner is Ada."},
)


@dataclass(frozen=True)
class Unit:
    request_id: str = "request-stable-1"
    round_id: str = "round-stable-1"
    round_index: int = 0
    provider: str = "together"
    provider_launch_index: int = 0
    task_family: str = "meeting-notes-lookup"
    item_id: str = "meeting-1"
    smoke_attempt: int = 1


@dataclass(frozen=True)
class Item:
    messages: tuple[dict[str, str], ...] = MESSAGES
    output_schema: dict[str, str] | None = None

    def __post_init__(self):
        if self.output_schema is None:
            object.__setattr__(self, "output_schema", {"answer": "string"})


class FakeResponse:
    def __init__(self, status_code=200, lines=(), headers=None, body=None):
        self.status_code = status_code
        self.lines = list(lines)
        self.headers = headers or {}
        self.body = body
        self.text = "not-json" if body is None else json.dumps(body)
        self.encoding = None
        self.closed = False

    def iter_lines(self, **kwargs):
        assert kwargs == {"chunk_size": 1, "decode_unicode": True}
        for line in self.lines:
            if isinstance(line, Exception):
                raise line
            yield line

    def json(self):
        if self.body is None:
            raise ValueError("not json")
        return self.body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


def sessions_for(provider: str, outcome: object) -> dict[str, FakeSession]:
    return {
        name: FakeSession(outcome) if name == provider else FakeSession()
        for name in PROVIDERS
    }


def sse(payload: object) -> list[str]:
    return ["data: " + (payload if isinstance(payload, str) else json.dumps(payload)), ""]


def test_provider_models_endpoints_and_auth_match_human_spec():
    config = load_config()
    assert tuple(PROVIDERS) == config.providers
    for name, provider in PROVIDERS.items():
        row = config.raw["providers"]["vendors"][name]
        assert provider.model == row["model_key"]
        if name == "modal":
            assert provider.endpoint == row["url"]
            assert authentication_headers(
                provider,
                {
                    "MODAL_PROXY_TOKEN_ID": "id",
                    "MODAL_PROXY_TOKEN_SECRET": "secret",
                },
            ) == {"Modal-Key": "id", "Modal-Secret": "secret"}
        else:
            key = provider.credential_env[0]
            assert authentication_headers(provider, {key: "secret"}) == {
                "Authorization": "Bearer secret"
            }


def test_wire_payloads_preserve_common_streaming_contract():
    logical = build_logical_request(
        request_id="logical-1",
        messages=MESSAGES,
        output_schema={"answer": "string"},
        schema_name="meeting-notes-lookup",
    )
    assert logical.settings == LOGICAL_INFERENCE_SETTINGS
    for name, provider in PROVIDERS.items():
        resolved = resolve_provider_request(provider, logical)
        body = resolved.payload
        assert body["model"] == provider.model
        assert body["messages"] == list(MESSAGES)
        assert body["stream"] is True
        assert body["temperature"] == 0
        assert body["top_p"] == 1
        assert body["presence_penalty"] == body["frequency_penalty"] == 0
        assert body[provider.max_tokens_field] == 256
        assert body["response_format"] == {"type": "json_object"}
        assert body["reasoning_effort"] == "low"
        assert "stop" not in body and "seed" not in body
        if name == "baseten":
            assert body["top_k"] == -1
        else:
            assert "top_k" not in body
        if provider.include_usage:
            assert body["stream_options"] == {"include_usage": True}
        else:
            assert "stream_options" not in body
    assert resolve_provider_request(PROVIDERS["Z.AI"], logical).payload["thinking"] == {
        "type": "enabled"
    }
    novita = resolve_provider_request(PROVIDERS["novita"], logical).payload
    assert novita["reasoning_effort"] == "low"
    assert "enable_thinking" not in novita and "separate_reasoning" not in novita
    compatibility = provider_compatibility()
    assert {
        name
        for name, row in compatibility.items()
        if row["reasoning_docs_status"] == "pending_smoke"
    } == {"baseten", "modal", "parasail", "novita"}
    assert compatibility["Z.AI"]["reasoning_docs_status"] == "documented"


def test_adapter_metadata_is_complete_and_secret_free():
    matrix = provider_compatibility()
    assert set(matrix) == set(PROVIDERS)
    assert all(row["reasoning_effort"] == "low" for row in matrix.values())
    assert all(row["streaming"] is True for row in matrix.values())
    assert all(row["max_attempts"] == 1 for row in matrix.values())
    assert all(row["hidden_warmup_requests"] is False for row in matrix.values())
    assert "API_KEY" not in repr(matrix)


def test_transport_requires_distinct_provider_pools():
    shared = FakeSession()
    with pytest.raises(ValueError, match="distinct persistent pool"):
        StreamingTransport(
            config=load_config(),
            environment={},
            sessions={name: shared for name in PROVIDERS},
        )


def test_prepare_validates_before_release_without_clock_or_network():
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return time.monotonic_ns()

    sessions = sessions_for(
        "together",
        FakeResponse(lines=sse("[DONE]")),
    )
    transport = StreamingTransport(
        config=load_config(),
        environment={"TOGETHER_API_KEY": "credential-value"},
        sessions=sessions,
        monotonic_ns=clock,
    )
    try:
        prepared = transport.prepare_request(unit=Unit(), item=Item())
        assert clock_calls == 0
        assert all(not session.calls for session in sessions.values())
        assert "credential-value" not in repr(prepared)
        with pytest.raises(TypeError):
            prepared.payload["temperature"] = 1
        with pytest.raises(TypeError):
            prepared.payload["response_format"]["type"] = "text"
        result = asyncio.run(transport.execute_prepared(prepared))
        assert clock_calls > 0
        assert result["terminal_outcome"] == "completed"
        assert len(sessions["together"].calls) == 1
    finally:
        transport.close()


def test_prepare_rejects_missing_credentials_and_unknown_provider_before_release():
    sessions = {name: FakeSession() for name in PROVIDERS}
    transport = StreamingTransport(config=load_config(), environment={}, sessions=sessions)
    try:
        with pytest.raises(ValueError, match="missing required credential"):
            transport.prepare_request(unit=Unit(), item=Item())
        with pytest.raises(ValueError, match="unknown provider"):
            transport.prepare_request(
                unit=Unit(provider="not-a-provider"),
                item=Item(),
            )
        assert all(not session.calls for session in sessions.values())
    finally:
        transport.close()


def test_stream_reconstruction_timing_usage_cost_and_redaction():
    secret = "super-secret-value"
    lines = [": keepalive", ""]
    lines += sse(
        {
            "id": "provider-req",
            "model": PROVIDERS["together"].model,
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    lines += sse(
        {
            "id": "provider-req",
            "model": PROVIDERS["together"].model,
            "choices": [
                {"delta": {"reasoning_content": "brief thought"}, "finish_reason": None}
            ],
            "debug": secret,
        }
    )
    lines += sse(
        {
            "id": "provider-req",
            "model": PROVIDERS["together"].model,
            "choices": [{"delta": {"content": "{\"answer\":"}, "finish_reason": None}],
        }
    )
    lines += sse(
        {
            "id": "provider-req",
            "model": PROVIDERS["together"].model,
            "choices": [
                {"delta": {"content": "\"Ada\"}"}, "finish_reason": "stop"}
            ],
        }
    )
    lines += sse(
        {
            "id": "provider-req",
            "model": PROVIDERS["together"].model,
            "choices": [],
            "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
        }
    )
    lines += sse("[DONE]")
    response = FakeResponse(
        lines=lines,
        headers={"Retry-After": "3", "X-RateLimit-Remaining": "9"},
    )
    sessions = sessions_for("together", response)
    transport = StreamingTransport(
        config=load_config(),
        environment={"TOGETHER_API_KEY": secret},
        sessions=sessions,
    )
    try:
        result = asyncio.run(transport.execute_request(unit=Unit(), item=Item()))
    finally:
        transport.close()

    assert len(sessions["together"].calls) == 1
    url, request = sessions["together"].calls[0]
    assert url == PROVIDERS["together"].endpoint
    assert request["stream"] is True and request["allow_redirects"] is False
    assert request["headers"]["Authorization"] == f"Bearer {secret}"
    assert result["request"]["headers"]["Authorization"] == "<redacted>"
    assert secret not in repr(result)
    assert result["terminal_outcome"] == "completed"
    assert result["reconstructed_reasoning"] == "brief thought"
    assert result["reconstructed_answer"] == '{"answer":"Ada"}'
    assert [event["event_index"] for event in result["stream_events"]] == list(
        range(len(result["stream_events"]))
    )
    assert [event["classification"] for event in result["stream_events"]] == [
        "sse_comment",
        "role",
        "reasoning",
        "answer",
        "answer",
        "usage",
        "done",
    ]
    assert "ttfr_ms" not in result and result["ttfa_ms"] is not None
    assert result["ttfo_ms"] is not None
    assert result["pre_answer_reasoning_tokens"] >= 1
    assert result["reasoning_emitted_before_answer"] is True
    assert result["e2e_latency_ms"] == result["last_answer_token_ms"]
    assert result["visible_answer_token_count"] >= 2
    assert result["usage"]["input_tokens"] == 20
    assert result["usage"]["output_tokens"] == 7
    assert result["cost_usd"] == pytest.approx((20 * 0.15 + 7 * 0.5) / 1_000_000)
    assert result["model_identity"]["matches"] is True
    assert result["adapter_compatibility"] == {
        "status": "documented",
        "reason": None,
        "exact_low_reasoning_evidenced": True,
    }
    assert result["retry_after"] == "3"
    assert result["rate_limit_headers"] == {
        "Retry-After": "3",
        "X-RateLimit-Remaining": "9",
    }
    assert result["request_start_monotonic_ns"] <= result["ended_monotonic_ns"]


@pytest.mark.parametrize("provider_name", ["baseten", "modal", "parasail", "novita"])
@pytest.mark.parametrize(
    ("model_kind", "reasoning_effort", "evidenced", "status", "reason_fragment"),
    [
        ("exact", "low", True, "smoke_validated", None),
        ("alias", "low", False, "incompatible", "did not exactly match"),
        ("exact", "high", False, "incompatible", "did not send reasoning_effort=low"),
    ],
)
def test_pending_reasoning_maps_require_exact_model_and_exact_low(
    provider_name, model_kind, reasoning_effort, evidenced, status, reason_fragment
):
    provider = PROVIDERS[provider_name]
    response_model = provider.model if model_kind == "exact" else provider.model + "-alias"
    response = FakeResponse(
        lines=sse({"model": response_model, "choices": []}) + sse("[DONE]")
    )
    sessions = sessions_for(provider_name, response)
    environment = {name: "secret" for name in provider.credential_env}
    transport = StreamingTransport(
        config=load_config(), environment=environment, sessions=sessions
    )
    try:
        prepared = transport.prepare_request(unit=Unit(provider=provider_name), item=Item())
        if reasoning_effort != "low":
            prepared = replace(
                prepared,
                payload={**dict(prepared.payload), "reasoning_effort": reasoning_effort},
            )
        result = asyncio.run(transport.execute_prepared(prepared))
    finally:
        transport.close()

    assert len(sessions[provider_name].calls) == 1
    assert sessions[provider_name].calls[0][1]["json"]["reasoning_effort"] == reasoning_effort
    assert result["adapter_compatibility"]["exact_low_reasoning_evidenced"] is evidenced
    assert result["adapter_compatibility"]["status"] == status
    reason = result["adapter_compatibility"]["reason"]
    if reason_fragment is None:
        assert reason is None
    else:
        assert reason_fragment in reason


def test_event_sink_is_ordered_synchronous_and_receives_redacted_events():
    secret = "sink-secret"
    sink_rows = []

    class OrderingResponse(FakeResponse):
        def iter_lines(self, **kwargs):
            assert kwargs == {"chunk_size": 1, "decode_unicode": True}
            yield 'data: {"choices":[{"delta":{"content":"{"}}],"debug":"sink-secret"}'
            yield ""
            assert len(sink_rows) == 1, "next event parsed before prior event reached sink"
            yield 'data: {"choices":[{"delta":{"content":"}"}}]}'
            yield ""
            assert len(sink_rows) == 2
            yield "data: [DONE]"
            yield ""

    response = OrderingResponse()
    sessions = sessions_for("together", response)
    transport = StreamingTransport(
        config=load_config(),
        environment={"TOGETHER_API_KEY": secret},
        sessions=sessions,
    )
    try:
        prepared = transport.prepare_request(unit=Unit(), item=Item())
        result = asyncio.run(transport.execute_prepared(prepared, event_sink=sink_rows.append))
    finally:
        transport.close()
    assert [row["event_index"] for row in sink_rows] == [0, 1, 2]
    assert sink_rows == result["stream_events"]
    assert secret not in repr(sink_rows)
    assert len(sessions["together"].calls) == 1


@pytest.mark.parametrize(
    ("tail", "outcome", "classifications"),
    [
        (["data: {not-json", ""], "transport_error", ["answer", "malformed_sse"]),
        ([requests.Timeout("late stream timeout")], "timeout", ["answer"]),
    ],
)
def test_sink_preserves_partial_events_before_later_stream_failure(
    tail, outcome, classifications
):
    lines = sse(
        {
            "model": PROVIDERS["together"].model,
            "choices": [{"delta": {"content": "{"}, "finish_reason": None}],
        }
    ) + tail
    sink_rows = []
    sessions = sessions_for("together", FakeResponse(lines=lines))
    transport = StreamingTransport(
        config=load_config(),
        environment={"TOGETHER_API_KEY": "secret"},
        sessions=sessions,
    )
    try:
        result = asyncio.run(
            transport.execute_request(unit=Unit(), item=Item(), event_sink=sink_rows.append)
        )
    finally:
        transport.close()
    assert result["terminal_outcome"] == outcome
    assert [row["classification"] for row in sink_rows] == classifications
    assert sink_rows == result["stream_events"]
    assert result["reconstructed_answer"] == "{"
    assert len(sessions["together"].calls) == 1


def test_http_429_is_terminal_and_never_retried():
    response = FakeResponse(
        status_code=429,
        headers={"Retry-After": "10", "X-RateLimit-Limit": "2"},
        body={"error": {"message": "rate limited"}},
    )
    sessions = sessions_for("together", response)
    transport = StreamingTransport(
        config=load_config(),
        environment={"TOGETHER_API_KEY": "secret"},
        sessions=sessions,
    )
    try:
        result = asyncio.run(transport.execute_request(unit=Unit(), item=Item()))
    finally:
        transport.close()
    assert len(sessions["together"].calls) == 1
    assert result["terminal_outcome"] == "http_error"
    assert result["failure_classification"] == "http_429"
    assert result["retry_after"] == "10"


def test_timeout_and_transport_error_are_redacted_and_not_retried():
    for exception, expected in (
        (requests.Timeout("secret timed out"), "timeout"),
        (requests.ConnectionError("secret disconnected"), "transport_error"),
    ):
        sessions = sessions_for("together", exception)
        transport = StreamingTransport(
            config=load_config(),
            environment={"TOGETHER_API_KEY": "secret"},
            sessions=sessions,
        )
        try:
            result = asyncio.run(transport.execute_request(unit=Unit(), item=Item()))
        finally:
            transport.close()
        assert len(sessions["together"].calls) == 1
        assert result["terminal_outcome"] == expected
        assert "secret" not in repr(result)


def test_terminal_and_rate_limit_helpers():
    assert terminal_classification(200) == "completed"
    assert terminal_classification(429) == "http_error"
    assert failure_classification(429, "http_error") == "http_429"
    assert failure_classification(404, "http_error") == "other_http_4xx"
    assert failure_classification(503, "http_error") == "http_5xx"
    assert capture_rate_limit_headers(
        {"Authorization": "no", "Retry-After": "2", "x-quota-limit": "3"}
    ) == {"Retry-After": "2", "x-quota-limit": "3"}
