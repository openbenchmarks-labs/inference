"""Shared no-retry streaming transport with provider-scoped connection pools."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import inspect
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import BenchmarkConfig, load_config
from .providers import (
    PROVIDERS,
    ProviderConfig,
    authentication_headers,
    build_logical_request,
    get_provider,
    redacted_auth_headers,
    resolve_provider_request,
    validate_against_spec,
)


JSON = Any
TIMEOUT_SECONDS = 20
COST_FORMULA = (
    "((provider_reported_input_tokens * input_usd_per_million_tokens) + "
    "(provider_reported_output_tokens * output_usd_per_million_tokens)) / 1000000"
)
_SENSITIVE_KEYS = {
    "authorization",
    "api-key",
    "api_key",
    "apikey",
    "modal-key",
    "modal-secret",
    "proxy_token_id",
    "proxy_token_secret",
    "secret",
    "token",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact(value: JSON, secrets: Sequence[str] = ()) -> JSON:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if str(key).casefold() in _SENSITIVE_KEYS
                else _redact(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secrets:
            if secret:
                result = result.replace(secret, "<redacted>")
        return result
    return value


def _header_items(headers: Any) -> Iterable[tuple[str, str]]:
    if headers is None:
        return ()
    return ((str(name), str(value)) for name, value in headers.items())


def capture_rate_limit_headers(
    headers: Any, *, secrets: Sequence[str] = ()
) -> dict[str, str]:
    captured: dict[str, str] = {}
    for name, value in _header_items(headers):
        lowered = name.casefold()
        if (
            lowered == "retry-after"
            or "ratelimit" in lowered
            or "rate-limit" in lowered
            or "quota" in lowered
        ):
            captured[name] = str(_redact(value, secrets))
    return captured


def _header(headers: Mapping[str, str], wanted: str) -> str | None:
    for name, value in headers.items():
        if name.casefold() == wanted.casefold():
            return value
    return None


def terminal_classification(status_code: int) -> str:
    return "completed" if 200 <= status_code < 300 else "http_error"


def failure_classification(status_code: int | None, outcome: str) -> str | None:
    if outcome == "timeout":
        return "timeout"
    if outcome == "transport_error":
        return "transport_error"
    if status_code == 429:
        return "http_429"
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "other_http_4xx"
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return "http_5xx"
    return None


def _integer(raw: Mapping[str, JSON], *names: str) -> int | None:
    for name in names:
        value = raw.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _usage(raw: JSON) -> dict[str, int | None]:
    value = raw.get("usage") if isinstance(raw, Mapping) else raw
    value = value if isinstance(value, Mapping) else {}
    prompt = _integer(value, "prompt_tokens", "input_tokens")
    completion = _integer(value, "completion_tokens", "output_tokens")
    total = _integer(value, "total_tokens")
    details = value.get("prompt_tokens_details")
    cached = _integer(details, "cached_tokens") if isinstance(details, Mapping) else None
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_prompt_tokens": cached,
    }


def _merge_usage(current: dict[str, int | None], payload: JSON) -> dict[str, int | None]:
    incoming = _usage(payload)
    return {
        key: incoming.get(key) if incoming.get(key) is not None else current.get(key)
        for key in current
    }


def _cost_inputs(
    config: BenchmarkConfig,
    provider: ProviderConfig,
    usage: Mapping[str, int | None],
    response_model: str | None,
) -> dict[str, JSON]:
    price = config.pricing[provider.name]
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    available = isinstance(input_tokens, int) and isinstance(output_tokens, int)
    cost = price.cost(input_tokens, output_tokens) if available else None
    return {
        "provider": provider.name,
        "provider_model": provider.model,
        "response_model": response_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usage_source": "provider_reported",
        "input_price_usd_per_million_tokens": price.input,
        "output_price_usd_per_million_tokens": price.output,
        "pricing_hash": config.pricing_hash,
        "formula": COST_FORMULA,
        "available": available,
        "unavailable_reason": None if available else "provider did not report complete token usage",
        "estimated_cost_usd": cost,
    }


def _text(value: JSON) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for block in value:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            candidate = block.get("text", block.get("content"))
            if isinstance(candidate, str):
                parts.append(candidate)
    return "".join(parts)


def _choice(payload: JSON) -> Mapping[str, JSON]:
    if not isinstance(payload, Mapping):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return {}
    return choices[0]


def _delta_content(payload: JSON) -> tuple[str, str]:
    choice = _choice(payload)
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return "", ""
    reasoning = ""
    for name in ("reasoning_content", "reasoning", "reasoning_details"):
        reasoning = _text(delta.get(name))
        if reasoning:
            break
    return reasoning, _text(delta.get("content"))


def _classification(payload: JSON, reasoning: str, answer: str) -> str:
    if reasoning and answer:
        return "reasoning_and_answer"
    if reasoning:
        return "reasoning"
    if answer:
        return "answer"
    if isinstance(payload, Mapping) and isinstance(payload.get("usage"), Mapping):
        return "usage"
    choice = _choice(payload)
    delta = choice.get("delta")
    if choice.get("finish_reason") is not None:
        return "finish"
    if isinstance(delta, Mapping) and delta.get("role") is not None and len(delta) == 1:
        return "role"
    if isinstance(delta, Mapping):
        return "empty_delta"
    return "metadata"


def _response_model(payload: JSON) -> str | None:
    value = payload.get("model") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) and value else None


def _response_request_id(payload: JSON, headers: Any) -> str | None:
    if isinstance(payload, Mapping):
        for name in ("request_id", "id"):
            value = payload.get(name)
            if isinstance(value, str) and value:
                return value
    for name, value in _header_items(headers):
        if name.casefold() in {"request-id", "x-request-id"}:
            return value
    return None


def _error_body(response: Any, secrets: Sequence[str]) -> JSON:
    try:
        return _redact(response.json(), secrets)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _redact(getattr(response, "text", ""), secrets)


def _line_text(line: Any) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace")
    return str(line)


def _freeze(value: JSON) -> JSON:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: JSON) -> JSON:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class PreparedRequest:
    """Fully validated immutable request, containing no printable credentials."""

    provider: ProviderConfig
    request_id: str
    logical_request_hash: str
    payload: Mapping[str, JSON] = field(repr=False)
    adapter_metadata: Mapping[str, JSON] = field(repr=False)
    identity: Mapping[str, JSON] = field(repr=False)
    authentication: Mapping[str, str] = field(repr=False)
    redacted_headers: Mapping[str, str] = field(repr=False)


class EventSinkError(RuntimeError):
    """Durable event persistence failed; this invalidates the runner."""


class StreamingTransport:
    """One provider-scoped persistent pool and exactly one POST per work unit."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        environment: Mapping[str, str] | None = None,
        *,
        sessions: Mapping[str, Any] | None = None,
        session_factory: Any | None = None,
        monotonic_ns: Any = time.monotonic_ns,
    ) -> None:
        self.config = config or load_config()
        validate_against_spec(self.config)
        self.environment = dict(os.environ if environment is None else environment)
        self._monotonic_ns = monotonic_ns
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=len(PROVIDERS), thread_name_prefix="micro-single-task-stream"
        )
        if sessions is not None and set(sessions) != set(PROVIDERS):
            raise ValueError("sessions must contain exactly one pool for every provider")
        self._owns_sessions = sessions is None
        if sessions is not None:
            self.sessions = dict(sessions)
        else:
            factory = session_factory or requests.Session
            self.sessions = {name: factory() for name in PROVIDERS}
            if len({id(session) for session in self.sessions.values()}) != len(PROVIDERS):
                raise ValueError("session_factory must return one distinct pool per provider")
            for session in self.sessions.values():
                no_retry = Retry(
                    total=0,
                    connect=0,
                    read=0,
                    redirect=0,
                    status=0,
                    other=0,
                    raise_on_redirect=False,
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(
                    max_retries=no_retry,
                    pool_connections=1,
                    pool_maxsize=1,
                    pool_block=True,
                )
                session.mount("https://", adapter)
                session.mount("http://", adapter)
        if len({id(session) for session in self.sessions.values()}) != len(PROVIDERS):
            raise ValueError("transport requires one distinct persistent pool per provider")

    async def __aenter__(self) -> "StreamingTransport":
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_sessions:
            for session in self.sessions.values():
                session.close()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def prepare_request(self, *, unit: Any, item: Any) -> PreparedRequest:
        """Resolve and validate a request without clock access or network I/O."""
        if self._closed:
            raise RuntimeError("transport is closed")
        messages = getattr(item, "messages", None)
        output_schema = getattr(item, "output_schema", None)
        if messages is None and isinstance(item, Mapping):
            messages = item.get("messages")
            output_schema = item.get("output_schema")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError("dataset item is missing messages")
        if not isinstance(output_schema, Mapping):
            raise ValueError("dataset item is missing output_schema")
        provider = get_provider(str(getattr(unit, "provider")))
        logical = build_logical_request(
            request_id=str(getattr(unit, "request_id")),
            messages=messages,
            output_schema=output_schema,
            schema_name=str(getattr(unit, "task_family")),
        )
        resolved = resolve_provider_request(provider, logical)
        # Credential presence and auth-style validation intentionally happen
        # before the runner releases its per-round barrier.
        authentication = authentication_headers(provider, self.environment)
        identity = {
            key: getattr(unit, key)
            for key in (
                "request_id",
                "round_id",
                "round_index",
                "provider",
                "provider_launch_index",
                "task_family",
                "item_id",
                "smoke_attempt",
            )
            if hasattr(unit, key)
        }
        return PreparedRequest(
            provider=provider,
            request_id=logical.request_id,
            logical_request_hash=logical.stable_hash,
            payload=_freeze(dict(resolved.payload)),
            adapter_metadata=_freeze(dict(resolved.adapter_metadata)),
            identity=_freeze(identity),
            authentication=_freeze(authentication),
            redacted_headers=_freeze(redacted_auth_headers(provider)),
        )

    async def execute_prepared(
        self,
        prepared: PreparedRequest,
        event_sink: Callable[[Mapping[str, JSON]], Any] | None = None,
    ) -> dict[str, JSON]:
        """Execute one prepared request, starting timing at transport handoff."""
        if self._closed:
            raise RuntimeError("transport is closed")
        if not isinstance(prepared, PreparedRequest):
            raise TypeError("execute_prepared requires a PreparedRequest")
        if prepared.provider.name not in self.sessions:
            raise ValueError("prepared request provider has no connection pool")
        # Captured immediately before the work is handed to the HTTP executor;
        # connection-pool and executor waiting therefore remain in the sample.
        started_ns = int(self._monotonic_ns())
        started_at = _utc_now()
        future = asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._post_once,
            prepared,
            started_ns,
            started_at,
            event_sink,
        )
        task = asyncio.ensure_future(future)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # A synchronous socket cannot be abandoned before terminal
                # evidence is finalized, even if its asyncio caller is cancelled.
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        return task.result()

    async def execute_request(
        self,
        *,
        unit: Any,
        item: Any,
        event_sink: Callable[[Mapping[str, JSON]], Any] | None = None,
    ) -> dict[str, JSON]:
        return await self.execute_prepared(
            self.prepare_request(unit=unit, item=item), event_sink=event_sink
        )

    async def execute(
        self,
        *,
        unit: Any,
        item: Any,
        event_sink: Callable[[Mapping[str, JSON]], Any] | None = None,
    ) -> dict[str, JSON]:
        return await self.execute_request(unit=unit, item=item, event_sink=event_sink)

    def _base_record(
        self,
        prepared: PreparedRequest,
        started_ns: int,
        started_at: str,
    ) -> dict[str, JSON]:
        return {
            **_thaw(prepared.identity),
            "request_id": prepared.request_id,
            "provider": prepared.provider.name,
            "provider_model": prepared.provider.model,
            "endpoint": prepared.provider.endpoint,
            "logical_request_hash": prepared.logical_request_hash,
            "adapter_metadata": _thaw(prepared.adapter_metadata),
            "attempt": 1,
            "max_attempts": 1,
            "timeout_seconds": TIMEOUT_SECONDS,
            "request_start_monotonic_ns": started_ns,
            "started_monotonic_ns": started_ns,
            "started_at": started_at,
            "request": {
                "headers": {
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    **_thaw(prepared.redacted_headers),
                },
                "body": _thaw(prepared.payload),
            },
            "stream_events": [],
            "reconstructed_reasoning": "",
            "reconstructed_answer": "",
        }

    def _finish(self, record: dict[str, JSON], started_ns: int) -> None:
        ended_ns = int(self._monotonic_ns())
        record["ended_monotonic_ns"] = ended_ns
        record["transport_elapsed_ms"] = max(0.0, (ended_ns - started_ns) / 1_000_000)
        record["completed_at"] = _utc_now()

    def _post_once(
        self,
        prepared: PreparedRequest,
        started_ns: int,
        started_at: str,
        event_sink: Callable[[Mapping[str, JSON]], Any] | None,
    ) -> dict[str, JSON]:
        record = self._base_record(prepared, started_ns, started_at)
        provider = prepared.provider
        secrets = tuple(self.environment.get(name, "") for name in provider.credential_env)
        usage = _usage(None)
        response_model: str | None = None
        provider_request_id: str | None = None
        response: Any | None = None
        rate_headers: dict[str, str] = {}
        retry_after: str | None = None
        deadline_ns = started_ns + TIMEOUT_SECONDS * 1_000_000_000
        try:
            remaining = (deadline_ns - int(self._monotonic_ns())) / 1_000_000_000
            if remaining <= 0:
                raise requests.Timeout("request exceeded the 20 second end-to-end timeout")
            # Exactly one network call. Redirects and urllib3 retries are disabled.
            response = self.sessions[provider.name].post(
                provider.endpoint,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    **_thaw(prepared.authentication),
                },
                json=_thaw(prepared.payload),
                stream=True,
                timeout=min(float(TIMEOUT_SECONDS), remaining),
                allow_redirects=False,
            )
            status = int(response.status_code)
            rate_headers = capture_rate_limit_headers(response.headers, secrets=secrets)
            retry_after = _header(rate_headers, "retry-after")
            if not 200 <= status < 300:
                body = _error_body(response, secrets)
                usage = _usage(body)
                response_model = _response_model(body)
                provider_request_id = _response_request_id(body, response.headers)
                cost_inputs = _cost_inputs(self.config, provider, usage, response_model)
                outcome = terminal_classification(status)
                record.update(
                    terminal_outcome=outcome,
                    failure_classification=failure_classification(status, outcome),
                    http_status=status,
                    response={"body": body},
                    provider_request_id=_redact(provider_request_id, secrets),
                    response_model=response_model,
                    model_identity={
                        "requested": provider.model,
                        "reported": response_model,
                        "matches": response_model == provider.model if response_model else None,
                    },
                    usage=usage,
                    cost_inputs=cost_inputs,
                    cost_usd=cost_inputs["estimated_cost_usd"],
                    retry_after=retry_after,
                    rate_limit_headers=rate_headers,
                    error={"type": "http_error", "message": f"HTTP {status}"},
                    adapter_compatibility={
                        "status": (
                            "unsupported"
                            if provider.reasoning_docs_status == "pending_smoke"
                            and 400 <= status < 500
                            else "request_failed"
                        ),
                        "reason": (
                            "provider rejected the smoke-probed reasoning_effort=low mapping"
                            if provider.reasoning_docs_status == "pending_smoke"
                            and 400 <= status < 500
                            else None
                        ),
                        "exact_low_reasoning_evidenced": False,
                    },
                )
                return record

            if getattr(response, "encoding", None) in {None, "ISO-8859-1"}:
                response.encoding = "utf-8"
            stream_state = self._consume_stream(
                response,
                prepared,
                started_ns,
                deadline_ns,
                secrets,
                record,
                event_sink,
            )
            usage = stream_state["usage"]
            response_model = stream_state["response_model"]
            provider_request_id = stream_state["provider_request_id"] or _response_request_id(
                None, response.headers
            )
            cost_inputs = _cost_inputs(self.config, provider, usage, response_model)
            sent_exact_low = prepared.payload.get("reasoning_effort") == "low"
            exact_model = response_model == provider.model
            exact_low_evidenced = sent_exact_low and exact_model
            if not sent_exact_low:
                compatibility_reason = "prepared payload did not send reasoning_effort=low"
            elif response_model is None:
                compatibility_reason = "successful response omitted model identity"
            elif not exact_model:
                compatibility_reason = (
                    f"response model identity {response_model!r} did not exactly match requested model"
                )
            else:
                compatibility_reason = None
            record.update(
                terminal_outcome="completed",
                failure_classification=None,
                http_status=status,
                response={"stream_captured": True, "done_marker_observed": stream_state["done"]},
                provider_request_id=_redact(provider_request_id, secrets),
                response_model=response_model,
                model_identity={
                    "requested": provider.model,
                    "reported": response_model,
                    "matches": response_model == provider.model if response_model else None,
                },
                stream_events=stream_state["events"],
                reconstructed_reasoning=stream_state["reasoning"],
                reconstructed_answer=stream_state["answer"],
                assistant_content=stream_state["answer"],
                usage=usage,
                cost_inputs=cost_inputs,
                cost_usd=cost_inputs["estimated_cost_usd"],
                retry_after=retry_after,
                rate_limit_headers=rate_headers,
                timings=stream_state["timings"],
                ttfo_ms=stream_state["timings"]["ttfo_ms"],
                ttfa_ms=stream_state["timings"]["ttfa_ms"],
                pre_answer_reasoning_tokens=stream_state["timings"]["pre_answer_reasoning_tokens"],
                reasoning_emitted_before_answer=stream_state["timings"]["reasoning_emitted_before_answer"],
                e2e_latency_ms=stream_state["timings"]["e2e_latency_ms"],
                first_answer_token_ms=stream_state["timings"]["first_answer_token_ms"],
                last_answer_token_ms=stream_state["timings"]["last_answer_token_ms"],
                visible_answer_token_count=stream_state["timings"]["visible_answer_token_count"],
                output_tokens_per_second=stream_state["timings"]["output_tokens_per_second"],
                error=None,
                adapter_compatibility={
                    "status": (
                        "smoke_validated"
                        if provider.reasoning_docs_status == "pending_smoke"
                        and exact_low_evidenced
                        else "documented"
                        if provider.reasoning_docs_status == "documented"
                        and exact_low_evidenced
                        else "incompatible"
                    ),
                    "reason": compatibility_reason,
                    "exact_low_reasoning_evidenced": exact_low_evidenced,
                },
            )
        except requests.Timeout as exc:
            usage = record.get("usage", usage)  # type: ignore[assignment]
            response_model = record.get("response_model", response_model)  # type: ignore[assignment]
            provider_request_id = record.get("provider_request_id", provider_request_id)  # type: ignore[assignment]
            cost_inputs = _cost_inputs(self.config, provider, usage, response_model)
            record.update(
                terminal_outcome="timeout",
                failure_classification="timeout",
                http_status=None,
                response=None,
                provider_request_id=_redact(provider_request_id, secrets),
                response_model=response_model,
                usage=usage,
                cost_inputs=cost_inputs,
                cost_usd=cost_inputs["estimated_cost_usd"],
                retry_after=retry_after,
                rate_limit_headers=rate_headers,
                error={"type": "timeout", "message": _redact(str(exc), secrets)},
            )
        except requests.RequestException as exc:
            usage = record.get("usage", usage)  # type: ignore[assignment]
            response_model = record.get("response_model", response_model)  # type: ignore[assignment]
            provider_request_id = record.get("provider_request_id", provider_request_id)  # type: ignore[assignment]
            cost_inputs = _cost_inputs(self.config, provider, usage, response_model)
            record.update(
                terminal_outcome="transport_error",
                failure_classification="transport_error",
                http_status=None,
                response=None,
                provider_request_id=_redact(provider_request_id, secrets),
                response_model=response_model,
                usage=usage,
                cost_inputs=cost_inputs,
                cost_usd=cost_inputs["estimated_cost_usd"],
                retry_after=retry_after,
                rate_limit_headers=rate_headers,
                error={"type": "transport_error", "message": _redact(str(exc), secrets)},
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            usage = record.get("usage", usage)  # type: ignore[assignment]
            response_model = record.get("response_model", response_model)  # type: ignore[assignment]
            provider_request_id = record.get("provider_request_id", provider_request_id)  # type: ignore[assignment]
            cost_inputs = _cost_inputs(self.config, provider, usage, response_model)
            record.update(
                terminal_outcome="transport_error",
                failure_classification="transport_error",
                http_status=getattr(response, "status_code", None),
                response=None,
                provider_request_id=_redact(provider_request_id, secrets),
                response_model=response_model,
                usage=usage,
                cost_inputs=cost_inputs,
                cost_usd=cost_inputs["estimated_cost_usd"],
                retry_after=retry_after,
                rate_limit_headers=rate_headers,
                error={"type": "invalid_sse", "message": _redact(str(exc), secrets)},
            )
        finally:
            if response is not None:
                response.close()
            self._finish(record, started_ns)
        return record

    def _consume_stream(
        self,
        response: Any,
        prepared: PreparedRequest,
        started_ns: int,
        deadline_ns: int,
        secrets: Sequence[str],
        record: dict[str, JSON],
        event_sink: Callable[[Mapping[str, JSON]], Any] | None,
    ) -> dict[str, JSON]:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        events: list[dict[str, JSON]] = record["stream_events"]
        reasoning_text = ""
        pre_answer_reasoning_text = ""
        answer_text = ""
        ttfo_ms: float | None = None
        ttfa_ms: float | None = None
        last_answer_ms: float | None = None
        response_model: str | None = None
        provider_request_id: str | None = None
        usage = _usage(None)
        done = False
        fields: dict[str, list[str]] = {}
        deadline_hit = threading.Event()
        remaining = max(0.0, (deadline_ns - int(self._monotonic_ns())) / 1_000_000_000)

        def expire() -> None:
            deadline_hit.set()
            response.close()

        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()

        def offset_ms() -> float:
            return max(0.0, (int(self._monotonic_ns()) - started_ns) / 1_000_000)

        def append_event(
            classification: str,
            payload: JSON,
            *,
            reasoning: str = "",
            answer: str = "",
            event_fields: Mapping[str, JSON] | None = None,
        ) -> None:
            content = reasoning if reasoning else answer
            event = {
                    "request_id": prepared.request_id,
                    "event_index": len(events),
                    "monotonic_offset_ms": offset_ms(),
                    "classification": classification,
                    "reasoning_content": reasoning,
                    "answer_content": answer,
                    "content": content,
                    "content_presence": {
                        "reasoning": bool(reasoning),
                        "answer": bool(answer),
                    },
                    "redacted_payload": _redact(payload, secrets),
                    **deepcopy(dict(event_fields or {})),
                }
            events.append(event)
            if event_sink is not None:
                try:
                    result = event_sink(deepcopy(event))
                except Exception as exc:
                    raise EventSinkError(
                        f"stream event sink failed after event {event['event_index']}"
                    ) from exc
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise EventSinkError("stream event sink must be synchronous")

        def dispatch() -> None:
            nonlocal reasoning_text, pre_answer_reasoning_text, answer_text, ttfo_ms, ttfa_ms, last_answer_ms
            nonlocal response_model, provider_request_id, usage, done
            if not fields:
                return
            data = "\n".join(fields.get("data", []))
            meta = {
                "sse_event": fields.get("event", [None])[-1],
                "sse_id": _redact(fields.get("id", [None])[-1], secrets),
                "sse_retry": fields.get("retry", [None])[-1],
            }
            fields.clear()
            if data == "[DONE]":
                done = True
                append_event("done", "[DONE]", event_fields=meta)
                return
            if not data:
                append_event("sse_metadata", meta, event_fields=meta)
                return
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                append_event("malformed_sse", data, event_fields=meta)
                raise ValueError(f"malformed SSE JSON at event {len(events) - 1}") from exc
            if not isinstance(payload, Mapping):
                append_event("malformed_sse", payload, event_fields=meta)
                raise ValueError("SSE data payload must be a JSON object")
            reasoning, answer = _delta_content(payload)
            classification = _classification(payload, reasoning, answer)
            append_event(
                classification,
                payload,
                reasoning=reasoning,
                answer=answer,
                event_fields=meta,
            )
            event_offset = events[-1]["monotonic_offset_ms"]
            if reasoning:
                reasoning_text += reasoning
                if ttfa_ms is None:
                    pre_answer_reasoning_text += reasoning
                if ttfo_ms is None and len(encoding.encode(reasoning_text)) >= 1:
                    ttfo_ms = float(event_offset)
            if answer:
                answer_text += answer
                if ttfa_ms is None and len(encoding.encode(answer_text)) >= 1:
                    ttfa_ms = float(event_offset)
                if ttfo_ms is None and len(encoding.encode(answer_text)) >= 1:
                    ttfo_ms = float(event_offset)
                last_answer_ms = float(event_offset)
            response_model = response_model or _response_model(payload)
            provider_request_id = provider_request_id or _response_request_id(payload, response.headers)
            usage = _merge_usage(usage, payload)
            tokens = len(encoding.encode(answer_text)) if answer_text else 0
            reasoning_tokens = (
                len(encoding.encode(pre_answer_reasoning_text))
                if pre_answer_reasoning_text
                else 0
            )
            partial_speed = None
            if (
                tokens >= 2
                and ttfa_ms is not None
                and last_answer_ms is not None
                and last_answer_ms > ttfa_ms
            ):
                partial_speed = (tokens - 1) / ((last_answer_ms - ttfa_ms) / 1000.0)
            record.update(
                reconstructed_reasoning=reasoning_text,
                reconstructed_answer=answer_text,
                assistant_content=answer_text,
                usage=usage,
                response_model=response_model,
                provider_request_id=_redact(provider_request_id, secrets),
                timings={
                    "clock": "time.monotonic_ns",
                    "unit": "milliseconds",
                    "ttfo_ms": ttfo_ms,
                    "ttfa_ms": ttfa_ms,
                    "pre_answer_reasoning_tokens": reasoning_tokens if ttfa_ms is not None else None,
                    "reasoning_emitted_before_answer": reasoning_tokens > 0 if ttfa_ms is not None else None,
                    "first_answer_token_ms": ttfa_ms,
                    "last_answer_token_ms": last_answer_ms,
                    "e2e_latency_ms": None,
                    "visible_answer_token_count": tokens,
                    "output_tokens_per_second": partial_speed,
                    "tokenizer": "tiktoken:o200k_base",
                },
            )

        try:
            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if deadline_hit.is_set() or int(self._monotonic_ns()) >= deadline_ns:
                    raise requests.Timeout("request exceeded the 20 second end-to-end timeout")
                line = _line_text(raw_line).removesuffix("\r")
                if not line:
                    dispatch()
                    continue
                if line.startswith(":"):
                    append_event("sse_comment", {"comment": line[1:].lstrip()})
                    continue
                name, separator, value = line.partition(":")
                if not separator:
                    name, value = line, ""
                if value.startswith(" "):
                    value = value[1:]
                fields.setdefault(name, []).append(value)
            dispatch()
            if deadline_hit.is_set():
                raise requests.Timeout("request exceeded the 20 second end-to-end timeout")
        finally:
            timer.cancel()

        token_count = len(encoding.encode(answer_text)) if answer_text else 0
        speed = None
        if (
            token_count >= 2
            and ttfa_ms is not None
            and last_answer_ms is not None
            and last_answer_ms > ttfa_ms
        ):
            speed = (token_count - 1) / ((last_answer_ms - ttfa_ms) / 1000.0)
        timings = {
            "clock": "time.monotonic_ns",
            "unit": "milliseconds",
            "ttfo_ms": ttfo_ms,
            "ttfa_ms": ttfa_ms,
            "pre_answer_reasoning_tokens": (
                len(encoding.encode(pre_answer_reasoning_text)) if ttfa_ms is not None else None
            ),
            "reasoning_emitted_before_answer": (
                len(encoding.encode(pre_answer_reasoning_text)) > 0
                if ttfa_ms is not None
                else None
            ),
            "first_answer_token_ms": ttfa_ms,
            "last_answer_token_ms": last_answer_ms,
            "e2e_latency_ms": last_answer_ms,
            "visible_answer_token_count": token_count,
            "output_tokens_per_second": speed,
            "tokenizer": "tiktoken:o200k_base",
        }
        return {
            "events": events,
            "reasoning": reasoning_text,
            "answer": answer_text,
            "usage": usage,
            "response_model": response_model,
            "provider_request_id": provider_request_id,
            "done": done,
            "timings": timings,
        }


def create_transport(
    *,
    config: BenchmarkConfig | None = None,
    environment: Mapping[str, str] | None = None,
    sessions: Mapping[str, Any] | None = None,
) -> StreamingTransport:
    return StreamingTransport(config=config, environment=environment, sessions=sessions)


async def execute_request(
    *, unit: Any, item: Any, transport: StreamingTransport
) -> dict[str, JSON]:
    return await transport.execute_request(unit=unit, item=item)


__all__ = [
    "COST_FORMULA",
    "EventSinkError",
    "TIMEOUT_SECONDS",
    "StreamingTransport",
    "PreparedRequest",
    "capture_rate_limit_headers",
    "create_transport",
    "execute_request",
    "failure_classification",
    "terminal_classification",
]
