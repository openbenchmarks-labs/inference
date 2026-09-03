"""Provider-specific OpenAI chat-completions wire mappings.

Only endpoint, authentication, model spelling, documented token-limit aliases,
and capability syntax differ between arms.  The logical request is otherwise
identical and its hash deliberately excludes the provider request identity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence


JSON = Any


class ProviderConfigurationError(ValueError):
    """A provider request cannot preserve the benchmark contract."""


class MissingCredential(ProviderConfigurationError):
    """A provider credential is absent."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    endpoint: str
    model: str
    credential_env: tuple[str, ...]
    docs_url: str | None
    auth_style: str = "bearer"
    max_tokens_field: str = "max_tokens"
    reasoning_fields: Mapping[str, JSON] | None = None
    top_k_disabled_value: int | None = None
    include_usage: bool = False
    reasoning_docs_status: str = "documented"


@dataclass(frozen=True)
class LogicalRequest:
    request_id: str
    messages: tuple[Mapping[str, str], ...]
    output_schema: Mapping[str, JSON]
    schema_name: str
    settings: Mapping[str, JSON]

    @property
    def stable_hash(self) -> str:
        material = {
            "messages": self.messages,
            "output_schema": self.output_schema,
            "schema_name": self.schema_name,
            "settings": self.settings,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedProviderRequest:
    provider: ProviderConfig
    logical: LogicalRequest
    payload: Mapping[str, JSON]
    adapter_metadata: Mapping[str, JSON]


COMMON_REASONING_COMPARABILITY_NOTE = (
    "every provider is sent its provider-equivalent configuration for reasoning effort low"
)


LOGICAL_INFERENCE_SETTINGS: Mapping[str, JSON] = MappingProxyType(
    {
        "api": "chat-completions",
        "stream": True,
        "stream_protocol": "server_sent_events",
        "reasoning_effort": "low",
        "temperature": 0,
        "top_p": 1,
        "top_k": "disabled",
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_completion_tokens": 256,
        "timeout_seconds": 20,
        "max_attempts": 1,
        "response_format": "json_object",
        "schema_enforcement": "local_strict",
        "turns": 1,
    }
)


def _reasoning(**fields: JSON) -> Mapping[str, JSON]:
    return MappingProxyType(fields)


PROVIDERS: Mapping[str, ProviderConfig] = MappingProxyType(
    {
        "Z.AI": ProviderConfig(
            "Z.AI",
            "https://api.z.ai/api/paas/v4/chat/completions",
            "glm-5.3-flash",
            ("ZAI_API_KEY",),
            "https://docs.z.ai/api-reference/llm/chat-completion",
            reasoning_fields=_reasoning(thinking={"type": "enabled"}, reasoning_effort="low"),
        ),
        "together": ProviderConfig(
            "together",
            "https://api.together.ai/v1/chat/completions",
            "zai-org/GLM-5.3-Flash",
            ("TOGETHER_API_KEY",),
            "https://docs.together.ai/docs/inference/chat/overview",
            reasoning_fields=_reasoning(reasoning_effort="low"),
        ),
        "fireworks": ProviderConfig(
            "fireworks",
            "https://api.fireworks.ai/inference/v1/chat/completions",
            "accounts/fireworks/models/glm-5p3-flash",
            ("FIREWORKS_API_KEY",),
            "https://docs.fireworks.ai/api-reference/post-chatcompletions",
            max_tokens_field="max_completion_tokens",
            reasoning_fields=_reasoning(reasoning_effort="low"),
        ),
        "baseten": ProviderConfig(
            "baseten",
            "https://inference.baseten.co/v1/chat/completions",
            "zai-org/GLM-5.3-Flash",
            ("BASETEN_API_KEY",),
            "https://docs.baseten.co/reference/inference-api/chat-completions",
            reasoning_fields=_reasoning(reasoning_effort="low"),
            top_k_disabled_value=-1,
            include_usage=True,
            reasoning_docs_status="pending_smoke",
        ),
        "telnyx": ProviderConfig(
            "telnyx",
            "https://api.telnyx.com/v2/ai/openai/chat/completions",
            "zai-org/GLM-5.3-Flash",
            ("TELNYX_API_KEY",),
            "https://developers.telnyx.com/api-reference/openai-chat/create-a-chat-completion-openai-compatible",
            reasoning_fields=_reasoning(reasoning_effort="low"),
        ),
        "modal": ProviderConfig(
            "modal",
            "https://openfunnel--ep-glm-5-3-flash-server.us-west.modal.direct/v1/chat/completions",
            "zai-org/GLM-5.3-Flash",
            ("MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"),
            None,
            auth_style="modal_proxy",
            max_tokens_field="max_completion_tokens",
            reasoning_fields=_reasoning(reasoning_effort="low"),
            reasoning_docs_status="pending_smoke",
        ),
        "nebius": ProviderConfig(
            "nebius",
            "https://api.tokenfactory.nebius.com/v1/chat/completions",
            "zai-org/GLM-5.3-Flash",
            ("NEBIUS_API_KEY",),
            "https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion",
            max_tokens_field="max_completion_tokens",
            reasoning_fields=_reasoning(reasoning_effort="low"),
            include_usage=True,
        ),
        "parasail": ProviderConfig(
            "parasail",
            "https://api.parasail.io/v1/chat/completions",
            "zai-org/GLM-5.3-Flash",
            ("PARASAIL_API_KEY",),
            "https://docs.parasail.io/parasail-docs/api-reference/chat-completions",
            max_tokens_field="max_completion_tokens",
            reasoning_fields=_reasoning(reasoning_effort="low"),
            reasoning_docs_status="pending_smoke",
        ),
        "novita": ProviderConfig(
            "novita",
            "https://api.novita.ai/openai/v1/chat/completions",
            "zai-org/glm-5.3-flash",
            ("NOVITA_API_KEY",),
            "https://novita.ai/docs/api-reference/model-apis-llm-create-chat-completion",
            # The current generic reference does not yet list a GLM-5.3-specific
            # low-effort switch.  Preserve the exact OpenAI-compatible logical
            # setting and require smoke acceptance rather than substituting the
            # documented boolean on/off controls (which are not an effort level).
            reasoning_fields=_reasoning(reasoning_effort="low"),
            include_usage=True,
            reasoning_docs_status="pending_smoke",
        ),
        "deepinfra": ProviderConfig(
            "deepinfra",
            "https://api.deepinfra.com/v1/chat/completions",
            "zai-org/GLM-5.3-Flash",
            ("DEEPINFRA_API_KEY",),
            "https://docs.deepinfra.com/api-reference/chat-completions/openai-chat-completions",
            reasoning_fields=_reasoning(reasoning_effort="low"),
            include_usage=True,
        ),
    }
)


def get_provider(name: str) -> ProviderConfig:
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        raise ProviderConfigurationError(f"unknown provider: {name}") from exc


def validate_against_spec(config: Any) -> None:
    """Refuse drift between static wire mappings and the workflow spec."""
    vendors = config.raw["providers"]["vendors"]
    if tuple(vendors) != tuple(PROVIDERS):
        raise ProviderConfigurationError("provider order or membership differs from the spec")
    for name, provider in PROVIDERS.items():
        row = vendors[name]
        if row.get("model_key") != provider.model:
            raise ProviderConfigurationError(f"{name}: model key differs from the spec")
        declared_credentials = tuple(
            row[key]
            for key in ("api_key", "proxy_token_id", "proxy_token_secret")
            if key in row
        )
        if declared_credentials != provider.credential_env:
            raise ProviderConfigurationError(f"{name}: credential mapping differs from the spec")
        if name == "modal" and row.get("url") != provider.endpoint:
            raise ProviderConfigurationError("modal: endpoint differs from the spec")


def build_logical_request(
    *,
    request_id: str,
    messages: Sequence[Mapping[str, str]],
    output_schema: Mapping[str, JSON],
    schema_name: str,
) -> LogicalRequest:
    if not isinstance(request_id, str) or not request_id:
        raise ProviderConfigurationError("request_id must be a non-empty string")
    copied = tuple(deepcopy(dict(message)) for message in messages)
    if len(copied) != 2 or tuple(message.get("role") for message in copied) != ("system", "user"):
        raise ProviderConfigurationError("one-turn requests require system and user messages")
    if any(not isinstance(message.get("content"), str) for message in copied):
        raise ProviderConfigurationError("message content must be text")
    if not isinstance(output_schema, Mapping) or not output_schema:
        raise ProviderConfigurationError("output_schema must be a non-empty mapping")
    if not isinstance(schema_name, str) or not schema_name:
        raise ProviderConfigurationError("schema_name must be a non-empty string")
    return LogicalRequest(
        request_id=request_id,
        messages=copied,
        output_schema=deepcopy(dict(output_schema)),
        schema_name=schema_name,
        settings=deepcopy(dict(LOGICAL_INFERENCE_SETTINGS)),
    )


def adapter_metadata(provider: ProviderConfig) -> dict[str, JSON]:
    reasoning = deepcopy(dict(provider.reasoning_fields or {}))
    return {
        "protocol": "openai-chat-completions",
        "endpoint_class": "serverless_or_shared",
        "model_key": provider.model,
        "structured_output_mode": "json_object",
        "schema_enforcement": "local_strict",
        "reasoning_effort": "low",
        "reasoning_wire_mapping": reasoning,
        "reasoning_docs_status": provider.reasoning_docs_status,
        "top_k_wire_mapping": (
            {"top_k": provider.top_k_disabled_value}
            if provider.top_k_disabled_value is not None
            else {"top_k": "omitted"}
        ),
        "streaming": True,
        "stream_protocol": "server_sent_events",
        "max_attempts": 1,
        "hidden_warmup_requests": False,
        "comparability_note": COMMON_REASONING_COMPARABILITY_NOTE,
    }


def resolve_provider_request(
    provider: ProviderConfig, logical: LogicalRequest
) -> ResolvedProviderRequest:
    reasoning = deepcopy(dict(provider.reasoning_fields or {}))
    if reasoning.get("reasoning_effort") != "low":
        raise ProviderConfigurationError(f"{provider.name}: low reasoning is not mapped")
    payload: dict[str, JSON] = {
        "model": provider.model,
        "messages": deepcopy(list(logical.messages)),
        "stream": True,
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        provider.max_tokens_field: 256,
        "response_format": {"type": "json_object"},
        **reasoning,
    }
    if provider.top_k_disabled_value is not None:
        payload["top_k"] = provider.top_k_disabled_value
    if provider.include_usage:
        payload["stream_options"] = {"include_usage": True}
    return ResolvedProviderRequest(
        provider=provider,
        logical=logical,
        payload=payload,
        adapter_metadata=adapter_metadata(provider),
    )


def authentication_headers(
    provider: ProviderConfig, environment: Mapping[str, str]
) -> dict[str, str]:
    missing = [name for name in provider.credential_env if not environment.get(name)]
    if missing:
        raise MissingCredential(
            f"{provider.name}: missing required credential variable(s): {', '.join(missing)}"
        )
    if provider.auth_style == "bearer":
        return {"Authorization": f"Bearer {environment[provider.credential_env[0]]}"}
    if provider.auth_style == "modal_proxy":
        return {
            "Modal-Key": environment["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": environment["MODAL_PROXY_TOKEN_SECRET"],
        }
    raise ProviderConfigurationError(f"{provider.name}: unknown auth style")


def redacted_auth_headers(provider: ProviderConfig) -> dict[str, str]:
    if provider.auth_style == "modal_proxy":
        return {"Modal-Key": "<redacted>", "Modal-Secret": "<redacted>"}
    return {"Authorization": "<redacted>"}


def provider_compatibility() -> dict[str, dict[str, JSON]]:
    return {
        name: {
            **adapter_metadata(provider),
            "eligible": True,
            "docs_url": provider.docs_url,
        }
        for name, provider in PROVIDERS.items()
    }


__all__ = [
    "COMMON_REASONING_COMPARABILITY_NOTE",
    "LOGICAL_INFERENCE_SETTINGS",
    "PROVIDERS",
    "LogicalRequest",
    "MissingCredential",
    "ProviderConfig",
    "ProviderConfigurationError",
    "ResolvedProviderRequest",
    "adapter_metadata",
    "authentication_headers",
    "build_logical_request",
    "get_provider",
    "provider_compatibility",
    "redacted_auth_headers",
    "resolve_provider_request",
    "validate_against_spec",
]
