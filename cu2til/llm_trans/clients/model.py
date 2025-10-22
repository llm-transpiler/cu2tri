from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import AsyncOpenAI, OpenAI

from llm.client.openai_compat import (
    CallingIdentifier,
    async_openai_llm_call,
    get_api_param_openai_default,
    get_api_params_method,
    make_openai_message_assistant,
    make_openai_message_system,
    make_openai_message_user,
    openai_llm_call,
)

from ..config.model_registry import ModelRegistryError, ProviderConfig
from ..config.settings import Settings


@dataclass
class EndpointEntry:
    """Concrete, ready-to-use client endpoint."""

    name: str
    provider: str
    base_url: str | None
    api_key: str | None
    calling_id: CallingIdentifier | None
    model_name: str
    client: OpenAI
    async_client: AsyncOpenAI

    def param_builder(self) -> Callable:
        return (
            get_api_params_method(self.calling_id)
            if self.calling_id is not None
            else get_api_param_openai_default
        )


@dataclass
class ModelClients:
    model_name: str
    client: OpenAI | None
    async_client: AsyncOpenAI | None
    get_api_param: Callable
    base_url: str | None
    endpoints: list[EndpointEntry] = field(default_factory=list)
    _current_idx: int = 0

    def has_pool(self) -> bool:
        return bool(self.endpoints)

    def current_endpoint(self) -> EndpointEntry | None:
        if not self.endpoints:
            return None
        return self.endpoints[self._current_idx % len(self.endpoints)]

    def rotate_next(self) -> EndpointEntry | None:
        if not self.endpoints:
            return None
        self._current_idx = (self._current_idx + 1) % len(self.endpoints)
        ep = self.current_endpoint()
        if ep is not None:
            self.client = ep.client
            self.async_client = ep.async_client
            self.base_url = ep.base_url
        return ep


def _resolve_calling(calling_name: str | None, provider_name: str, vendor_name: str) -> CallingIdentifier | None:
    if not calling_name:
        return None
    try:
        return CallingIdentifier[calling_name]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise ModelRegistryError(
            f"Provider '{provider_name}' for vendor '{vendor_name}' references unknown calling identifier '{calling_name}'"
        ) from exc


def _resolve_base_url(provider: ProviderConfig) -> str | None:
    return provider.base_url


def _resolve_api_key(provider: ProviderConfig, vendor_name: str) -> str:
    api_key = provider.api_key
    if api_key is None:
        raise ModelRegistryError(
            f"Provider '{provider.name}' for vendor '{vendor_name}' requires an API key"
        )
    return api_key


def _instantiate_endpoint(
    *,
    provider_name: str,
    provider: ProviderConfig,
    vendor_name: str,
    model_name: str,
) -> EndpointEntry:
    base_url = _resolve_base_url(provider)
    api_key = _resolve_api_key(provider, vendor_name)
    calling_id = _resolve_calling(provider.calling, provider_name, vendor_name)

    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key)
        async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    else:
        client = OpenAI(api_key=api_key)
        async_client = AsyncOpenAI(api_key=api_key)

    return EndpointEntry(
        name=f"{vendor_name}:{provider_name}",
        provider=provider_name,
        base_url=base_url,
        api_key=api_key,
        calling_id=calling_id,
        model_name=model_name,
        client=client,
        async_client=async_client,
    )


def create_model_clients(settings: Settings) -> ModelClients:
    args = settings.args
    registry = settings.model_registry

    selection = registry.resolve(args.model)
    forced_provider: str | None = getattr(args, "model_provider", None)
    provider_sequence = selection.provider_sequence(forced_provider=forced_provider)

    endpoints: list[EndpointEntry] = []
    for provider_name in provider_sequence:
        provider_cfg = selection.vendor.providers[provider_name]
        endpoints.append(
            _instantiate_endpoint(
                provider_name=provider_name,
                provider=provider_cfg,
                vendor_name=selection.vendor_name,
                model_name=selection.model_name,
            )
        )

    if not endpoints:
        raise ModelRegistryError(
            f"No providers available for model '{selection.model_name}' (vendor '{selection.vendor_name}')"
        )

    primary = endpoints[0]

    def _get_api_param(messages: list[dict[str, Any]], model_name: str | None = None, **kwargs):
        container = getattr(_get_api_param, "_container", None)
        endpoint = primary
        if container is not None and isinstance(container, ModelClients):
            current = container.current_endpoint()
            if current is not None:
                endpoint = current

        effective_model = model_name or endpoint.model_name
        builder = endpoint.param_builder()
        return builder(messages, model_name=effective_model, **kwargs)

    model_clients = ModelClients(
        model_name=selection.model_name,
        client=primary.client,
        async_client=primary.async_client,
        get_api_param=_get_api_param,
        base_url=primary.base_url,
        endpoints=endpoints,
    )

    setattr(_get_api_param, "_container", model_clients)

    return model_clients


__all__ = [
    "ModelClients",
    "create_model_clients",
    "async_openai_llm_call",
    "openai_llm_call",
    "make_openai_message_system",
    "make_openai_message_user",
    "make_openai_message_assistant",
]
