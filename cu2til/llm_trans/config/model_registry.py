from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ModelRegistryError(RuntimeError):
    """Raised when the model registry YAML is invalid."""


@dataclass
class ProviderConfig:
    name: str
    base_url: str | None = None
    api_key: str | None = None
    calling: str | None = None


@dataclass
class ModelConfig:
    name: str
    providers: list[str]
    default_provider: str
    display_name: str | None = None


@dataclass
class VendorConfig:
    name: str
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)


@dataclass
class AliasConfig:
    key: str
    vendor: str
    model: str
    provider: str | None = None


@dataclass
class ResolvedModel:
    vendor_name: str
    model_name: str
    vendor: VendorConfig
    model: ModelConfig
    alias_provider: str | None = None

    def provider_sequence(self, forced_provider: str | None = None) -> list[str]:
        """Compute provider order honoring forced overrides and defaults."""

        def _normalize(name: str | None) -> str | None:
            if name is None:
                return None
            return name.strip()

        forced = _normalize(forced_provider)
        alias = _normalize(self.alias_provider)
        preferred = _normalize(self.model.default_provider)

        seen: set[str] = set()
        order: list[str] = []

        def _push(candidate: str | None) -> None:
            if candidate is None:
                return
            if candidate not in self.model.providers:
                raise ModelRegistryError(
                    f"Provider '{candidate}' is not defined for model '{self.model_name}'"
                )
            if candidate not in seen:
                order.append(candidate)
                seen.add(candidate)

        _push(forced)
        _push(alias)
        _push(preferred)
        for provider_name in self.model.providers:
            _push(provider_name)

        return order


class ModelRegistry:
    def __init__(self, vendors: dict[str, VendorConfig], aliases: dict[str, AliasConfig]):
        self._vendors = vendors
        self._aliases = aliases

    @property
    def vendors(self) -> dict[str, VendorConfig]:
        return self._vendors

    @property
    def aliases(self) -> dict[str, AliasConfig]:
        return self._aliases

    def resolve(self, key: str) -> ResolvedModel:
        """Resolve CLI key or explicit model name to a concrete model."""

        if not key:
            raise ModelRegistryError("Model key must be provided")

        alias = self._aliases.get(key)
        if alias is not None:
            vendor_name = alias.vendor
            model_name = alias.model
            alias_provider = alias.provider
        else:
            vendor_name, model_name, alias_provider = self._find_model_by_name(key)

        vendor = self._vendors.get(vendor_name)
        if vendor is None:
            raise ModelRegistryError(f"Vendor '{vendor_name}' not found in registry")

        model = vendor.models.get(model_name)
        if model is None:
            raise ModelRegistryError(
                f"Model '{model_name}' not registered under vendor '{vendor_name}'"
            )

        return ResolvedModel(
            vendor_name=vendor_name,
            model_name=model_name,
            vendor=vendor,
            model=model,
            alias_provider=alias_provider,
        )

    def _find_model_by_name(self, model_key: str) -> tuple[str, str, str | None]:
        """Find model by fully qualified name across vendors."""

        # Allow keys in the form vendor:model (legacy style)
        if ":" in model_key:
            vendor_name, model_name = model_key.split(":", 1)
            vendor_name = vendor_name.strip()
            model_name = model_name.strip()
            if vendor_name in self._vendors and model_name in self._vendors[vendor_name].models:
                return vendor_name, model_name, None

        # Allow direct model name lookup (first match wins)
        matches: list[tuple[str, str]] = []
        for vendor_name, vendor in self._vendors.items():
            if model_key in vendor.models:
                matches.append((vendor_name, model_key))

        if len(matches) == 1:
            return matches[0][0], matches[0][1], None
        if len(matches) > 1:
            vendors = ", ".join(v for v, _ in matches)
            raise ModelRegistryError(
                f"Model '{model_key}' is ambiguous; specify vendor explicitly ({vendors})"
            )

        raise ModelRegistryError(
            f"Model '{model_key}' not found in registry aliases or vendor definitions"
        )


def _load_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
    except FileNotFoundError as exc:  # pragma: no cover - configuration error
        raise ModelRegistryError(f"Model registry file not found: {path}") from exc
    except yaml.YAMLError as exc:  # pragma: no cover - configuration error
        raise ModelRegistryError(f"Failed to parse model registry YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelRegistryError("Model registry YAML must define a mapping at top level")
    return data


def _resolve_scalar(value, *, field: str, context: str, required: bool) -> str | None:
    if value is None:
        if required:
            raise ModelRegistryError(f"{context} missing required field '{field}'")
        return None

    if isinstance(value, dict):
        if "env" not in value:
            raise ModelRegistryError(
                f"{context} field '{field}' mapping must contain 'env' key"
            )
        env_name = str(value["env"]).strip()
        default = value.get("default")
        resolved = os.getenv(env_name, default)
        if resolved is None and required:
            raise ModelRegistryError(
                f"Environment variable '{env_name}' for {context} field '{field}' is not set"
            )
        return resolved

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("$") and len(stripped) > 1:
            env_name = stripped[1:]
            resolved = os.getenv(env_name)
            if resolved is None:
                if required:
                    raise ModelRegistryError(
                        f"Environment variable '{env_name}' for {context} field '{field}' is not set"
                    )
                return None
            return resolved
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    if required and value is None:
        raise ModelRegistryError(f"{context} missing required field '{field}'")
    if value is None:
        return None
    raise ModelRegistryError(
        f"{context} field '{field}' must be a string or env mapping, got {type(value).__name__}"
    )


def load_model_registry(path: Path) -> ModelRegistry:
    raw = _load_yaml(path)

    vendors_raw = raw.get("vendors", {})
    if not isinstance(vendors_raw, dict):
        raise ModelRegistryError("'vendors' must be a mapping of vendor definitions")

    vendors: dict[str, VendorConfig] = {}
    for vendor_name, vendor_payload in vendors_raw.items():
        if not isinstance(vendor_payload, dict):
            raise ModelRegistryError(f"Vendor '{vendor_name}' definition must be a mapping")

        providers_raw = vendor_payload.get("providers", {})
        if not isinstance(providers_raw, dict):
            raise ModelRegistryError(f"Vendor '{vendor_name}' providers must be a mapping")

        providers: dict[str, ProviderConfig] = {}
        for provider_name, provider_payload in providers_raw.items():
            if not isinstance(provider_payload, dict):
                raise ModelRegistryError(
                    f"Provider '{provider_name}' under vendor '{vendor_name}' must be a mapping"
                )
            base_url = _resolve_scalar(
                provider_payload.get("base_url"),
                field="base_url",
                context=f"Provider '{provider_name}' for vendor '{vendor_name}'",
                required=False,
            )
            api_key = _resolve_scalar(
                provider_payload.get("api_key"),
                field="api_key",
                context=f"Provider '{provider_name}' for vendor '{vendor_name}'",
                required=False,
            )
            calling = provider_payload.get("calling")
            if calling is not None and not isinstance(calling, str):
                raise ModelRegistryError(
                    f"Provider '{provider_name}' under vendor '{vendor_name}' must use string 'calling'"
                )
            providers[provider_name] = ProviderConfig(
                name=provider_name,
                base_url=base_url,
                api_key=api_key,
                calling=calling,
            )

        models_raw = vendor_payload.get("models", {})
        if not isinstance(models_raw, dict):
            raise ModelRegistryError(f"Vendor '{vendor_name}' models must be a mapping")

        models: dict[str, ModelConfig] = {}
        for model_name, model_payload in models_raw.items():
            if not isinstance(model_payload, dict):
                raise ModelRegistryError(
                    f"Model '{model_name}' under vendor '{vendor_name}' must be a mapping"
                )

            providers_list = model_payload.get("providers")
            if not isinstance(providers_list, Iterable) or isinstance(providers_list, (str, bytes)):
                raise ModelRegistryError(
                    f"Model '{model_name}' under vendor '{vendor_name}' must list providers"
                )
            providers_seq = [str(name) for name in providers_list]
            if not providers_seq:
                raise ModelRegistryError(
                    f"Model '{model_name}' under vendor '{vendor_name}' must declare at least one provider"
                )

            for provider_ref in providers_seq:
                if provider_ref not in providers:
                    raise ModelRegistryError(
                        f"Model '{model_name}' references unknown provider '{provider_ref}' for vendor '{vendor_name}'"
                    )

            default_provider = model_payload.get("default_provider") or providers_seq[0]
            if default_provider not in providers_seq:
                raise ModelRegistryError(
                    f"Model '{model_name}' default provider '{default_provider}' is not in its provider list"
                )

            models[model_name] = ModelConfig(
                name=model_name,
                providers=providers_seq,
                default_provider=default_provider,
                display_name=model_payload.get("display_name"),
            )

        vendors[vendor_name] = VendorConfig(name=vendor_name, providers=providers, models=models)

    aliases_raw = raw.get("aliases", {})
    if not isinstance(aliases_raw, dict):
        raise ModelRegistryError("'aliases' must be a mapping if provided")

    aliases: dict[str, AliasConfig] = {}
    for alias_key, alias_payload in aliases_raw.items():
        if not isinstance(alias_payload, dict):
            raise ModelRegistryError(f"Alias '{alias_key}' must be defined as a mapping")
        vendor_name = alias_payload.get("vendor")
        model_name = alias_payload.get("model")
        provider_name = alias_payload.get("provider")

        if not vendor_name or not model_name:
            raise ModelRegistryError(f"Alias '{alias_key}' must specify both vendor and model")
        if vendor_name not in vendors:
            raise ModelRegistryError(f"Alias '{alias_key}' references unknown vendor '{vendor_name}'")
        if model_name not in vendors[vendor_name].models:
            raise ModelRegistryError(
                f"Alias '{alias_key}' references unknown model '{model_name}' for vendor '{vendor_name}'"
            )
        if provider_name is not None and provider_name not in vendors[vendor_name].providers:
            raise ModelRegistryError(
                f"Alias '{alias_key}' references unknown provider '{provider_name}' for vendor '{vendor_name}'"
            )

        aliases[alias_key] = AliasConfig(
            key=alias_key,
            vendor=vendor_name,
            model=model_name,
            provider=provider_name,
        )

    return ModelRegistry(vendors=vendors, aliases=aliases)


__all__ = [
    "AliasConfig",
    "ModelConfig",
    "ModelRegistry",
    "ModelRegistryError",
    "ProviderConfig",
    "ResolvedModel",
    "VendorConfig",
    "load_model_registry",
]
