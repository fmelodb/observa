"""Factory for ChatOCIGenAI instances configured from observa Settings."""
from __future__ import annotations

from typing import Any

from observa.config import get_settings
from observa.llm.oci_capabilities import ConfigError


# OCI vendor prefixes that ChatOCIGenAI cannot auto-derive from model_id when
# a custom service_endpoint is supplied. We pass `provider=` explicitly so the
# correct chat schema is selected. Cohere is auto-derived but we set it for
# symmetry.
_PROVIDER_BY_PREFIX = {
    "cohere": "cohere",
    "meta":   "meta",
    "xai":    "xai",
    "google": "google",
    "openai": "openai",
}


def _provider_from_model_id(model_id: str) -> str | None:
    if "." not in model_id:
        return None
    return _PROVIDER_BY_PREFIX.get(model_id.split(".", 1)[0].lower())


def _import_chat_oci_genai() -> Any:
    """Late-import the Observa-shipped ChatOCIGenAI subclass.

    The subclass overrides ChatOCIGenAI._provider_map to add XaiProvider
    (and, eventually, GoogleProvider). Isolated as a single seam so tests
    can monkey-patch this without touching real langchain_community imports.
    """
    from observa.llm.oci_providers import ObservaChatOCIGenAI
    return ObservaChatOCIGenAI


def build_oci_chat(model_id: str) -> Any:
    """Build a configured ChatOCIGenAI instance.

    Raises:
        ConfigError: if llm.providers.oci is not configured (no compartment_id
            / service_endpoint), with a message pointing at config.yaml.
        ConfigError: if the langchain_community / oci extras are not installed,
            with the exact ``uv sync`` command to fix it.
    """
    settings = get_settings()
    if not getattr(settings, "oci_enabled", False) or not settings.oci_compartment_id \
            or not settings.oci_service_endpoint:
        raise ConfigError(
            "OCI provider not configured. Add llm.providers.oci with "
            "compartment_id and service_endpoint to config.yaml. See "
            "https://docs.oracle.com/en-us/iaas/Content/generative-ai/"
            "model-endpoint-regions.htm"
        )
    try:
        chat_cls = _import_chat_oci_genai()
    except ImportError as exc:
        raise ConfigError(
            "OCI provider requires extras. Run: uv sync --extra oci"
        ) from exc
    # OpenAI models on OCI (gpt-5.5+) have two quirks vs. other OCI vendors:
    #   * reject ``max_tokens`` — require ``max_completion_tokens`` instead.
    #   * reject any non-default ``temperature`` — only the default (1) is
    #     supported, so we omit the field entirely.
    # OCI's GenericChatRequest accepts both token-limit field names.
    is_openai = model_id.lower().startswith("openai.")
    model_kwargs: dict[str, Any] = {}
    if not is_openai:
        model_kwargs["temperature"] = 0.0
    model_kwargs["max_completion_tokens" if is_openai else "max_tokens"] = 4000
    kwargs: dict[str, Any] = dict(
        model_id=model_id,
        compartment_id=settings.oci_compartment_id,
        service_endpoint=settings.oci_service_endpoint,
        auth_type=settings.oci_auth_type,
        auth_profile=settings.oci_auth_profile,
        model_kwargs=model_kwargs,
    )
    provider = _provider_from_model_id(model_id)
    if provider is not None:
        kwargs["provider"] = provider
    return chat_cls(**kwargs)
