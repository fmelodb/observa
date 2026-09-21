"""OCI GenAI capability matrix + slot validation.

The matrix is curated against
https://docs.oracle.com/en-us/iaas/Content/generative-ai/model-endpoint-regions.htm
and must be updated when OCI adds or retires chat models. We deliberately list
only models that have been verified to support both tool calling and structured
output via langchain-community's ChatOCIGenAI; adding a new family means
running the manual spike (see plan Task 12) and editing this file.
"""
from __future__ import annotations

from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when a model+slot combination is incompatible or unknown."""


@dataclass(frozen=True)
class ModelCaps:
    tools: bool
    structured: bool


# Verified chat-completion models on OCI GenAI.
#
# What's intentionally NOT here, and why:
#
#   * meta.llama-* — supported by ChatOCIGenAI's MetaProvider in principle,
#     but tool calling is not implemented (convert_to_oci_tool raises
#     NotImplementedError). Not used by Observa. Add only after a manual
#     spike confirms tools + structured output for the specific model id.
#
#   * openai.gpt-oss-* — same caveat as Meta; not yet validated.
#
# What IS here:
#
#   * cohere.* — verified via langchain-community's CohereProvider.
#   * xai.grok-4-fast-* — verified via observa-shipped XaiProvider, which
#     implements the GENERIC chat-request schema with tool calling. See
#     observa/llm/oci_providers.py.
#   * google.gemini-2.5-* — verified via observa-shipped GoogleProvider,
#     sharing the same GENERIC chat-request implementation as XaiProvider.
OCI_CAPS: dict[str, ModelCaps] = {
    "cohere.command-a-03-2025":         ModelCaps(tools=True, structured=True),
    "cohere.command-r-plus-08-2024":    ModelCaps(tools=True, structured=True),
    "cohere.command-r-08-2024":         ModelCaps(tools=True, structured=True),
    "xai.grok-4-fast-non-reasoning":    ModelCaps(tools=True, structured=True),
    "xai.grok-4-fast-reasoning":        ModelCaps(tools=True, structured=True),
    "google.gemini-2.5-flash":          ModelCaps(tools=True, structured=True),
    "google.gemini-2.5-pro":            ModelCaps(tools=True, structured=True),
    "openai.gpt-5.5":                   ModelCaps(tools=True, structured=True),
    "openai.gpt-5.4-mini":              ModelCaps(tools=True, structured=True),
}

SLOT_REQUIREMENTS: dict[str, ModelCaps] = {
    "agent":        ModelCaps(tools=True,  structured=True),
    "master_chat":  ModelCaps(tools=True,  structured=False),
    "consolidator": ModelCaps(tools=False, structured=True),
    "cove":         ModelCaps(tools=False, structured=True),
    "hypotheses":   ModelCaps(tools=False, structured=True),
}


def compatible_models_for_slot(slot: str) -> list[str]:
    """Return the sorted list of OCI model ids that satisfy a slot's needs."""
    if slot not in SLOT_REQUIREMENTS:
        return []
    req = SLOT_REQUIREMENTS[slot]
    out = [
        model_id
        for model_id, caps in OCI_CAPS.items()
        if (not req.tools or caps.tools) and (not req.structured or caps.structured)
    ]
    return sorted(out)


def validate_oci_model_for_slot(model_id: str, slot: str) -> None:
    """Validate that an OCI model satisfies a slot's capability requirements.

    Returns None on success; raises ConfigError with a message that names the
    offending model, the slot's requirements, and the list of compatible models.
    """
    if slot not in SLOT_REQUIREMENTS:
        raise ConfigError(
            f"Unknown LLM slot {slot!r}. Known slots: "
            f"{sorted(SLOT_REQUIREMENTS)}."
        )
    if model_id not in OCI_CAPS:
        raise ConfigError(
            f"OCI model {model_id!r} not in OCI_CAPS. "
            f"Verify it against the Oracle docs and add it explicitly to "
            f"observa/llm/oci_capabilities.py, or pick one of: "
            f"{compatible_models_for_slot(slot)}."
        )
    caps = OCI_CAPS[model_id]
    req = SLOT_REQUIREMENTS[slot]
    missing: list[str] = []
    if req.tools and not caps.tools:
        missing.append("tool calling")
    if req.structured and not caps.structured:
        missing.append("structured output")
    if missing:
        raise ConfigError(
            f"OCI model {model_id!r} does not support {', '.join(missing)} "
            f"(required by slot {slot!r}). Compatible models: "
            f"{compatible_models_for_slot(slot)}."
        )
