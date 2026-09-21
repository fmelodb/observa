"""Eager validation of LLM model assignments at app startup.

Walks every slot in Settings that may receive a model string. For values with
the ``oci/`` prefix, validates the resolved model id against
SLOT_REQUIREMENTS. Aggregates every error before raising so the user sees the
full list of problems in a single pass.
"""
from __future__ import annotations

from observa.llm.oci_capabilities import (
    ConfigError,
    validate_oci_model_for_slot,
)


def _check_one(field: str, model: str, slot: str, errors: list[str]) -> None:
    if not model or not model.startswith("oci/"):
        return
    model_id = model[len("oci/"):]
    try:
        validate_oci_model_for_slot(model_id, slot)
    except ConfigError as exc:
        errors.append(f'{field}="{model}" — {exc}')


def validate_llm_config(settings) -> None:
    """Validate every OCI model assignment in Settings.

    Raises ConfigError with an aggregated message naming every offending slot
    and a list of compatible alternatives, or returns None on success.
    """
    errors: list[str] = []

    # Per-agent overrides
    for agent_name, model in (settings.llm_agent_models or {}).items():
        _check_one(f"agent_models.{agent_name}", model, "agent", errors)

    # Master/consolidator/cove/master_chat
    _check_one("consolidator_model", settings.llm_consolidator_model, "consolidator", errors)
    _check_one("cove_model", settings.llm_cove_model, "cove", errors)
    _check_one(
        "master_chat_model",
        getattr(settings, "llm_master_chat_model", ""),
        "master_chat",
        errors,
    )

    # Hypotheses (synthesis + contradiction resolution)
    _check_one("hypotheses_model", settings.llm_hypotheses_model, "hypotheses", errors)

    # Default model - used as fallback for slots that omit theirs. Treat it
    # under the strictest contract (agent) since it can be used anywhere.
    if settings.llm_model and settings.llm_model.startswith("oci/"):
        _check_one("llm.providers.<default>.model", settings.llm_model, "agent", errors)

    if errors:
        raise ConfigError(
            "Config invalid:\n  - " + "\n  - ".join(errors)
        )
