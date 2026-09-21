"""Per-agent model resolution and provider inference."""
from __future__ import annotations

from observa.config import provider_for_model


def _make_settings(**overrides):
    """Build a Settings with sane defaults plus the given overrides."""
    from observa.config import Settings

    base = dict(
        oracle_host="", oracle_port=1521, oracle_service_name="",
        oracle_user="", oracle_password="", oracle_dsn="",
        oracle_pool_min=1, oracle_pool_max=2,
        llm_provider="openai", llm_model="gpt-5.4-mini", llm_api_key="",
        llm_agent_models={},
        llm_consolidator_model="",
        llm_cove_model="",
        llm_master_chat_model="",
        agents_enable_hypotheses=False, agents_enable_cove=True,
        agents_contradiction_threshold=0.3, agents_verification_support_threshold=0.7,
        llm_hypotheses_model="gpt-5.4-mini",
        turns_count=2, turns_time_budget_seconds=60, turns_max_concurrent_agents=4,
        mcp_sqlcl_path="", mcp_sqlcl_connection_name="",
        mcp_query_row_limit=50, mcp_file_read_byte_limit=65536,
        mcp_enforce_window="warn",
        case_timeline_days=15, case_baseline_offset_days=7,
        ui_theme="x", ui_show_token_cost=True, ui_log_level="INFO",
        hitl_chat_enabled=True, hitl_chat_max_chars=1000,
        agents_enabled={},
        refinement_enabled=False,
        refinement_universal_agents=("wait_time", "ash"),
        refinement_keep_severities=frozenset({"high", "critical"}),
        refinement_enable_cross_reference=True,
        refinement_min_active=3,
        oci_enabled=False,
        oci_default_model="",
        oci_compartment_id="",
        oci_service_endpoint="",
        oci_auth_profile="DEFAULT",
        oci_auth_type="API_KEY",
    )
    base.update(overrides)
    return Settings(**base)


def test_provider_for_model_anthropic():
    assert provider_for_model("claude-sonnet-4-6") == "anthropic"
    assert provider_for_model("claude-opus-4-7") == "anthropic"


def test_provider_for_model_openai_default():
    assert provider_for_model("gpt-4o") == "openai"
    assert provider_for_model("gpt-5.4-mini") == "openai"
    # Anything not matching 'claude' falls to openai (handles o1, o3, etc.)
    assert provider_for_model("o1-mini") == "openai"


def test_model_for_agent_falls_back_to_default():
    s = _make_settings(llm_model="gpt-5.4-mini", llm_agent_models={})
    assert s.model_for_agent("wait_time") == "gpt-5.4-mini"


def test_model_for_agent_uses_override():
    s = _make_settings(
        llm_model="gpt-5.4-mini",
        llm_agent_models={"infra": "gpt-4o-mini", "sql": "claude-sonnet-4-6"},
    )
    assert s.model_for_agent("infra") == "gpt-4o-mini"
    assert s.model_for_agent("sql") == "claude-sonnet-4-6"
    assert s.model_for_agent("wait_time") == "gpt-5.4-mini"  # not overridden


def test_model_for_consolidator_falls_back_to_default():
    s = _make_settings(llm_model="gpt-5.4-mini", llm_consolidator_model="")
    assert s.model_for_consolidator() == "gpt-5.4-mini"


def test_model_for_consolidator_uses_override():
    s = _make_settings(llm_consolidator_model="claude-opus-4-7")
    assert s.model_for_consolidator() == "claude-opus-4-7"


def test_model_for_master_chat_falls_back_to_default():
    s = _make_settings(llm_model="gpt-5.4-mini", llm_master_chat_model="")
    assert s.model_for_master_chat() == "gpt-5.4-mini"


def test_model_for_master_chat_uses_override():
    s = _make_settings(llm_master_chat_model="oci/google.gemini-2.5-pro")
    assert s.model_for_master_chat() == "oci/google.gemini-2.5-pro"


def test_referenced_models_includes_master_chat_override():
    s = _make_settings(
        llm_model="gpt-5.4-mini",
        llm_master_chat_model="oci/google.gemini-2.5-pro",
    )
    assert "oci/google.gemini-2.5-pro" in s.referenced_models()


def test_required_providers_picks_up_per_agent_overrides():
    """A config with one Anthropic per-agent override must require both providers."""
    s = _make_settings(
        llm_model="gpt-5.4-mini",
        llm_agent_models={"sql": "claude-sonnet-4-6"},
    )
    assert s.required_providers() == {"anthropic", "openai"}


def test_required_providers_single_provider_when_homogeneous():
    s = _make_settings(
        llm_model="gpt-5.4-mini",
        llm_hypotheses_model="gpt-5.4-mini",
        llm_agent_models={"infra": "gpt-4o-mini"},
    )
    assert s.required_providers() == {"openai"}


def test_referenced_models_includes_overrides_and_defaults():
    s = _make_settings(
        llm_model="gpt-5.4-mini",
        llm_consolidator_model="claude-opus-4-7",
        llm_agent_models={"sql": "claude-sonnet-4-6"},
    )
    refs = s.referenced_models()
    assert "gpt-5.4-mini" in refs
    assert "claude-opus-4-7" in refs
    assert "claude-sonnet-4-6" in refs
