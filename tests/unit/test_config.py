import os
import pytest
from pathlib import Path


CONFIG_PATH = str(Path(__file__).parent.parent / "fixtures" / "config_canonical.yaml")


def _clear_cache():
    from observa.config import _load_settings
    _load_settings.cache_clear()


def test_loads_defaults():
    from observa.config import get_settings
    _clear_cache()
    settings = get_settings(CONFIG_PATH)
    assert settings.oracle_dsn, "oracle_dsn must be a non-empty string"


def test_oracle_pwd_env_override(monkeypatch):
    from observa.config import get_settings
    _clear_cache()
    monkeypatch.setenv("ORACLE_PWD", "testpwd")
    settings = get_settings(CONFIG_PATH)
    assert settings.oracle_password == "testpwd"


def test_missing_file_raises():
    from observa.config import get_settings
    _clear_cache()
    with pytest.raises((FileNotFoundError, ValueError)):
        get_settings("/nonexistent/path/config.yaml")


def test_config_show_redacts_secrets(monkeypatch):
    from observa.config import get_settings
    _clear_cache()
    monkeypatch.setenv("ORACLE_PWD", "supersecret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    settings = get_settings(CONFIG_PATH)
    redacted = settings.redacted_dict()
    assert redacted["oracle_password"] == "***"
    assert redacted["llm_api_key"] == "***"


def test_hypotheses_settings_loaded():
    from observa.config import get_settings
    _clear_cache()
    settings = get_settings(CONFIG_PATH)
    assert settings.agents_enable_hypotheses is True
    assert settings.agents_contradiction_threshold == 0.3
    assert settings.agents_verification_support_threshold == 0.65
    assert settings.llm_hypotheses_model == "claude-sonnet-4-6"


def test_turns_and_mcp_settings_loaded(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
oracle:
  host: "h"
  port: 1521
  service_name: "s"
  user: "u"
  password_env: "ORACLE_PWD"
llm:
  default_provider: "openai"
  providers:
    openai:
      model: "gpt-5.4-mini"
      api_key_env: "OPENAI_API_KEY"
turns:
  count: 5
  time_budget_seconds: 60
mcp:
  sqlcl_path: "/opt/sqlcl/bin/sql"
  query_row_limit: 200
  file_read_byte_limit: 32768
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_PWD", "x")
    from observa.config import _load_settings
    _load_settings.cache_clear()
    s = _load_settings.__wrapped__(str(cfg))
    assert s.turns_count == 5
    assert s.turns_time_budget_seconds == 60
    assert s.mcp_sqlcl_path == "/opt/sqlcl/bin/sql"
    assert s.mcp_query_row_limit == 200
    assert s.mcp_file_read_byte_limit == 32768


def test_turns_and_mcp_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
oracle:
  host: "h"
  port: 1521
  service_name: "s"
  user: "u"
  password_env: "ORACLE_PWD"
llm:
  default_provider: "openai"
  providers:
    openai:
      model: "gpt-5.4-mini"
      api_key_env: "OPENAI_API_KEY"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_PWD", "x")
    from observa.config import _load_settings
    _load_settings.cache_clear()
    s = _load_settings.__wrapped__(str(cfg))
    assert s.turns_count == 3
    assert s.turns_time_budget_seconds == 120
    assert s.mcp_sqlcl_path == ""
    assert s.mcp_query_row_limit == 500
    assert s.mcp_file_read_byte_limit == 65536


def test_settings_loads_oci_provider(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
oracle:
  host: localhost
  port: 1521
  service_name: ORCLPDB1
  user: observa
  password_env: ORACLE_PWD
llm:
  default_provider: openai
  providers:
    openai:
      model: gpt-5.4-mini
      api_key_env: OPENAI_API_KEY
    oci:
      model: xai.grok-4-fast
      compartment_id: ocid1.compartment.oc1..aaa
      service_endpoint: https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
      auth_profile: DEFAULT
      auth_type: API_KEY
mcp:
  sqlcl_path: ""
  sqlcl_connection_name: observa
""",
        encoding="utf-8",
    )
    from observa.config import get_settings
    s = get_settings(str(cfg))
    assert s.oci_enabled is True
    assert s.oci_default_model == "xai.grok-4-fast"
    assert s.oci_compartment_id == "ocid1.compartment.oc1..aaa"
    assert s.oci_service_endpoint.startswith("https://inference.generativeai.")
    assert s.oci_auth_profile == "DEFAULT"
    assert s.oci_auth_type == "API_KEY"


def test_settings_oci_absent_disabled(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
oracle: {host: localhost, port: 1521, service_name: ORCLPDB1, user: observa, password_env: ORACLE_PWD}
llm:
  default_provider: openai
  providers:
    openai: {model: gpt-5.4-mini, api_key_env: OPENAI_API_KEY}
mcp: {sqlcl_path: "", sqlcl_connection_name: observa}
""",
        encoding="utf-8",
    )
    from observa.config import get_settings
    s = get_settings(str(cfg))
    assert s.oci_enabled is False
    assert s.oci_default_model == ""
    assert s.oci_compartment_id == ""


def test_provider_for_model_oci_prefix():
    from observa.config import provider_for_model
    assert provider_for_model("oci/xai.grok-4-fast") == "oci"
    assert provider_for_model("oci/cohere.command-r-plus-08-2024") == "oci"
    assert provider_for_model("claude-sonnet-4-6") == "anthropic"
    assert provider_for_model("gpt-5.4-mini") == "openai"


def test_required_providers_includes_oci(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
oracle: {host: localhost, port: 1521, service_name: ORCLPDB1, user: observa, password_env: ORACLE_PWD}
llm:
  default_provider: openai
  agent_models:
    infra: oci/xai.grok-4-fast
  providers:
    openai: {model: gpt-5.4-mini, api_key_env: OPENAI_API_KEY}
    oci:
      model: xai.grok-4-fast
      compartment_id: ocid1.compartment.oc1..aaa
      service_endpoint: https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
mcp: {sqlcl_path: "", sqlcl_connection_name: observa}
""",
        encoding="utf-8",
    )
    from observa.config import get_settings
    s = get_settings(str(cfg))
    assert "oci" in s.required_providers()
    assert "openai" in s.required_providers()


def _minimal_yaml() -> str:
    return (
        "oracle:\n"
        "  host: h\n"
        "llm:\n"
        "  default_provider: openai\n"
        "  providers:\n"
        "    openai:\n"
        "      model: gpt-5.5\n"
        "      api_key_env: OPENAI_API_KEY\n"
    )


def test_case_and_window_settings_defaults(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_yaml(), encoding="utf-8")
    from observa.config import _load_settings
    s = _load_settings.__wrapped__(str(cfg))
    assert s.case_timeline_days == 15
    assert s.case_baseline_offset_days == 7
    assert s.mcp_enforce_window == "warn"


def test_case_and_window_settings_loaded(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        _minimal_yaml()
        + "case:\n  timeline_days: 30\n  baseline_offset_days: 14\n"
        + "mcp:\n  enforce_window: reject\n",
        encoding="utf-8",
    )
    from observa.config import _load_settings
    s = _load_settings.__wrapped__(str(cfg))
    assert s.case_timeline_days == 30
    assert s.case_baseline_offset_days == 14
    assert s.mcp_enforce_window == "reject"


def test_enforce_window_invalid_value_falls_back_to_warn(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_minimal_yaml() + "mcp:\n  enforce_window: banana\n", encoding="utf-8")
    from observa.config import _load_settings
    s = _load_settings.__wrapped__(str(cfg))
    assert s.mcp_enforce_window == "warn"
