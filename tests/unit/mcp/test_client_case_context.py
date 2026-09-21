import asyncio

import pytest

import observa.mcp.client as client_mod
from observa.config import Settings
from observa.mcp.client import InputFileEntry, McpClient


def _settings(**over) -> Settings:
    base = dict(
        oracle_host="h", oracle_port=1521, oracle_service_name="s",
        oracle_user="u", oracle_password="", oracle_dsn="h:1521/s",
        oracle_pool_min=1, oracle_pool_max=2,
        llm_provider="openai", llm_model="gpt-5.5", llm_api_key="",
        llm_agent_models={}, llm_consolidator_model="", llm_cove_model="",
        llm_master_chat_model="",
        oci_enabled=False, oci_default_model="", oci_compartment_id="",
        oci_service_endpoint="", oci_auth_profile="DEFAULT", oci_auth_type="API_KEY",
        agents_enable_hypotheses=False, agents_enable_cove=False,
        agents_contradiction_threshold=0.3, agents_verification_support_threshold=0.7,
        llm_hypotheses_model="m",
        turns_count=1, turns_time_budget_seconds=10, turns_max_concurrent_agents=1,
        mcp_sqlcl_path="", mcp_sqlcl_connection_name="c",
        mcp_query_row_limit=50, mcp_file_read_byte_limit=65536,
        mcp_enforce_window="warn",
        case_timeline_days=15, case_baseline_offset_days=7,
        ui_theme="flexoki", ui_show_token_cost=True, ui_log_level="INFO",
        hitl_chat_enabled=True, hitl_chat_max_chars=2000,
        agents_enabled={}, refinement_enabled=False,
        refinement_universal_agents=(), refinement_keep_severities=frozenset(),
        refinement_enable_cross_reference=False, refinement_min_active=1,
    )
    base.update(over)
    return Settings(**base)


def test_input_files_optional_in_constructor():
    client = McpClient(_settings())
    assert client._input_files == []


def test_set_case_context_activates_enforcement_preserving_tool_choice():
    client = McpClient(_settings(mcp_enforce_window="reject"), [])
    before = client._guard_cfg
    client.set_case_context(111)
    cfg = client._guard_cfg
    assert cfg.allowlist.case_dbid == 111
    assert cfg.allowlist.enforce_window == "reject"
    assert cfg.allowlist.row_limit == 50
    assert cfg.tool_name == before.tool_name and cfg.sql_arg == before.sql_arg


@pytest.mark.asyncio
async def test_query_catalog_bypasses_case_enforcement(monkeypatch):
    captured = {}

    async def fake_guarded_query(session, sql, config):
        captured["config"] = config
        return {"rows": [], "truncated": False, "row_count": 0, "warnings": []}

    monkeypatch.setattr(client_mod, "guarded_query", fake_guarded_query)
    client = McpClient(_settings())
    client._sqlcl = object()
    client._sqlcl_lock = asyncio.Lock()
    client.set_case_context(111)

    await client.query_catalog("SELECT 1 FROM DBA_HIST_SNAPSHOT", row_limit=2000)

    assert captured["config"].allowlist.case_dbid == 0
    assert captured["config"].allowlist.row_limit == 2000
    assert client._guard_cfg.allowlist.case_dbid == 111


def test_set_case_context_last_call_wins():
    client = McpClient(_settings())
    before = client._guard_cfg
    client.set_case_context(111)
    client.set_case_context(222)
    assert client._guard_cfg.allowlist.case_dbid == 222
    assert client._guard_cfg.tool_name == before.tool_name
