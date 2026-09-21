# tests/unit/test_setup_check.py
"""Unit tests for observa.setup_check module."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from observa.setup_check import (
    CheckResult,
    CheckStatus,
    check_config_file,
    check_env_vars,
    check_oracle_connection,
    check_python_deps,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# check_python_deps
# ---------------------------------------------------------------------------


def test_check_python_deps_all_present():
    """All required packages importable → single ok result."""
    # Patch importlib.import_module so every import succeeds
    with patch("observa.setup_check.importlib.import_module", return_value=MagicMock()):
        result = check_python_deps()
    assert isinstance(result, CheckResult)
    assert result.status == "ok"
    assert "ok" in result.message.lower() or result.status == "ok"


def test_check_python_deps_missing_package():
    """A missing package yields a fail result."""

    def _raise_for_oracledb(name: str):
        if name == "oracledb":
            raise ImportError("No module named 'oracledb'")
        return MagicMock()

    with patch("observa.setup_check.importlib.import_module", side_effect=_raise_for_oracledb):
        result = check_python_deps()
    assert result.status == "fail"
    assert "oracledb" in result.message


# ---------------------------------------------------------------------------
# check_env_vars
# ---------------------------------------------------------------------------


def test_check_env_vars_missing_llm_key(monkeypatch):
    """Both LLM API keys absent → fail result for the LLM check.

    The check inspects the active config's required_providers(). To make this
    test stable across config edits (e.g. switching default_provider to oci,
    which doesn't need an env-var-based key), force the fallback path by
    making get_settings unavailable — that branch reverts to the old
    "at least one of ANTHROPIC_API_KEY/OPENAI_API_KEY must be set" rule.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _raise():
        raise RuntimeError("settings unavailable")
    monkeypatch.setattr("observa.setup_check.get_settings", _raise)

    results = check_env_vars()
    llm_result = next((r for r in results if "llm" in r.name.lower() or "api_key" in r.name.lower()), None)
    assert llm_result is not None, "Expected an LLM-key check result"
    assert llm_result.status == "fail"


def test_check_env_vars_missing_oracle_pwd_is_warn(monkeypatch):
    """ORACLE_PWD absent → warn, not fail."""
    monkeypatch.delenv("ORACLE_PWD", raising=False)
    results = check_env_vars()
    oracle_result = next((r for r in results if "oracle" in r.name.lower()), None)
    assert oracle_result is not None, "Expected an Oracle-password check result"
    assert oracle_result.status == "warn"


def test_check_env_vars_all_set(monkeypatch):
    """All env vars set → ok statuses."""
    monkeypatch.setenv("ORACLE_PWD", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    results = check_env_vars()
    assert all(r.status == "ok" for r in results)


def test_check_env_vars_openai_key_accepted(monkeypatch):
    """OPENAI_API_KEY alone satisfies the LLM key requirement."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    results = check_env_vars()
    llm_result = next((r for r in results if "llm" in r.name.lower() or "api_key" in r.name.lower()), None)
    assert llm_result is not None
    assert llm_result.status == "ok"


# ---------------------------------------------------------------------------
# check_oracle_connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_oracle_connection_skipped_when_no_pwd(monkeypatch):
    """No ORACLE_PWD → warn (skip), not fail."""
    monkeypatch.delenv("ORACLE_PWD", raising=False)
    result = await check_oracle_connection(dsn="localhost:1521/XE", user="observa", password="")
    assert result.status == "warn"
    assert "not configured" in result.message.lower() or "skip" in result.message.lower()


@pytest.mark.asyncio
async def test_check_oracle_connection_success(monkeypatch):
    """Successful connection → ok."""
    monkeypatch.setenv("ORACLE_PWD", "secret")
    # Use AsyncMock so that conn.close() returns an awaitable
    mock_conn = AsyncMock()

    with patch("observa.setup_check.oracledb.connect_async") as mock_connect:
        async def _fake_connect(**kwargs):
            return mock_conn

        mock_connect.side_effect = _fake_connect
        result = await check_oracle_connection(dsn="localhost:1521/XE", user="observa", password="secret")
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_check_oracle_connection_failure(monkeypatch):
    """Connection error → fail with error message."""
    monkeypatch.setenv("ORACLE_PWD", "secret")
    with patch("observa.setup_check.oracledb.connect_async") as mock_connect:
        import oracledb as _oracledb

        async def _fail(**kwargs):
            raise _oracledb.DatabaseError("ORA-12541: no listener")

        mock_connect.side_effect = _fail
        result = await check_oracle_connection(dsn="localhost:1521/XE", user="observa", password="secret")
    assert result.status == "fail"
    assert "ORA-12541" in result.message or "listener" in result.message.lower()


# ---------------------------------------------------------------------------
# check_config_file
# ---------------------------------------------------------------------------


def test_check_config_file_ok():
    """get_settings succeeds → ok."""
    mock_settings = MagicMock()
    with patch("observa.setup_check.get_settings", return_value=mock_settings):
        result = check_config_file()
    assert result.status == "ok"


def test_check_config_file_missing():
    """get_settings raises FileNotFoundError → fail."""
    with patch("observa.setup_check.get_settings", side_effect=FileNotFoundError("not found")):
        result = check_config_file()
    assert result.status == "fail"
    assert "not found" in result.message.lower() or "config" in result.message.lower()


def test_check_config_file_bad_yaml():
    """get_settings raises generic exception → fail."""
    with patch("observa.setup_check.get_settings", side_effect=ValueError("bad yaml")):
        result = check_config_file()
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_all_checks_returns_list():
    """run_all_checks returns a non-empty list of CheckResult instances."""
    mock_settings = MagicMock()
    mock_settings.oracle_dsn = "localhost:1521/XE"
    mock_settings.oracle_user = "observa"
    mock_settings.oracle_password = ""

    with (
        patch("observa.setup_check.importlib.import_module", return_value=MagicMock()),
        patch("observa.setup_check.get_settings", return_value=mock_settings),
    ):
        results = await run_all_checks(mock_settings)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, CheckResult) for r in results)


@pytest.mark.asyncio
async def test_run_all_checks_never_propagates_exceptions():
    """run_all_checks catches all internal exceptions and returns results."""
    mock_settings = MagicMock()
    mock_settings.oracle_dsn = "bad-dsn"
    mock_settings.oracle_user = "x"
    mock_settings.oracle_password = "x"

    # Patch get_settings first (resolved before import_module is mocked),
    # then patch importlib.import_module inside setup_check.
    with patch("observa.setup_check.get_settings", side_effect=Exception("config boom")):
        with patch("observa.setup_check.importlib.import_module", side_effect=Exception("boom")):
            # Should not raise — catches everything internally
            results = await run_all_checks(mock_settings)

    assert isinstance(results, list)


def test_check_oci_provider_disabled_returns_ok():
    """Settings without oci_enabled → ok with 'skipped' message."""
    from observa.setup_check import check_oci_provider, CheckResult
    fake = MagicMock(oci_enabled=False)
    with patch("observa.setup_check.get_settings", return_value=fake):
        result = check_oci_provider()
    assert isinstance(result, CheckResult)
    assert result.status == "ok"
    assert "skip" in result.message.lower() or "not configured" in result.message.lower()


def test_check_oci_provider_missing_config_file(tmp_path, monkeypatch):
    """Settings configured but ~/.oci/config absent → fail."""
    fake = MagicMock(
        oci_enabled=True,
        oci_compartment_id="ocid1.compartment.oc1..aaa",
        oci_service_endpoint="https://x",
        oci_auth_profile="DEFAULT",
    )
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "no-such-config"))
    with patch("observa.setup_check.get_settings", return_value=fake):
        from observa.setup_check import check_oci_provider
        result = check_oci_provider()
    assert result.status == "fail"
    assert "not found" in result.message.lower() or "no such" in result.message.lower()
