# observa/setup_check.py
"""Environment and dependency checks for `observa setup check`."""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass
from typing import Literal

import oracledb

from observa.config import get_settings

CheckStatus = Literal["ok", "warn", "fail"]

# Packages that must be importable for Observa to run.
_REQUIRED_PACKAGES = ["oracledb", "langgraph", "textual", "mcp"]


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_deps() -> CheckResult:
    """Verify all required Python packages can be imported.

    Returns a single CheckResult: ok if all present, fail listing the missing ones.
    """
    missing: list[str] = []
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        return CheckResult(
            name="python_deps",
            status="fail",
            message=f"Missing packages: {', '.join(missing)}",
        )
    return CheckResult(
        name="python_deps",
        status="ok",
        message=f"All required packages present ({', '.join(_REQUIRED_PACKAGES)})",
    )


def check_env_vars() -> list[CheckResult]:
    """Verify required environment variables are set.

    Returns two CheckResult items: one for ORACLE_PWD, one for LLM API key.
    ORACLE_PWD absence is a warn (read-only mode possible); LLM key absence is fail.
    """
    results: list[CheckResult] = []

    # Oracle password — warn only (can still inspect local artifacts without DB)
    if os.environ.get("ORACLE_PWD"):
        results.append(
            CheckResult(
                name="oracle_pwd",
                status="ok",
                message="ORACLE_PWD is set",
            )
        )
    else:
        results.append(
            CheckResult(
                name="oracle_pwd",
                status="warn",
                message="ORACLE_PWD not set — Oracle connection unavailable",
            )
        )

    # LLM key — verify EVERY provider referenced by the active configuration
    # has its API key set. With per-agent model overrides we can have a config
    # that uses both Anthropic and OpenAI in the same case.
    try:
        settings = get_settings()
        required = settings.required_providers()
    except Exception:  # noqa: BLE001
        required = set()

    if not required:
        # Config didn't load — fall back to "at least one key present".
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        if has_anthropic or has_openai:
            provider = "ANTHROPIC_API_KEY" if has_anthropic else "OPENAI_API_KEY"
            results.append(CheckResult(
                name="llm_api_key", status="ok",
                message=f"LLM API key present ({provider})",
            ))
        else:
            results.append(CheckResult(
                name="llm_api_key", status="fail",
                message="No LLM API key found — set ANTHROPIC_API_KEY or OPENAI_API_KEY",
            ))
        return results

    env_for_provider = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    missing: list[str] = []
    present: list[str] = []
    for prov in sorted(required):
        env_var = env_for_provider.get(prov, "")
        if env_var and os.environ.get(env_var):
            present.append(env_var)
        elif env_var:
            missing.append(env_var)
    if missing:
        results.append(CheckResult(
            name="llm_api_key", status="fail",
            message=(
                f"Missing API key(s) required by config: {', '.join(missing)}. "
                f"Active providers: {sorted(required)}."
            ),
        ))
    else:
        results.append(CheckResult(
            name="llm_api_key", status="ok",
            message=f"LLM API keys present for active providers: {', '.join(present)}",
        ))

    return results


async def check_oracle_connection(
    *,
    dsn: str,
    user: str,
    password: str,
) -> CheckResult:
    """Attempt a test connection to Oracle with a 5-second timeout.

    If ``password`` is empty (ORACLE_PWD not set), returns a warn without
    attempting a connection.
    """
    if not password:
        return CheckResult(
            name="oracle_connection",
            status="warn",
            message="Oracle credentials not configured — skipping connection test",
        )

    async def _connect() -> None:
        conn = await oracledb.connect_async(user=user, password=password, dsn=dsn)
        await conn.close()

    try:
        await asyncio.wait_for(_connect(), timeout=5.0)
        return CheckResult(
            name="oracle_connection",
            status="ok",
            message=f"Connected to Oracle at {dsn}",
        )
    except asyncio.TimeoutError:
        return CheckResult(
            name="oracle_connection",
            status="fail",
            message=f"Connection to {dsn} timed out after 5 seconds",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="oracle_connection",
            status="fail",
            message=str(exc),
        )


def check_sqlcl_path() -> CheckResult:
    """Verify mcp.sqlcl_path points at an existing file (warn when blank)."""
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name="sqlcl_path", status="fail", message=f"config error: {exc}")
    path = settings.mcp_sqlcl_path
    if not path:
        return CheckResult(
            name="sqlcl_path",
            status="warn",
            message="mcp.sqlcl_path is blank — agents will run without SQLcl MCP (query_awr disabled)",
        )
    if not os.path.isfile(path):
        return CheckResult(
            name="sqlcl_path",
            status="fail",
            message=f"mcp.sqlcl_path is set but file not found: {path}",
        )
    return CheckResult(
        name="sqlcl_path",
        status="ok",
        message=f"SQLcl binary located at {path}",
    )


async def check_sqlcl_connection() -> CheckResult:
    """Open an MCP session and call `connect` to confirm the saved connection works."""
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="sqlcl_connection", status="fail", message=f"config error: {exc}"
        )
    if not settings.mcp_sqlcl_path or not os.path.isfile(settings.mcp_sqlcl_path):
        return CheckResult(
            name="sqlcl_connection",
            status="warn",
            message="sqlcl_path not usable — skipping connection test",
        )
    if not settings.mcp_sqlcl_connection_name:
        return CheckResult(
            name="sqlcl_connection",
            status="fail",
            message=(
                "mcp.sqlcl_connection_name is blank. Save a SQLcl connection with "
                "`sql /nolog`, `connect <user>/<pwd>@<dsn>`, `conn -save <name>` "
                "and set that name in config/config.yaml."
            ),
        )
    try:
        from observa.mcp.client import McpClient

        async with McpClient(settings, []) as _:
            return CheckResult(
                name="sqlcl_connection",
                status="ok",
                message=(
                    f"SQLcl MCP connected via saved connection "
                    f"{settings.mcp_sqlcl_connection_name!r}"
                ),
            )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="sqlcl_connection",
            status="fail",
            message=f"{exc}",
        )


def check_oci_provider() -> CheckResult:
    """Validate llm.providers.oci config when present.

    Skips silently when the provider section is absent. Otherwise checks:
      * ~/.oci/config exists and contains the configured profile
      * the `oci` SDK is importable
      * a cheap list-models call succeeds against the configured compartment
    """
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name="oci_provider", status="fail", message=f"config error: {exc}")

    if not getattr(settings, "oci_enabled", False):
        return CheckResult(
            name="oci_provider",
            status="ok",
            message="OCI provider not configured — skipped",
        )

    # 1. ~/.oci/config and profile
    oci_config = os.path.expanduser("~/.oci/config")
    if not os.path.isfile(oci_config):
        return CheckResult(
            name="oci_provider",
            status="fail",
            message=(
                f"OCI provider configured but {oci_config} not found. "
                f"Run `oci setup config` or place a config file there."
            ),
        )
    try:
        with open(oci_config, "r", encoding="utf-8") as fh:
            content = fh.read()
        profile_header = f"[{settings.oci_auth_profile}]"
        if profile_header not in content:
            return CheckResult(
                name="oci_provider",
                status="fail",
                message=f"OCI profile {settings.oci_auth_profile!r} not found in {oci_config}",
            )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="oci_provider",
            status="fail",
            message=f"Could not read {oci_config}: {exc}",
        )

    # 2. SDK importable
    try:
        importlib.import_module("oci")
    except ImportError:
        return CheckResult(
            name="oci_provider",
            status="fail",
            message="OCI SDK not installed. Run: uv sync --extra oci",
        )

    # 3. Cheap credential round-trip — list compartment-visible models.
    try:
        import oci as oci_sdk  # type: ignore[import-not-found]
        from oci.generative_ai import GenerativeAiClient  # type: ignore[import-not-found]
        config = oci_sdk.config.from_file(file_location=oci_config, profile_name=settings.oci_auth_profile)
        client = GenerativeAiClient(config)
        client.list_models(compartment_id=settings.oci_compartment_id, limit=1)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="oci_provider",
            status="fail",
            message=f"OCI list_models failed: {exc}",
        )

    return CheckResult(
        name="oci_provider",
        status="ok",
        message=(
            f"OCI provider OK — profile={settings.oci_auth_profile!r} "
            f"compartment={settings.oci_compartment_id[:24]}..."
        ),
    )


def check_config_file() -> CheckResult:
    """Verify the config file exists and loads without error."""
    try:
        get_settings()
        return CheckResult(
            name="config_file",
            status="ok",
            message="Config file loaded successfully",
        )
    except FileNotFoundError as exc:
        return CheckResult(
            name="config_file",
            status="fail",
            message=f"Config file not found: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="config_file",
            status="fail",
            message=f"Config file error: {exc}",
        )


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


async def run_all_checks(settings: object) -> list[CheckResult]:
    """Run all environment checks and return results.

    Catches all exceptions internally — never propagates to the caller.
    ``settings`` is accepted for forward-compatibility (DSN, user come from it
    when available), but checks fall back to env vars / defaults when settings
    fields are missing or settings is a stub.
    """
    results: list[CheckResult] = []

    # 1. Python dependencies (sync)
    try:
        results.append(check_python_deps())
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(name="python_deps", status="fail", message=f"Unexpected error: {exc}")
        )

    # 2. Environment variables (sync)
    try:
        results.extend(check_env_vars())
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(name="env_vars", status="fail", message=f"Unexpected error: {exc}")
        )

    # 3. Oracle connection (async)
    try:
        dsn = getattr(settings, "oracle_dsn", "") or ""
        user = getattr(settings, "oracle_user", "") or ""
        password = getattr(settings, "oracle_password", "") or os.environ.get("ORACLE_PWD", "")
        results.append(
            await check_oracle_connection(dsn=dsn, user=user, password=password)
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(
                name="oracle_connection",
                status="fail",
                message=f"Unexpected error: {exc}",
            )
        )

    # 4. Config file (sync)
    try:
        results.append(check_config_file())
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(name="config_file", status="fail", message=f"Unexpected error: {exc}")
        )

    # 5. SQLcl MCP binary (sync)
    try:
        results.append(check_sqlcl_path())
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(name="sqlcl_path", status="fail", message=f"Unexpected error: {exc}")
        )

    # 6. SQLcl MCP saved-connection round-trip (async)
    try:
        results.append(await check_sqlcl_connection())
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(
                name="sqlcl_connection", status="fail", message=f"Unexpected error: {exc}"
            )
        )

    # 7. OCI provider (sync, but does a network call when configured)
    try:
        results.append(check_oci_provider())
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(name="oci_provider", status="fail", message=f"Unexpected error: {exc}")
        )

    return results
