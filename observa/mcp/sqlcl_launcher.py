"""Launches Oracle SQLcl as an MCP stdio subprocess.

SQLcl >= 25.x ships an MCP server mode (``sql -mcp``). We spawn it and hand a
connected ``ClientSession`` to the guarded query layer.

The connect string is read from ``settings.mcp_sqlcl_connect_string`` when set;
otherwise we derive ``user/<ORACLE_PWD>@host:port/service`` from the ``oracle.*``
config + ``ORACLE_PWD`` environment variable.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from observa.config import Settings

logger = logging.getLogger(__name__)


class SqlclNotConfigured(RuntimeError):
    """Raised when settings do not point to a usable SQLcl binary."""


def _build_params(settings: Settings) -> StdioServerParameters:
    path = settings.mcp_sqlcl_path
    if not path:
        raise SqlclNotConfigured(
            "mcp.sqlcl_path is not set in config/config.yaml. Install SQLcl >= 25.x "
            "and set mcp.sqlcl_path to the 'sql' (Windows: sql.exe) binary."
        )
    if not os.path.isfile(path):
        raise SqlclNotConfigured(f"SQLcl binary not found: {path}")
    logger.info("sqlcl launcher: spawning %r -mcp", path)
    return StdioServerParameters(command=path, args=["-mcp"], env=os.environ.copy())


_SQLCL_STDERR_LOG = "observa-sqlcl.log"


@asynccontextmanager
async def sqlcl_session(settings: Settings) -> AsyncIterator[ClientSession]:
    """Context manager that yields an initialized MCP ClientSession to SQLcl.

    Redirects the SQLcl subprocess stderr to a dedicated log file (``observa-sqlcl.log``)
    rather than inheriting the parent's stderr. Inside a Textual TUI the parent's
    stderr is a captured stream whose ``fileno()`` is invalid on Windows, which
    makes ``msvcrt.get_osfhandle`` fail with "Bad file descriptor" when asyncio's
    subprocess machinery tries to wire it up.
    """
    params = _build_params(settings)
    logger.debug(
        "sqlcl launcher: opening stdio pipe to %s (stderr → %s)",
        params.command, _SQLCL_STDERR_LOG,
    )
    try:
        # Binary mode + line-buffered so SQLcl's messages land on disk promptly.
        errfile = open(_SQLCL_STDERR_LOG, "a", encoding="utf-8", buffering=1)
    except OSError as exc:
        logger.warning("sqlcl launcher: cannot open %s (%s); using os.devnull", _SQLCL_STDERR_LOG, exc)
        errfile = open(os.devnull, "w")
    try:
        async with stdio_client(params, errlog=errfile) as (read, write):
            logger.debug("sqlcl launcher: stdio pipe open; initializing ClientSession")
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("sqlcl launcher: MCP session initialized")
                yield session
    except Exception:
        logger.exception("sqlcl launcher: session setup failed")
        raise
    finally:
        try:
            errfile.close()
        except Exception:  # noqa: BLE001
            pass
