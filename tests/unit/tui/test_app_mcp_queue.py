"""Queue-protocol test for ObservaApp._mcp_lifecycle (no Textual, no subprocesses)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _bare_app():
    from observa.tui.app import ObservaApp

    app = ObservaApp.__new__(ObservaApp)  # bypass Textual App.__init__
    app._mcp_commands = asyncio.Queue()
    app._mcp_ready = asyncio.Event()
    app._mcp_error = None
    app._mcp_client = None
    return app


@pytest.mark.asyncio
async def test_lifecycle_attach_then_shutdown():
    fake_client = MagicMock()
    fake_client.attach_files = AsyncMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    app = _bare_app()
    with patch("observa.mcp.client.McpClient", return_value=fake_client), \
         patch("observa.config.get_settings", return_value=MagicMock(mcp_sqlcl_path="sql")):
        task = asyncio.create_task(app._mcp_lifecycle())
        await app._mcp_ready.wait()
        assert app._mcp_client is fake_client

        entries = [MagicMock()]
        done = asyncio.Event()
        await app._mcp_commands.put((entries, done))
        await done.wait()
        fake_client.attach_files.assert_awaited_once_with(entries)

        await app._mcp_commands.put(None)
        await task
    assert app._mcp_client is None
    assert app._mcp_error is None


@pytest.mark.asyncio
async def test_lifecycle_attach_failure_sets_error_and_releases_waiter():
    fake_client = MagicMock()
    fake_client.attach_files = AsyncMock(side_effect=RuntimeError("boom"))
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    app = _bare_app()
    with patch("observa.mcp.client.McpClient", return_value=fake_client), \
         patch("observa.config.get_settings", return_value=MagicMock(mcp_sqlcl_path="sql")):
        task = asyncio.create_task(app._mcp_lifecycle())
        await app._mcp_ready.wait()

        done = asyncio.Event()
        await app._mcp_commands.put(([], done))
        await asyncio.wait_for(done.wait(), timeout=2)  # finally must release us
        assert app._mcp_error and "attach_files" in app._mcp_error

        await app._mcp_commands.put(None)
        await task


@pytest.mark.asyncio
async def test_boot_failure_still_releases_wait_mcp():
    failing = MagicMock()
    failing.__aenter__ = AsyncMock(side_effect=RuntimeError("no sqlcl"))
    failing.__aexit__ = AsyncMock(return_value=False)

    app = _bare_app()
    with patch("observa.mcp.client.McpClient", return_value=failing), \
         patch("observa.config.get_settings", return_value=MagicMock(mcp_sqlcl_path="sql")):
        await app._mcp_lifecycle()
    client, error = await app.wait_mcp()
    assert client is None
    assert error and "no sqlcl" in error
