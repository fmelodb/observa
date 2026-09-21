"""MCP stdio server exposing read-only access to user-specified diagnostic files.

Launched via ``python -m observa.mcp.files_server``. The file registry is
injected through the ``OBSERVA_FILES_JSON`` environment variable at launch
time. Expected shape:

    [{"type": "alertlog", "path": "C:/...", "label": "prod1_alert.log"}, ...]

Four MCP tools are exposed:
    * ``list_files``   — returns the registered files with metadata.
    * ``read_file``    — reads a byte slice of a registered file.
    * ``search_file``  — regex-greps a registered file (line + byte offset matches).
    * ``extract_ora``  — extracts ORA-/incident lines from a registered alert log.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from observa.mcp.file_scan import (
    ORA_MAX_BYTES,
    SEARCH_MAX_MATCHES,
    SEARCH_MAX_SCAN_BYTES,
    extract_ora_incidents,
    scan_matches,
)

DEFAULT_MAX_BYTES = 64 * 1024  # 64 KB per read by default
MAX_READ_BYTES = 1024 * 1024  # Hard cap: 1 MB per call


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def _load_file_registry() -> dict[str, dict[str, Any]]:
    """Parse ``OBSERVA_FILES_JSON`` and return a registry keyed by resolved path."""
    raw = os.environ.get("OBSERVA_FILES_JSON", "[]")
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        entries = []
    registry: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        key = str(Path(entry["path"]).resolve())
        registry[key] = {
            "type": entry.get("type", "unknown"),
            "path": key,
            "label": entry.get("label", Path(key).name),
        }
    return registry


# ---------------------------------------------------------------------------
# Pure, testable helpers
# ---------------------------------------------------------------------------

def _list_files(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return registry entries enriched with ``size_bytes`` when available."""
    result: list[dict[str, Any]] = []
    for key, entry in registry.items():
        try:
            size = Path(key).stat().st_size
        except OSError:
            size = -1
        result.append({
            "type": entry["type"],
            "path": entry["path"],
            "label": entry["label"],
            "size_bytes": size,
        })
    return result


def _read_file(
    registry: dict[str, dict[str, Any]],
    path: str,
    offset: int = 0,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Read a slice of a registered file. Raises ``ValueError`` on rejection."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    # Enforce the hard cap regardless of what the caller requested.
    effective_max = min(max_bytes, MAX_READ_BYTES)

    resolved = str(Path(path).resolve())
    if resolved not in registry:
        raise ValueError(f"path not registered: {path}")

    file_path = Path(resolved)
    total_size = file_path.stat().st_size

    if offset > total_size:
        # Out-of-range offset yields empty content with EOF flag.
        return {
            "content": "",
            "offset": offset,
            "bytes_read": 0,
            "eof": True,
            "truncated": False,
            "total_size": total_size,
        }

    with file_path.open("rb") as fh:
        fh.seek(offset)
        raw = fh.read(effective_max)

    bytes_read = len(raw)
    end_pos = offset + bytes_read
    eof = end_pos >= total_size
    # "truncated" means we hit the byte limit before reaching EOF.
    truncated = (not eof) and bytes_read >= effective_max

    content = raw.decode("utf-8", errors="replace")

    return {
        "content": content,
        "offset": offset,
        "bytes_read": bytes_read,
        "eof": eof,
        "truncated": truncated,
        "total_size": total_size,
    }


def _search_file(
    registry: dict[str, dict[str, Any]],
    path: str,
    pattern: str,
    max_matches: int = SEARCH_MAX_MATCHES,
) -> dict[str, Any]:
    """Regex-grep a registered file. Raises ``ValueError`` on rejection."""
    resolved = str(Path(path).resolve())
    if resolved not in registry:
        raise ValueError(f"path not registered: {path}")
    max_matches = max(1, min(max_matches, SEARCH_MAX_MATCHES))
    return scan_matches(
        resolved, pattern,
        max_matches=max_matches, max_scan_bytes=SEARCH_MAX_SCAN_BYTES,
    )


def _extract_ora(
    registry: dict[str, dict[str, Any]], path: str
) -> dict[str, Any]:
    """Extract ORA/incident lines from a registered alert log."""
    resolved = str(Path(path).resolve())
    if resolved not in registry:
        raise ValueError(f"path not registered: {path}")
    return extract_ora_incidents(resolved, max_bytes=ORA_MAX_BYTES)


# ---------------------------------------------------------------------------
# MCP app + thin tool wrappers
# ---------------------------------------------------------------------------

app = FastMCP("observa-files")
_REGISTRY: dict[str, dict[str, Any]] = _load_file_registry()


@app.tool()
def list_files() -> list[dict[str, Any]]:
    """Return the list of available diagnostic files.

    Each entry has fields: ``type``, ``path``, ``label``, ``size_bytes``.
    """
    return _list_files(_REGISTRY)


@app.tool()
def read_file(
    path: str,
    offset: int = 0,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Read a slice of one of the registered files.

    Args:
        path: Absolute or relative path to a registered file.
        offset: Byte offset at which to start reading (default 0).
        max_bytes: Maximum number of bytes to return (capped at 1 MB).

    Returns:
        A dict with ``content``, ``offset``, ``bytes_read``, ``eof``,
        ``truncated``, and ``total_size``.

    Raises:
        ValueError: If the path is not in the registry or arguments are invalid.
    """
    return _read_file(_REGISTRY, path, offset=offset, max_bytes=max_bytes)


@app.tool()
def search_file(
    path: str,
    pattern: str,
    max_matches: int = SEARCH_MAX_MATCHES,
) -> dict[str, Any]:
    """Regex-search a registered file.

    Returns ``{matches, match_count, scanned_bytes, truncated, total_size}``
    where each match is ``{line_number, byte_offset, line}``. The ``byte_offset``
    feeds ``read_file`` to fetch surrounding context. Bounded by hard caps
    (200 matches, 64 MB scanned) — a runaway pattern cannot hang.
    """
    return _search_file(_REGISTRY, path, pattern, max_matches=max_matches)


@app.tool()
def extract_ora(path: str) -> dict[str, Any]:
    """Extract ORA-/incident lines from a registered alert log, each tied to
    its block timestamp. Returns ``{incidents, scanned_bytes, truncated,
    coverage}``. Internal — used by the ora_timeline probe."""
    return _extract_ora(_REGISTRY, path)


def main() -> None:
    """Entry point for ``python -m observa.mcp.files_server``."""
    app.run()


if __name__ == "__main__":
    main()
