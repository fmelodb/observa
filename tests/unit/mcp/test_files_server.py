"""Unit tests for the MCP diagnostic-files stdio server."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from observa.mcp.files_server import (
    MAX_READ_BYTES,
    _extract_ora,
    _list_files,
    _load_file_registry,
    _read_file,
    _search_file,
)


def _build_registry(entries: list[dict]) -> dict[str, dict]:
    return {str(Path(e["path"]).resolve()): e for e in entries}


def test_list_files_returns_registered_entries(tmp_path: Path) -> None:
    f1 = tmp_path / "alert.log"
    f1.write_text("hello alert", encoding="utf-8")
    f2 = tmp_path / "trace.trc"
    f2.write_text("trace content here", encoding="utf-8")

    registry = _build_registry([
        {"type": "alertlog", "path": str(f1), "label": "alert.log"},
        {"type": "trace", "path": str(f2), "label": "trace.trc"},
    ])

    result = _list_files(registry)

    assert len(result) == 2
    labels = {r["label"] for r in result}
    assert labels == {"alert.log", "trace.trc"}
    sizes = {r["label"]: r["size_bytes"] for r in result}
    assert sizes["alert.log"] == len("hello alert")
    assert sizes["trace.trc"] == len("trace content here")
    for r in result:
        assert "type" in r and "path" in r


def test_load_file_registry_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f1 = tmp_path / "a.log"
    f1.write_text("x", encoding="utf-8")
    f2 = tmp_path / "b.log"
    f2.write_text("y", encoding="utf-8")

    payload = json.dumps([
        {"type": "alertlog", "path": str(f1), "label": "a"},
        {"type": "trace", "path": str(f2), "label": "b"},
    ])
    monkeypatch.setenv("OBSERVA_FILES_JSON", payload)

    registry = _load_file_registry()

    assert len(registry) == 2
    assert str(f1.resolve()) in registry
    assert str(f2.resolve()) in registry


def test_read_file_registered_path_returns_content(tmp_path: Path) -> None:
    f = tmp_path / "x.log"
    f.write_text("hello world", encoding="utf-8")
    registry = _build_registry([{"type": "alertlog", "path": str(f), "label": "x"}])

    result = _read_file(registry, str(f))

    assert result["content"] == "hello world"
    assert result["offset"] == 0
    assert result["bytes_read"] == len("hello world")
    assert result["eof"] is True
    assert result["truncated"] is False
    assert result["total_size"] == len("hello world")


def test_read_file_unregistered_path_raises(tmp_path: Path) -> None:
    registered = tmp_path / "ok.log"
    registered.write_text("ok", encoding="utf-8")
    other = tmp_path / "secret.log"
    other.write_text("nope", encoding="utf-8")

    registry = _build_registry([{"type": "alertlog", "path": str(registered), "label": "ok"}])

    with pytest.raises(ValueError):
        _read_file(registry, str(other))


def test_read_file_offset_beyond_size_is_empty_eof(tmp_path: Path) -> None:
    f = tmp_path / "x.log"
    f.write_text("abc", encoding="utf-8")
    registry = _build_registry([{"type": "alertlog", "path": str(f), "label": "x"}])

    result = _read_file(registry, str(f), offset=100)

    assert result["content"] == ""
    assert result["bytes_read"] == 0
    assert result["eof"] is True
    assert result["truncated"] is False


def test_read_file_max_bytes_truncates(tmp_path: Path) -> None:
    payload = "A" * 100
    f = tmp_path / "big.log"
    f.write_text(payload, encoding="utf-8")
    registry = _build_registry([{"type": "alertlog", "path": str(f), "label": "big"}])

    result = _read_file(registry, str(f), offset=0, max_bytes=10)

    assert result["bytes_read"] == 10
    assert result["content"] == "A" * 10
    assert result["truncated"] is True
    assert result["eof"] is False
    assert result["total_size"] == 100


def test_read_file_caps_max_bytes_at_one_megabyte(tmp_path: Path) -> None:
    # Use a file slightly larger than 1 MB so the cap matters AND there's
    # more data left than the cap would allow (ensures truncated=True).
    size = MAX_READ_BYTES + 2048
    f = tmp_path / "huge.log"
    f.write_bytes(b"Z" * size)
    registry = _build_registry([{"type": "trace", "path": str(f), "label": "huge"}])

    result = _read_file(registry, str(f), offset=0, max_bytes=5_000_000)

    assert result["bytes_read"] == MAX_READ_BYTES
    assert result["truncated"] is True
    assert result["eof"] is False
    assert result["total_size"] == size


def _reg(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="")
    key = str(p.resolve())
    return {key: {"type": "alertlog", "path": key, "label": name}}, key


def test_search_file_returns_matches(tmp_path):
    registry, key = _reg(tmp_path, "a.log", "line one\nORA-00600 boom\n")
    out = _search_file(registry, key, r"ORA-\d+")
    assert out["match_count"] == 1
    assert out["matches"][0]["line"] == "ORA-00600 boom"


def test_search_file_rejects_unregistered(tmp_path):
    registry, _key = _reg(tmp_path, "a.log", "x\n")
    with pytest.raises(ValueError, match="not registered"):
        _search_file(registry, str(tmp_path / "other.log"), r"x")


def test_extract_ora_returns_incidents(tmp_path):
    registry, key = _reg(
        tmp_path, "alert.log",
        "2024-03-15T14:23:01\nORA-00060: deadlock\n",
    )
    out = _extract_ora(registry, key)
    assert out["incidents"][0]["code"] == "ORA-00060"


def test_extract_ora_rejects_unregistered(tmp_path):
    registry, _key = _reg(tmp_path, "alert.log", "x\n")
    with pytest.raises(ValueError, match="not registered"):
        _extract_ora(registry, str(tmp_path / "other.log"))


def test_search_file_clamps_max_matches(tmp_path):
    from observa.mcp.file_scan import SEARCH_MAX_MATCHES
    # 300 matching lines; an over-large max_matches must clamp to 200.
    registry, key = _reg(tmp_path, "big.log", "".join("ORA-00600\n" for _ in range(300)))
    out = _search_file(registry, key, r"ORA-", max_matches=100000)
    assert out["match_count"] == SEARCH_MAX_MATCHES  # 200, clamped
    assert out["truncated"] is True
