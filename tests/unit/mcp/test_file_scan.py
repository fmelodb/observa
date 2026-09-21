"""Unit tests for the pure file-scan engine."""
from __future__ import annotations

import pytest

from observa.mcp.file_scan import scan_matches


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="")
    return str(p)


def test_scan_matches_returns_line_and_byte_offset(tmp_path):
    # Two lines; the match is on line 2. Byte offset of line 2 == len(line1 bytes).
    path = _write(tmp_path, "a.log", "first line\nORA-00600 boom\n")
    out = scan_matches(path, r"ORA-\d+", max_matches=10, max_scan_bytes=1_000_000)
    assert out["match_count"] == 1
    m = out["matches"][0]
    assert m["line_number"] == 2
    assert m["byte_offset"] == len("first line\n".encode("utf-8"))
    assert m["line"] == "ORA-00600 boom"
    assert out["truncated"] is False
    assert out["total_size"] == len("first line\nORA-00600 boom\n".encode("utf-8"))


def test_scan_matches_caps_at_max_matches(tmp_path):
    path = _write(tmp_path, "b.log", "\n".join(f"ORA-0000{i}" for i in range(10)) + "\n")
    out = scan_matches(path, r"ORA-", max_matches=3, max_scan_bytes=1_000_000)
    assert out["match_count"] == 3
    assert out["truncated"] is True


def test_scan_matches_caps_at_scan_bytes(tmp_path):
    # 5 lines of 20 bytes each; cap scanning at 30 bytes => only first 2 lines seen.
    path = _write(tmp_path, "c.log", "".join(f"ORA-{i:015d}\n" for i in range(5)))
    out = scan_matches(path, r"ORA-", max_matches=99, max_scan_bytes=30)
    assert out["match_count"] == 2
    assert out["truncated"] is True


def test_scan_matches_truncates_long_lines(tmp_path):
    path = _write(tmp_path, "d.log", "ORA-1 " + "x" * 1000 + "\n")
    out = scan_matches(path, r"ORA-", max_matches=1, max_scan_bytes=1_000_000)
    assert len(out["matches"][0]["line"]) == 300


def test_scan_matches_invalid_regex(tmp_path):
    path = _write(tmp_path, "e.log", "anything\n")
    with pytest.raises(ValueError, match="invalid regex"):
        scan_matches(path, r"ORA-(", max_matches=1, max_scan_bytes=1000)


def test_scan_matches_no_match(tmp_path):
    path = _write(tmp_path, "f.log", "nothing here\n")
    out = scan_matches(path, r"ORA-\d+", max_matches=10, max_scan_bytes=1000)
    assert out["match_count"] == 0 and out["matches"] == [] and out["truncated"] is False


def test_scan_matches_truncated_at_exact_boundary(tmp_path):
    # 6-byte lines; budget=12 => lines at offsets 0 and 6 scanned, line at 12 dropped.
    path = _write(tmp_path, "g.log", "ORA-a\nORA-b\nORA-c\n")
    out = scan_matches(path, r"ORA-", max_matches=99, max_scan_bytes=12)
    assert out["match_count"] == 2
    assert out["truncated"] is True


def test_scan_matches_byte_offset_multibyte_utf8(tmp_path):
    p = tmp_path / "u.log"
    p.write_bytes("café\nORA-1\n".encode("utf-8"))
    out = scan_matches(str(p), r"ORA-", max_matches=10, max_scan_bytes=1_000_000)
    m = out["matches"][0]
    assert p.read_bytes()[m["byte_offset"]:].startswith(b"ORA-1")


def test_scan_matches_byte_offset_crlf(tmp_path):
    p = tmp_path / "crlf.log"
    p.write_bytes(b"ab\r\nORA-1\r\n")
    out = scan_matches(str(p), r"ORA-", max_matches=10, max_scan_bytes=1_000_000)
    m = out["matches"][0]
    assert p.read_bytes()[m["byte_offset"]:].startswith(b"ORA-1")
    assert m["line"] == "ORA-1"


from observa.mcp.file_scan import extract_ora_incidents


_ISO_LOG = (
    "2024-03-15T14:23:01.000000+00:00\n"
    "ORA-00060: deadlock detected while waiting for resource\n"
    "Errors in file /u01/diag/prod2_ora_1234.trc (incident=48291):\n"
    "2024-03-15T15:03:47.000000+00:00\n"
    "ORA-01555: snapshot too old\n"
)

_11G_LOG = (
    "Wed Mar 15 14:23:01 2024\n"
    "ORA-00600: internal error code, arguments: [kcbgtcr_5]\n"
)


def test_extract_ora_iso_format(tmp_path):
    path = _write(tmp_path, "alert_iso.log", _ISO_LOG)
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    incs = out["incidents"]
    codes = [i["code"] for i in incs]
    assert "ORA-00060" in codes and "ORA-01555" in codes
    deadlock = next(i for i in incs if i["code"] == "ORA-00060")
    assert deadlock["ts"] == "2024-03-15T14:23:01"
    # the incident= line ties to the same block timestamp
    incident = next(i for i in incs if i["incident_id"] == 48291)
    assert incident["ts"] == "2024-03-15T14:23:01"
    assert out["coverage"]["begin_ts"] == "2024-03-15T14:23:01"
    assert out["coverage"]["end_ts"] == "2024-03-15T15:03:47"


def test_extract_ora_11g_format(tmp_path):
    path = _write(tmp_path, "alert_11g.log", _11G_LOG)
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    inc = out["incidents"][0]
    assert inc["code"] == "ORA-00600"
    assert inc["ts"] == "2024-03-15T14:23:01"


def test_extract_ora_before_any_timestamp_has_none_ts(tmp_path):
    path = _write(tmp_path, "alert_nots.log", "ORA-04031: unable to allocate\n")
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    assert out["incidents"][0]["ts"] is None


def test_extract_ora_byte_offset_usable(tmp_path):
    path = _write(tmp_path, "alert_off.log", _ISO_LOG)
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    first = out["incidents"][0]
    raw = open(path, "rb").read()
    # byte_offset points at the ORA line start
    assert raw[first["byte_offset"]:].startswith(b"ORA-00060")


def test_extract_ora_empty_when_no_ora(tmp_path):
    path = _write(tmp_path, "alert_clean.log", "Wed Mar 15 14:23:01 2024\nStartup complete\n")
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    assert out["incidents"] == []
    assert out["coverage"]["begin_ts"] == "2024-03-15T14:23:01"


def test_extract_ora_unpadded_code(tmp_path):
    path = _write(tmp_path, "alert_unpadded.log",
                  "2024-03-15T14:23:01\nMMON reference to ORA-600 in trace\n")
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    assert out["incidents"][0]["code"] == "ORA-00600"  # normalized to 5 digits


def test_extract_ora_chained_codes_on_one_line(tmp_path):
    path = _write(tmp_path, "alert_chain.log",
                  "2024-03-15T14:23:01\nORA-00600 [x] ORA-07445 [y]\n")
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    codes = [i["code"] for i in out["incidents"]]
    assert codes == ["ORA-00600", "ORA-07445"]
    assert all(i["ts"] == "2024-03-15T14:23:01" for i in out["incidents"])


def test_extract_ora_negative_tz_offset(tmp_path):
    path = _write(tmp_path, "alert_tz.log",
                  "2024-03-15T14:23:01.123456-03:00\nORA-00060: deadlock\n")
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    assert out["incidents"][0]["ts"] == "2024-03-15T14:23:01"  # tz/fractional dropped


def test_extract_ora_output_is_json_safe(tmp_path):
    import json
    path = _write(tmp_path, "alert_json.log",
                  "2024-03-15T14:23:01\nORA-00600: boom (incident=42)\n")
    out = extract_ora_incidents(path, max_bytes=1_000_000)
    json.dumps(out)  # must not raise — pins the ISO-string invariant across the MCP boundary
