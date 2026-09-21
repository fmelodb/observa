"""TFA/SQLHC zip + directory expansion."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from observa.mcp.file_expansion import (
    EXPANDABLE_TYPES,
    EXTENSIONS_BY_TYPE,
    MAX_FILES_PER_EXPANSION,
    FileEntry,
    expand_input_files,
)


@pytest.fixture
def tmp_extract(tmp_path: Path) -> Path:
    d = tmp_path / "extract"
    d.mkdir()
    return d


def _entry(type_: str, path: Path, label: str = "") -> FileEntry:
    return FileEntry(type=type_, path=str(path), label=label)


# ---------------------------------------------------------------------------
# Pass-through behaviors
# ---------------------------------------------------------------------------

def test_non_expandable_type_passes_through(tmp_path: Path, tmp_extract: Path):
    f = tmp_path / "alert.log"
    f.write_text("hello")
    out = expand_input_files([_entry("alertlog", f)], tmp_extract)
    assert len(out) == 1
    assert out[0].type == "alertlog"
    assert out[0].path == str(f)


def test_single_file_of_expandable_type_passes_through(tmp_path: Path, tmp_extract: Path):
    f = tmp_path / "report.html"
    f.write_text("<html/>")
    out = expand_input_files([_entry("sqlhc", f)], tmp_extract)
    assert len(out) == 1
    assert out[0].path == str(f)


def test_missing_path_is_dropped_with_warning(tmp_path: Path, tmp_extract: Path, caplog):
    out = expand_input_files(
        [_entry("tfa", tmp_path / "does-not-exist.zip")],
        tmp_extract,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Directory expansion
# ---------------------------------------------------------------------------

def test_directory_expanded_with_extension_filter(tmp_path: Path, tmp_extract: Path):
    bundle = tmp_path / "tfa_bundle"
    bundle.mkdir()
    (bundle / "alert.log").write_text("a")
    (bundle / "trace.trc").write_text("b")
    (bundle / "binary.bin").write_bytes(b"\x00\x01")  # excluded
    (bundle / "nested").mkdir()
    (bundle / "nested" / "deep.txt").write_text("c")

    out = expand_input_files([_entry("tfa", bundle, label="bundle")], tmp_extract)
    paths = sorted(e.path for e in out)
    names = [Path(p).name for p in paths]
    assert "alert.log" in names
    assert "trace.trc" in names
    assert "deep.txt" in names
    assert "binary.bin" not in names
    assert all(e.type == "tfa" for e in out)


def test_directory_label_carries_relative_path(tmp_path: Path, tmp_extract: Path):
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "x.log").write_text("hi")
    out = expand_input_files([_entry("tfa", bundle, label="bundle")], tmp_extract)
    assert out[0].label.startswith("bundle:")
    assert out[0].label.endswith("x.log")


def test_directory_with_no_matches_drops_entry(tmp_path: Path, tmp_extract: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "binary.bin").write_bytes(b"\x00")
    out = expand_input_files([_entry("tfa", empty)], tmp_extract)
    assert out == []


def test_expansion_cap_enforced(tmp_path: Path, tmp_extract: Path):
    bundle = tmp_path / "huge"
    bundle.mkdir()
    for i in range(MAX_FILES_PER_EXPANSION + 50):
        (bundle / f"f{i:04d}.log").write_text("x")
    out = expand_input_files([_entry("tfa", bundle)], tmp_extract)
    assert len(out) == MAX_FILES_PER_EXPANSION


# ---------------------------------------------------------------------------
# ZIP expansion
# ---------------------------------------------------------------------------

def _make_zip(zip_path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return zip_path


def test_zip_extracted_and_walked(tmp_path: Path, tmp_extract: Path):
    z = _make_zip(tmp_path / "coll.zip", {
        "alert.log": b"a",
        "trace/abc.trc": b"b",
        "binary.bin": b"\x00",
    })
    out = expand_input_files([_entry("tfa", z, label="coll.zip")], tmp_extract)
    names = sorted(Path(e.path).name for e in out)
    assert "alert.log" in names
    assert "abc.trc" in names
    assert "binary.bin" not in names


def test_zip_extraction_is_sandboxed_under_extract_root(tmp_path: Path, tmp_extract: Path):
    z = _make_zip(tmp_path / "c.zip", {"file.log": b"x"})
    out = expand_input_files([_entry("tfa", z)], tmp_extract)
    for entry in out:
        # Every extracted file lives under tmp_extract.
        assert str(Path(entry.path).resolve()).startswith(str(tmp_extract.resolve()))


def test_zip_path_traversal_rejected(tmp_path: Path, tmp_extract: Path):
    """A zip member that tries to escape the extract dir is refused."""
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escape.log", b"x")
    out = expand_input_files([_entry("tfa", z)], tmp_extract)
    # Extraction failed → entry dropped silently (warning logged).
    assert out == []


def test_zip_re_extraction_is_idempotent(tmp_path: Path, tmp_extract: Path):
    z = _make_zip(tmp_path / "x.zip", {"a.log": b"1"})
    out1 = expand_input_files([_entry("tfa", z)], tmp_extract)
    out2 = expand_input_files([_entry("tfa", z)], tmp_extract)
    assert [e.path for e in out1] == [e.path for e in out2]


# ---------------------------------------------------------------------------
# Per-type allowlist
# ---------------------------------------------------------------------------

def test_sqlhc_allows_html(tmp_path: Path, tmp_extract: Path):
    bundle = tmp_path / "s"
    bundle.mkdir()
    (bundle / "report.html").write_text("<html/>")
    (bundle / "junk.exe").write_bytes(b"MZ")
    out = expand_input_files([_entry("sqlhc", bundle)], tmp_extract)
    names = [Path(e.path).name for e in out]
    assert names == ["report.html"]


def test_only_tfa_and_sqlhc_are_expandable():
    assert EXPANDABLE_TYPES == frozenset({"tfa", "sqlhc"})
    assert "tfa" in EXTENSIONS_BY_TYPE
    assert "sqlhc" in EXTENSIONS_BY_TYPE
    assert "alertlog" not in EXTENSIONS_BY_TYPE


# ---------------------------------------------------------------------------
# Mixed input
# ---------------------------------------------------------------------------

def test_mixed_inputs_each_handled_independently(tmp_path: Path, tmp_extract: Path):
    # alertlog passes through, tfa zip expands, sqlhc directory expands.
    log = tmp_path / "alert.log"
    log.write_text("a")
    z = _make_zip(tmp_path / "t.zip", {"x.log": b"x", "y.trc": b"y"})
    sd = tmp_path / "sqlhc_dir"
    sd.mkdir()
    (sd / "report.html").write_text("h")

    out = expand_input_files(
        [
            _entry("alertlog", log),
            _entry("tfa", z),
            _entry("sqlhc", sd),
        ],
        tmp_extract,
    )
    by_type: dict[str, list[str]] = {}
    for e in out:
        by_type.setdefault(e.type, []).append(Path(e.path).name)
    assert by_type["alertlog"] == ["alert.log"]
    assert sorted(by_type["tfa"]) == ["x.log", "y.trc"]
    assert by_type["sqlhc"] == ["report.html"]
