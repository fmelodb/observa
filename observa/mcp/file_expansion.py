"""Expand TFA/SQLHC inputs that point to a directory or zip into individual files.

The user can register a TFA bundle or SQLHC report by pointing at:

* a single file (passes through unchanged)
* a directory (walked recursively, filtered by extension allowlist)
* a ``.zip`` archive (extracted under a temp dir, then walked the same way)

Other file types (alertlog, trace, hanganalyze, ddl, awr_report) are passed
through unchanged — those are always single files.

Pure functions only. The caller (McpClient) owns temp-dir lifetime.
"""
from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Types whose path may be a zip or a directory of many files.
EXPANDABLE_TYPES: frozenset[str] = frozenset({"tfa", "sqlhc"})

# Per-type extension allowlist (lower-case, with leading dot). Used when
# walking expanded directories to skip binaries / unrelated files.
EXTENSIONS_BY_TYPE: dict[str, frozenset[str]] = {
    "tfa": frozenset({".log", ".trc", ".txt", ".html", ".htm", ".out", ".csv"}),
    "sqlhc": frozenset({".html", ".htm", ".txt", ".log", ".sql"}),
}

# Cap: never register more than N files from a single user input. Protects
# against a runaway TFA bundle with thousands of trace files.
MAX_FILES_PER_EXPANSION = 200


@dataclass(frozen=True)
class FileEntry:
    """Minimal duck-type for ``McpClient.InputFileEntry`` — kept here so this
    module has zero dependency on ``observa.mcp.client``."""
    type: str
    path: str
    label: str = ""


def _label_for(root_label: str, relative: Path) -> str:
    """Build a human-readable label for an expanded entry."""
    rel = str(relative).replace("\\", "/")
    return f"{root_label}:{rel}" if root_label else rel


def _walk_directory(
    root: Path,
    file_type: str,
    root_label: str,
) -> list[FileEntry]:
    """Walk ``root`` recursively, filtering by ``EXTENSIONS_BY_TYPE[file_type]``."""
    allowed = EXTENSIONS_BY_TYPE.get(file_type, frozenset())
    entries: list[FileEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        rel = path.relative_to(root)
        entries.append(
            FileEntry(
                type=file_type,
                path=str(path.resolve()),
                label=_label_for(root_label, rel),
            )
        )
        if len(entries) >= MAX_FILES_PER_EXPANSION:
            logger.warning(
                "expansion of %s capped at %d files (more were available)",
                root, MAX_FILES_PER_EXPANSION,
            )
            break
    return entries


def _extract_zip(zip_path: Path, extract_root: Path) -> Path:
    """Extract ``zip_path`` into a stable subdirectory of ``extract_root``.

    Returns the directory containing the extracted contents. Subsequent
    re-extractions of the same zip into the same root are idempotent — if
    the target dir already exists, it is reused.
    """
    target = extract_root / f"{zip_path.stem}__{abs(hash(str(zip_path))) % 0xFFFF:04x}"
    if target.exists():
        return target
    target.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Reject path-traversal attempts. zipfile in 3.12 already prevents
            # most of these but be explicit.
            for member in zf.namelist():
                norm = Path(member).as_posix()
                if norm.startswith("/") or ".." in Path(norm).parts:
                    raise ValueError(f"zip member rejected (path traversal): {member}")
            zf.extractall(target)
    except Exception:
        # Best-effort cleanup so a partial extraction doesn't poison reuse.
        import shutil
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def expand_input_files(
    inputs: Iterable[FileEntry],
    extract_root: Path,
) -> list[FileEntry]:
    """Expand any TFA/SQLHC entries that point to zip/directory.

    ``extract_root`` must exist and be writable; zip contents are extracted
    into subdirectories beneath it.

    Behavior per entry:

    * type not in ``EXPANDABLE_TYPES`` → passes through unchanged.
    * path is a regular file → passes through unchanged (no extension
      filter; the user explicitly named this file).
    * path is a directory → walked, filtered by extension, capped.
    * path ends in ``.zip`` → extracted then walked.
    * path doesn't exist or expansion fails → entry dropped, warning logged.
    """
    out: list[FileEntry] = []
    for entry in inputs:
        if entry.type not in EXPANDABLE_TYPES:
            out.append(entry)
            continue

        src = Path(entry.path)
        if not src.exists():
            logger.warning("input file not found, dropping: %s", entry.path)
            continue

        root_label = entry.label or src.name

        if src.is_file() and src.suffix.lower() != ".zip":
            # Single file of an expandable type — keep as-is.
            out.append(entry)
            continue

        if src.is_file() and src.suffix.lower() == ".zip":
            try:
                extracted = _extract_zip(src, extract_root)
            except Exception as exc:  # noqa: BLE001
                logger.warning("zip extraction failed for %s: %s", entry.path, exc)
                continue
            expanded = _walk_directory(extracted, entry.type, root_label)
        elif src.is_dir():
            expanded = _walk_directory(src, entry.type, root_label)
        else:
            logger.warning("input path is neither file nor directory: %s", entry.path)
            continue

        if not expanded:
            logger.warning(
                "no files matched extension filter for %s under %s",
                entry.type, entry.path,
            )
            continue
        logger.info(
            "expanded %s '%s' into %d file(s)", entry.type, entry.path, len(expanded),
        )
        out.extend(expanded)
    return out


__all__ = [
    "EXPANDABLE_TYPES",
    "EXTENSIONS_BY_TYPE",
    "MAX_FILES_PER_EXPANSION",
    "FileEntry",
    "expand_input_files",
]
