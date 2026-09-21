"""Evidence Ledger — provenance + shared result cache for every data access.

Owned by ``McpClient`` (one per case session). Every SQL query (agent or
internal), every file read, and every file search is recorded as an
``EvidenceRecord`` with a stable ID (``Q-001…`` for SQL, ``F-001…`` for file
reads, ``S-001…`` for file searches) that flows back to the LLM inside the
tool payload, so findings can cite evidence by ID.

The ledger doubles as a result cache: the AWR repository is immutable during
a case, so a hit returns the original record's payload (same ``evidence_id``,
``hits`` incremented — no duplicate record). Cache keys are namespaced by
access path: ``query_awr`` and ``query_catalog`` results never cross-serve,
because they run under different guard configs (row caps, predicate
enforcement). File searches are keyed by ``(pattern, path)``.

Recording never blocks a data access: any internal failure logs and returns
None instead of raising.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp

logger = logging.getLogger(__name__)

RecordKind = Literal["query_awr", "query_catalog", "read_file", "search_file"]
QueryKind = Literal["query_awr", "query_catalog"]
RecordStatus = Literal["ok", "error", "rejected"]


class EvidenceRecord(BaseModel):
    evidence_id: str
    kind: RecordKind
    requester: str
    turn: Optional[int] = None
    # SQL kinds only
    sql: str = ""
    canonical_sql: str = ""
    dbid: Optional[int] = None
    snap_range: Optional[tuple[int, int]] = None
    # File kind only
    path: str = ""
    offset: int = 0
    max_bytes: Optional[int] = None
    # Search kind only
    pattern: str = ""
    match_count: Optional[int] = None
    # Result — the full payload handed to the tool caller (minus evidence_id).
    result: dict = Field(default_factory=dict)
    row_count: Optional[int] = None
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    status: RecordStatus = "ok"
    error: str = ""
    hits: int = 0
    ts: float = 0.0


@dataclass
class RequesterCounters:
    queries: int = 0
    reads: int = 0
    cache_hits: int = 0

    @property
    def accesses(self) -> int:
        return self.queries + self.reads


def canonical_sql(sql: str) -> str:
    """Normalized cache key: sqlglot round-trip strips comments and normalizes
    identifier case/whitespace. Falls back to whitespace-collapsed raw text
    when the SQL does not parse — worse hit rate, never an error."""
    try:
        parsed = sqlglot.parse_one(sql, read="oracle")
        if parsed is not None:
            return parsed.sql(dialect="oracle", comments=False, normalize=True)
    except Exception:  # noqa: BLE001 — cache degrades, never raises
        pass
    return " ".join(sql.split())


def extract_case_predicates(sql: str) -> tuple[Optional[int], Optional[tuple[int, int]]]:
    """Best-effort (dbid, snap_range) from WHERE predicates, for display."""
    try:
        parsed = sqlglot.parse_one(sql, read="oracle")
    except Exception:  # noqa: BLE001
        return None, None
    if parsed is None:
        return None, None
    dbid: Optional[int] = None
    for node in parsed.find_all(exp.EQ):
        for col_side, other in ((node.left, node.expression), (node.expression, node.left)):
            if (
                isinstance(col_side, exp.Column)
                and col_side.name.upper() == "DBID"
                and isinstance(other, exp.Literal)
                and other.is_number
            ):
                dbid = int(float(other.name))
    snap: Optional[tuple[int, int]] = None
    for node in parsed.find_all(exp.Between):
        this = node.this
        if isinstance(this, exp.Column) and this.name.upper() == "SNAP_ID":
            low, high = node.args.get("low"), node.args.get("high")
            if (
                isinstance(low, exp.Literal) and low.is_number
                and isinstance(high, exp.Literal) and high.is_number
            ):
                snap = (int(float(low.name)), int(float(high.name)))
    if snap is None:
        for node in parsed.find_all(exp.In):
            this = node.this
            if isinstance(this, exp.Column) and this.name.upper() == "SNAP_ID":
                nums = [
                    int(float(lit.name))
                    for lit in (node.expressions or [])
                    if isinstance(lit, exp.Literal) and lit.is_number
                ]
                if nums:
                    snap = (min(nums), max(nums))
    return dbid, snap


class EvidenceLedger:
    """Append-only record list + canonical-key cache index + counters.

    Asyncio atomicity invariant: all methods are synchronous with no awaits,
    so each call is atomic between event-loop suspension points — do not add
    awaits inside.
    """

    def __init__(self) -> None:
        self._records: list[EvidenceRecord] = []
        self._by_id: dict[str, EvidenceRecord] = {}
        self._cache: dict[tuple, str] = {}  # canonical key -> evidence_id
        self._counters: dict[str, RequesterCounters] = {}
        self._q_seq = 0
        self._f_seq = 0
        self._s_seq = 0

    # -- introspection -------------------------------------------------

    @property
    def records(self) -> list[EvidenceRecord]:
        """Snapshot list; elements are live references — treat them as read-only."""
        return list(self._records)

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        return self._by_id.get(evidence_id)

    def known_ids(self) -> frozenset[str]:
        return frozenset(self._by_id)

    def counters(self) -> dict[str, RequesterCounters]:
        """Snapshot dict; values are live references — treat them as read-only."""
        return dict(self._counters)

    def totals(self) -> tuple[int, int]:
        """(total accesses, total cache hits) across all requesters."""
        accesses = sum(c.accesses for c in self._counters.values())
        hits = sum(c.cache_hits for c in self._counters.values())
        return accesses, hits

    # -- internals -----------------------------------------------------

    def _next_id(self, kind: RecordKind) -> str:
        if kind == "read_file":
            self._f_seq += 1
            return f"F-{self._f_seq:03d}"
        if kind == "search_file":
            self._s_seq += 1
            return f"S-{self._s_seq:03d}"
        self._q_seq += 1
        return f"Q-{self._q_seq:03d}"

    def _bump(self, requester: str, kind: RecordKind, *, hit: bool = False) -> None:
        c = self._counters.setdefault(requester, RequesterCounters())
        if kind in ("read_file", "search_file"):
            c.reads += 1
        else:
            c.queries += 1
        if hit:
            c.cache_hits += 1

    def _serve(self, evidence_id: str, requester: str, kind: RecordKind) -> dict:
        rec = self._by_id[evidence_id]
        rec.hits += 1
        self._bump(requester, kind, hit=True)
        logger.info("evidence cache hit: %s served to %s", evidence_id, requester)
        # Deep copy: a consumer mutating served rows must not poison the record.
        # evidence_id FIRST: survives tool-loop truncation of large payloads.
        return {"evidence_id": rec.evidence_id, **copy.deepcopy(rec.result)}

    # -- cache lookups (return payload dict incl. evidence_id, or None) --

    def lookup_query(self, kind: QueryKind, sql: str, requester: str) -> Optional[dict]:
        try:
            rid = self._cache.get((kind, canonical_sql(sql)))
            if rid is None:
                return None
            return self._serve(rid, requester, kind)
        except Exception:  # noqa: BLE001 — never block the access path
            logger.exception("evidence ledger: lookup_query failed (ignored)")
            return None

    def lookup_read(
        self, path: str, offset: int, max_bytes: Optional[int], requester: str
    ) -> Optional[dict]:
        try:
            rid = self._cache.get(("read_file", path, offset, max_bytes))
            if rid is None:
                return None
            return self._serve(rid, requester, "read_file")
        except Exception:  # noqa: BLE001
            logger.exception("evidence ledger: lookup_read failed (ignored)")
            return None

    def lookup_search(
        self, pattern: str, path: str, requester: str
    ) -> Optional[dict]:
        try:
            rid = self._cache.get(("search_file", pattern, path))
            if rid is None:
                return None
            return self._serve(rid, requester, "search_file")
        except Exception:  # noqa: BLE001
            logger.exception("evidence ledger: lookup_search failed (ignored)")
            return None

    # -- recording (return evidence_id, or None when recording failed) --

    def record_query(
        self,
        kind: QueryKind,
        sql: str,
        requester: str,
        *,
        turn: Optional[int] = None,
        result: Optional[dict] = None,
        duration_ms: int = 0,
        status: RecordStatus = "ok",
        error: str = "",
    ) -> Optional[str]:
        try:
            dbid, snap = extract_case_predicates(sql)
            canon = canonical_sql(sql)
            res = copy.deepcopy(result) if result else {}
            rec = EvidenceRecord(
                evidence_id=self._next_id(kind),
                kind=kind,
                requester=requester,
                turn=turn,
                sql=sql,
                canonical_sql=canon,
                dbid=dbid,
                snap_range=snap,
                result=res,
                row_count=res.get("row_count"),
                truncated=bool(res.get("truncated")),
                warnings=list(res.get("warnings") or []),
                duration_ms=duration_ms,
                status=status,
                error=error,
                ts=time.time(),
            )
            self._records.append(rec)
            self._by_id[rec.evidence_id] = rec
            if status == "ok":
                self._cache[(kind, canon)] = rec.evidence_id
            self._bump(requester, kind)
            return rec.evidence_id
        except Exception:  # noqa: BLE001
            logger.exception("evidence ledger: record_query failed (ignored)")
            return None

    def record_read(
        self,
        path: str,
        offset: int,
        max_bytes: Optional[int],
        requester: str,
        *,
        turn: Optional[int] = None,
        result: Optional[dict] = None,
        duration_ms: int = 0,
        status: RecordStatus = "ok",
        error: str = "",
    ) -> Optional[str]:
        try:
            res = copy.deepcopy(result) if result else {}
            rec = EvidenceRecord(
                evidence_id=self._next_id("read_file"),
                kind="read_file",
                requester=requester,
                turn=turn,
                path=path,
                offset=offset,
                max_bytes=max_bytes,
                result=res,
                truncated=bool(res.get("truncated")),
                duration_ms=duration_ms,
                status=status,
                error=error,
                ts=time.time(),
            )
            self._records.append(rec)
            self._by_id[rec.evidence_id] = rec
            if status == "ok":
                self._cache[("read_file", path, offset, max_bytes)] = rec.evidence_id
            self._bump(requester, "read_file")
            return rec.evidence_id
        except Exception:  # noqa: BLE001
            logger.exception("evidence ledger: record_read failed (ignored)")
            return None

    def record_search(
        self,
        path: str,
        pattern: str,
        requester: str,
        *,
        turn: Optional[int] = None,
        result: Optional[dict] = None,
        duration_ms: int = 0,
        status: RecordStatus = "ok",
        error: str = "",
    ) -> Optional[str]:
        try:
            res = copy.deepcopy(result) if result else {}
            rec = EvidenceRecord(
                evidence_id=self._next_id("search_file"),
                kind="search_file",
                requester=requester,
                turn=turn,
                path=path,
                pattern=pattern,
                result=res,
                match_count=res.get("match_count"),
                truncated=bool(res.get("truncated")),
                duration_ms=duration_ms,
                status=status,
                error=error,
                ts=time.time(),
            )
            self._records.append(rec)
            self._by_id[rec.evidence_id] = rec
            if status == "ok":
                self._cache[("search_file", pattern, path)] = rec.evidence_id
            self._bump(requester, "search_file")
            return rec.evidence_id
        except Exception:  # noqa: BLE001
            logger.exception("evidence ledger: record_search failed (ignored)")
            return None
