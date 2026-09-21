"""SQL allowlist validator for MCP-mediated Oracle queries.

Pure module (zero I/O). Parses incoming SQL with sqlglot and enforces a
read-only allowlist restricted to ``DBA_HIST_*`` views.

The target database is an AWR **repository** — AWR data from a remote
production instance has been imported here. V$ / GV$ views reflect the LOCAL
repository instance, not the problem database, so they are forbidden.
Every query must filter by the case's DBID. Since Feature 1,
``check_case_predicates`` enforces this deterministically once a case is
active (``case_dbid > 0``): missing/wrong dbid predicate → reject; missing
snap-window restriction → warn/reject/off per config.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

ALLOWED_PREFIXES: tuple[str, ...] = ("DBA_HIST_",)
ALLOWED_EXACT: frozenset[str] = frozenset(
    {
        # Purely AWR-resident reference tables used by some specialists.
        "DBA_FEATURE_USAGE_STATISTICS",
        "DBA_REGISTRY",
        # Metadata catalog — agents use it to discover real column names on
        # DBA_HIST_* views before writing queries, instead of guessing and
        # producing ORA-00904 errors.
        "DBA_TAB_COLUMNS",
    }
)


class AllowlistError(ValueError):
    """Raised when a SQL statement violates the read-only allowlist."""


@dataclass(frozen=True)
class AllowlistConfig:
    allowed_prefixes: tuple[str, ...] = ALLOWED_PREFIXES
    allowed_exact: frozenset[str] = ALLOWED_EXACT
    row_limit: int = 500
    # Case-predicate enforcement (Feature 1). Inactive while case_dbid == 0 —
    # the intake picker and diff-probe internal queries run before a case
    # exists and legitimately scan all dbids.
    case_dbid: int = 0
    enforce_window: str = "warn"  # warn | reject | off


# Expression types that represent data-modification / non-SELECT operations.
# sqlglot parses DML/DDL/PLSQL into distinct Expression subclasses; we match
# against a tuple of classes rather than the parsed name to be robust.
_FORBIDDEN_STATEMENT_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,  # Generic catch-all for unparsed statements (e.g. BEGIN..END)
)


def _split_statements(sql: str) -> list[str]:
    """Split on ``;`` while ignoring semicolons inside strings/comments."""
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_single:
            buf.append(ch)
            if ch == "'":
                # Handle doubled '' as escaped quote.
                if nxt == "'":
                    buf.append(nxt)
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if in_double:
            buf.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _table_name_allowed(name: str, config: AllowlistConfig) -> bool:
    upper = name.upper()
    if upper in config.allowed_exact:
        return True
    return any(upper.startswith(p) for p in config.allowed_prefixes)


def validate_sql(sql: str, config: AllowlistConfig | None = None) -> None:
    """Validate SQL. Raises :class:`AllowlistError` on any violation.

    Rules:
      - Must be a single SELECT / WITH statement. No DDL/DML/PLSQL blocks.
      - Every table/view reference must be in the allowlist (prefix or exact).
      - No ``FOR UPDATE``, no ``SELECT ... INTO`` local-variable form,
        no hints that write.
      - Rejects multiple statements (split by ``;`` ignoring strings/comments).
    """
    cfg = config or AllowlistConfig()

    if sql is None or not sql.strip():
        raise AllowlistError("Empty SQL statement.")

    statements = _split_statements(sql)
    if len(statements) > 1:
        raise AllowlistError(
            f"Multiple statements are not allowed (found {len(statements)})."
        )
    if not statements:
        raise AllowlistError("Empty SQL statement.")

    single = statements[0]

    try:
        parsed = sqlglot.parse_one(single, read="oracle")
    except sqlglot.errors.ParseError as err:
        raise AllowlistError(f"Could not parse SQL: {err}") from err

    if parsed is None:
        raise AllowlistError("Could not parse SQL.")

    # Reject PLSQL / DDL / DML / anonymous blocks up front.
    if isinstance(parsed, _FORBIDDEN_STATEMENT_TYPES):
        raise AllowlistError(
            f"Statement type '{type(parsed).__name__}' is not permitted; "
            "only SELECT / WITH queries are allowed."
        )

    # Top-level must be a query (Select, Subquery, Union, With-wrapped Select).
    if not isinstance(parsed, (exp.Select, exp.Subquery, exp.Union, exp.Query)):
        raise AllowlistError(
            f"Only SELECT / WITH queries are allowed (got {type(parsed).__name__})."
        )

    # Defense in depth: any embedded DML/DDL node anywhere in the tree is fatal.
    for node in parsed.walk():
        if isinstance(node, _FORBIDDEN_STATEMENT_TYPES):
            raise AllowlistError(
                f"Embedded '{type(node).__name__}' statement is not permitted."
            )

    # FOR UPDATE → Lock node in sqlglot.
    for lock in parsed.find_all(exp.Lock):
        raise AllowlistError("FOR UPDATE / row-lock clauses are not permitted.")

    # SELECT ... INTO <local var> form. sqlglot parses Oracle PL/SQL
    # ``SELECT col INTO v FROM t`` with an ``into`` arg on the Select node.
    for select in parsed.find_all(exp.Select):
        if select.args.get("into") is not None:
            raise AllowlistError(
                "SELECT ... INTO (local-variable form) is not permitted."
            )

    # Collect CTE aliases to skip them when validating Table nodes.
    cte_aliases: set[str] = set()
    with_node = parsed.args.get("with") if hasattr(parsed, "args") else None
    if with_node is not None:
        for cte in with_node.expressions:
            alias = cte.alias
            if alias:
                cte_aliases.add(alias.upper())
    # Also check nested CTEs.
    for w in parsed.find_all(exp.With):
        for cte in w.expressions:
            alias = cte.alias
            if alias:
                cte_aliases.add(alias.upper())

    # Validate every table reference.
    referenced: list[str] = []
    for table in parsed.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        if name.upper() in cte_aliases:
            continue
        referenced.append(name)
        if not _table_name_allowed(name, cfg):
            raise AllowlistError(
                f"Table/view '{name}' is not in the allowlist. "
                f"Allowed prefixes: {cfg.allowed_prefixes}; "
                f"allowed exact: {sorted(cfg.allowed_exact)}."
            )

    if not referenced:
        # Queries like ``SELECT 1 FROM DUAL`` are not useful for agents and
        # could hide problems; require at least one explicit allowed table.
        # Exception: pure scalar ``SELECT 1`` with no FROM is harmless but
        # also useless — reject to keep the contract tight.
        raise AllowlistError(
            "Query references no tables; at least one allowlisted view is required."
        )


# Views exempt from the snap-window check: they carry no snap_id semantics.
_WINDOW_EXEMPT_VIEWS: frozenset[str] = frozenset({"DBA_HIST_DATABASE_INSTANCE"})
# Column names that count as a time/window restriction.
_WINDOW_COLUMNS: frozenset[str] = frozenset(
    {"SNAP_ID", "BEGIN_INTERVAL_TIME", "END_INTERVAL_TIME"}
)


def _referenced_hist_tables(parsed: exp.Expression) -> list[str]:
    """Uppercase DBA_HIST_* tables referenced (CTE aliases excluded)."""
    cte_aliases: set[str] = set()
    for w in parsed.find_all(exp.With):
        for cte in w.expressions:
            if cte.alias:
                cte_aliases.add(cte.alias.upper())
    out: list[str] = []
    for table in parsed.find_all(exp.Table):
        name = (table.name or "").upper()
        if name and name not in cte_aliases and name.startswith("DBA_HIST_"):
            out.append(name)
    return out


def check_case_predicates(sql: str, config: AllowlistConfig) -> list[str]:
    """Enforce the case's dbid predicate and (softly) the snap-window predicate.

    Returns a list of warning strings. Raises :class:`AllowlistError` when the
    dbid predicate is missing/wrong, or when the window predicate is missing
    and ``enforce_window == "reject"``. Call AFTER ``validate_sql`` — this
    function assumes the SQL already parsed cleanly.
    """
    if config.case_dbid <= 0:
        return []
    parsed = sqlglot.parse_one(sql, read="oracle")
    if parsed is None:
        return []
    hist_tables = _referenced_hist_tables(parsed)
    if not hist_tables:
        return []

    # --- dbid: an equality (or IN) predicate on a column named dbid.
    dbid_found = False
    for node in parsed.find_all(exp.EQ):
        left, right = node.left, node.expression
        for col_side, other in ((left, right), (right, left)):
            if isinstance(col_side, exp.Column) and col_side.name.upper() == "DBID":
                dbid_found = True
                if isinstance(other, exp.Literal) and other.is_number:
                    if int(float(other.name)) != config.case_dbid:
                        raise AllowlistError(
                            f"dbid literal {other.name} does not match the case dbid "
                            f"{config.case_dbid}. Query THIS case's database only."
                        )
    if not dbid_found:
        for node in parsed.find_all(exp.In):
            this = node.this
            if isinstance(this, exp.Column) and this.name.upper() == "DBID":
                dbid_found = True
                for lit in node.expressions or []:
                    if isinstance(lit, exp.Literal) and lit.is_number:
                        if int(float(lit.name)) != config.case_dbid:
                            raise AllowlistError(
                                f"dbid literal {lit.name} does not match the case dbid "
                                f"{config.case_dbid}. Query THIS case's database only."
                            )
    if not dbid_found:
        raise AllowlistError(
            f"Query references {sorted(set(hist_tables))} but has no "
            f"`dbid = {config.case_dbid}` predicate. The repository holds multiple "
            "databases — every DBA_HIST_* query MUST filter by the case dbid."
        )

    # --- snap window: any reference to snap_id / *_interval_time counts.
    if config.enforce_window == "off":
        return []
    window_relevant = [t for t in hist_tables if t not in _WINDOW_EXEMPT_VIEWS]
    if not window_relevant:
        return []
    for col in parsed.find_all(exp.Column):
        if col.name.upper() in _WINDOW_COLUMNS:
            return []
    msg = (
        "Query has no snap_id / interval-time restriction — it scans the entire "
        "history for this dbid. Restrict to the problem window (and baseline "
        "window when comparing)."
    )
    if config.enforce_window == "reject":
        raise AllowlistError(msg)
    return [msg]
