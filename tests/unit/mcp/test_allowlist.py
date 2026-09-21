"""Unit tests for the SQL allowlist validator."""

from __future__ import annotations

import pytest

from observa.mcp.allowlist import (
    AllowlistConfig,
    AllowlistError,
    validate_sql,
)


# --- Accepted queries -------------------------------------------------------


def test_select_from_dba_hist_snapshot_passes() -> None:
    validate_sql("SELECT * FROM DBA_HIST_SNAPSHOT")


def test_case_insensitive_prefix_with_bind_passes() -> None:
    validate_sql("SELECT * FROM dba_hist_sqlstat WHERE sql_id = :id")


def test_v_instance_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("SELECT * FROM V$INSTANCE")


def test_schema_qualified_dba_hist_passes() -> None:
    validate_sql("SELECT * FROM SYS.DBA_HIST_SEG_STAT")


def test_cte_passes() -> None:
    validate_sql(
        "WITH a AS (SELECT 1 AS n FROM DBA_HIST_SNAPSHOT) SELECT * FROM a"
    )


def test_subquery_passes() -> None:
    validate_sql("SELECT * FROM (SELECT * FROM DBA_HIST_SNAPSHOT) t")


def test_allowed_join_passes() -> None:
    validate_sql(
        "SELECT * FROM DBA_HIST_SNAPSHOT s "
        "JOIN DBA_HIST_SQLSTAT q ON s.snap_id = q.snap_id"
    )


def test_v_database_and_v_version_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("SELECT * FROM V$DATABASE")
    with pytest.raises(AllowlistError):
        validate_sql("SELECT * FROM V$VERSION")


def test_gv_instance_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("SELECT * FROM GV$INSTANCE")


# --- Rejected queries -------------------------------------------------------


def test_non_allowlisted_table_rejected() -> None:
    with pytest.raises(AllowlistError, match="DBA_USERS"):
        validate_sql("SELECT * FROM DBA_USERS")


def test_ddl_drop_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("DROP TABLE X")


def test_dml_insert_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("INSERT INTO X VALUES (1)")


def test_dml_update_on_allowed_table_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("UPDATE DBA_HIST_SNAPSHOT SET snap_id = 1")


def test_plsql_block_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("BEGIN NULL; END;")


def test_for_update_rejected() -> None:
    with pytest.raises(AllowlistError, match="FOR UPDATE"):
        validate_sql("SELECT * FROM DBA_HIST_SNAPSHOT FOR UPDATE")


def test_multiple_statements_rejected() -> None:
    with pytest.raises(AllowlistError, match="Multiple"):
        validate_sql("SELECT * FROM DBA_HIST_SNAPSHOT; DROP TABLE X")


def test_join_with_disallowed_table_rejected() -> None:
    with pytest.raises(AllowlistError, match="DBA_USERS"):
        validate_sql(
            "SELECT * FROM DBA_HIST_SNAPSHOT JOIN DBA_USERS USING (ID)"
        )


def test_empty_sql_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("")


def test_select_from_dual_rejected() -> None:
    # DUAL is not on the allowlist; also ensures ``SELECT 1 FROM DUAL`` probes
    # never leak through.
    with pytest.raises(AllowlistError):
        validate_sql("SELECT 1 FROM DUAL")


def test_select_without_from_rejected() -> None:
    with pytest.raises(AllowlistError, match="no tables"):
        validate_sql("SELECT 1")


def test_truncate_rejected() -> None:
    with pytest.raises(AllowlistError):
        validate_sql("TRUNCATE TABLE DBA_HIST_SNAPSHOT")


def test_select_into_plsql_form_rejected() -> None:
    # Oracle PL/SQL SELECT ... INTO local variable — forbidden even if table
    # is allowlisted, because it implies execution inside a PL/SQL context.
    with pytest.raises(AllowlistError):
        validate_sql("SELECT snap_id INTO v FROM DBA_HIST_SNAPSHOT")


# --- Config plumbing --------------------------------------------------------


def test_config_row_limit_default() -> None:
    cfg = AllowlistConfig()
    assert cfg.row_limit == 500
    assert "DBA_HIST_" in cfg.allowed_prefixes
    # V$ / GV$ were removed — we are querying an AWR repository, not the
    # problem instance.
    assert not any(v.startswith(("V$", "GV$")) for v in cfg.allowed_exact)


def test_custom_config_honored() -> None:
    cfg = AllowlistConfig(
        allowed_prefixes=("FOO_",),
        allowed_exact=frozenset({"BAR"}),
        row_limit=10,
    )
    validate_sql("SELECT * FROM FOO_X", cfg)
    validate_sql("SELECT * FROM BAR", cfg)
    with pytest.raises(AllowlistError):
        validate_sql("SELECT * FROM DBA_HIST_SNAPSHOT", cfg)


def test_multi_statement_split_ignores_semicolons_in_strings() -> None:
    # Semicolon inside a literal should NOT be treated as a separator.
    validate_sql(
        "SELECT * FROM DBA_HIST_SQLSTAT WHERE sql_text = 'a;b'"
    )
