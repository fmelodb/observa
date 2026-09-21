import pytest

from observa.mcp.allowlist import AllowlistConfig, AllowlistError, check_case_predicates

CFG = AllowlistConfig(case_dbid=111, enforce_window="warn")


def test_inactive_when_no_case_dbid():
    cfg = AllowlistConfig()  # case_dbid=0 → picker/probe phase, no enforcement
    assert check_case_predicates("SELECT dbid FROM DBA_HIST_SNAPSHOT", cfg) == []


def test_missing_dbid_predicate_raises():
    with pytest.raises(AllowlistError, match="dbid"):
        check_case_predicates(
            "SELECT snap_id FROM DBA_HIST_SNAPSHOT WHERE snap_id > 10", CFG
        )


def test_wrong_dbid_literal_raises():
    with pytest.raises(AllowlistError, match="case dbid"):
        check_case_predicates(
            "SELECT snap_id FROM DBA_HIST_SNAPSHOT WHERE dbid = 999 AND snap_id > 1", CFG
        )


def test_correct_dbid_and_snap_window_passes_clean():
    warnings = check_case_predicates(
        "SELECT snap_id FROM DBA_HIST_SYSSTAT "
        "WHERE dbid = 111 AND snap_id BETWEEN 100 AND 108", CFG
    )
    assert warnings == []


def test_missing_snap_restriction_warns():
    warnings = check_case_predicates(
        "SELECT stat_name FROM DBA_HIST_SYSSTAT WHERE dbid = 111", CFG
    )
    assert len(warnings) == 1 and "snap" in warnings[0].lower()


def test_missing_snap_restriction_rejects_when_configured():
    cfg = AllowlistConfig(case_dbid=111, enforce_window="reject")
    with pytest.raises(AllowlistError, match="snap"):
        check_case_predicates("SELECT stat_name FROM DBA_HIST_SYSSTAT WHERE dbid = 111", cfg)


def test_exempt_views_skip_all_checks():
    # DBA_TAB_COLUMNS has no dbid; DBA_HIST_DATABASE_INSTANCE has no snap_id.
    assert check_case_predicates(
        "SELECT column_name FROM DBA_TAB_COLUMNS WHERE table_name = 'X'", CFG
    ) == []
    assert check_case_predicates(
        "SELECT db_name FROM DBA_HIST_DATABASE_INSTANCE WHERE dbid = 111", CFG
    ) == []


def test_time_filter_via_interval_column_counts_as_window():
    warnings = check_case_predicates(
        "SELECT s.snap_id FROM DBA_HIST_SNAPSHOT s "
        "WHERE s.dbid = 111 AND s.begin_interval_time > SYSDATE - 1", CFG
    )
    assert warnings == []


def test_wrong_dbid_in_list_raises():
    with pytest.raises(AllowlistError, match="case dbid"):
        check_case_predicates(
            "SELECT snap_id FROM DBA_HIST_SNAPSHOT WHERE dbid IN (999) AND snap_id > 1", CFG
        )


def test_dbid_join_without_literal_is_accepted_characterization():
    # Column-to-column dbid join with no literal binds nothing to the case —
    # accepted deliberately: rejecting would false-positive legit multi-view
    # joins, and false positives cost more than this exotic false negative.
    warnings = check_case_predicates(
        "SELECT s.sql_id FROM DBA_HIST_SQLSTAT s JOIN DBA_HIST_SQLTEXT t "
        "ON s.dbid = t.dbid AND s.sql_id = t.sql_id "
        "WHERE s.snap_id BETWEEN 1 AND 2", CFG
    )
    assert warnings == []
