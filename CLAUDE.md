# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Commands

```bash
# Install (requires uv)
uv sync --group dev

# Run CLI (launches Textual TUI)
uv run observa new

# Config + health checks
uv run observa config show
uv run observa setup check     # verifies deps, LLM key, Oracle conn, SQLcl path

# Tests
uv run pytest                       # all unit tests
uv run pytest tests/unit/mcp/ -v    # scoped run

# Type check
uv run pyright
```

## Architecture

Observa is a **multi-turn, collaborative** Oracle Database diagnostic tool. Specialist agents reflect over MCP tools across N turns (default 3, `config.turns.count`), publishing findings to a blackboard between turns; a Master LLM consolidates at the end.

**Execution flow** — `observa new` launches the Textual TUI:

1. **NewCaseScreen** (`observa/tui/new_case.py`): collects Problem Statement + Investigation Scope + Known Facts + DBID + a list of diagnostic files tagged by type (`alertlog`, `trace`, `tfa`, `hanganalyze`, `sqlhc`, `ddl`, `awr_report`). Files are NOT parsed — only their paths are exposed to agents via MCP.
2. **LiveRunScreen** (`observa/tui/live_run.py`): streams node events while the graph runs the turns.
3. **SummaryChatScreen** (`observa/tui/summary_chat.py`): shows `FinalSummary` and enables post-analysis chat with the Master (which retains full case context + MCP tool access).

**LangGraph graph** (`observa/graph/master.py`):

```
init → run_all_turns → consolidate → detect_contradictions → [debate?] → cove → END
```

- `run_all_turns_node` loops `turns.count` times, each turn calling `observa/graph/turn_controller.py::run_turn` which fans every enabled agent out in parallel under an `asyncio.wait_for` watchdog (budget = `turns.time_budget_seconds` + grace). Agents that time out or raise emit an `abstained=True` `TurnFindings`.
- `consolidate_node` calls `observa/graph/master_consolidator.py::consolidate` — the Master LLM receives the full blackboard digest and returns a `FinalSummary` (≤3 findings tied strictly to the investigation scope, explicit unknowns, calibrated confidence).
- `detect_contradictions` + `debate` + `cove` are carried over from the previous iteration, adapted to the new schema (see `observa/graph/cove.py` — reads `final_summary`, flattens `turn_findings` into findings).

**Reflective agents** (`observa/agents/`): 11 specialists (`wait_time`, `ash`, `infra`, `sql`, `memory`, `io_storage`, `concurrency`, `segment_object`, `rac`, `exadata`, `dg`). Each subclasses `ReflectiveAgent` (`observa/agents/base.py`) and sets only `name`, `display_name`, and `specialty_prompt`. The base handles prompt assembly (universal rules + specialty + blackboard digest + inputs), the tool-use loop, deadline management, and structured-output extraction into `TurnFindings`. Agents MUST stay within their specialty and abstain explicitly when nothing relevant is found.

**MCP layer** (`observa/mcp/`):
- `sqlcl_launcher.py` spawns Oracle SQLcl (≥25.x, `mcp.sqlcl_path`) as an MCP stdio subprocess.
- `sqlcl_guard.py` intercepts every SQL call, validates it against the allowlist (`DBA_HIST_*` + `V$INSTANCE/DATABASE/VERSION`, `GV$INSTANCE`) via `allowlist.py`, caps rows at `mcp.query_row_limit`, and parses the result.
- `files_server.py` is our own stdio MCP server exposing `list_files`, `read_file`, `search_file` (regex grep → line + byte-offset matches, hard-capped at 200 matches / 64 MB scanned, no timeout) and `extract_ora` (internal: pulls ORA-/incident lines from an alert log) against the analyst-provided registry (env var `OBSERVA_FILES_JSON`). `search_file`'s `byte_offset` feeds the existing `read_file(offset)` so agents jump straight to a hit instead of scanning blind. The pure scan engine lives in `file_scan.py`.
- `client.py::McpClient` is the single aggregator; it connects to both servers on entry and exposes four LangChain tools (`query_awr`, `list_files`, `read_file`, `search_file`). The TUI constructs it in `ObservaApp.on_case_start_requested` and passes it through the graph via `RunnableConfig["configurable"]["mcp_client"]`. Every `query_awr` / `query_catalog` / `read_file` / `search_file` call is recorded in `McpClient.ledger` (`observa/mcp/evidence.py`) with a stable `evidence_id` (Q-### for SQL, F-### for file reads, S-### for file searches) returned first in the tool payload; the ledger doubles as a per-case result cache keyed on canonical SQL or (pattern, path), namespaced by access path so agent and catalog queries never cross-serve. Findings cite IDs via `evidence_refs` (validated in `turn_controller`, hallucinated refs dropped); findings with no refs display `○ uncited` in the TUI. The Ctrl+D evidence browser and the HTML report appendix render the full ledger.
- **ORA incident timeline** (`observa/graph/ora_timeline.py`): a deterministic probe node on the `diff_probe → ora_timeline → turn_1` edge extracts ORA-/incident lines from the attached alert logs (via `client.scan_alertlog`, recorded as an S-### access), labels each against the problem/baseline windows (`problem` / `baseline` / `outside`), computes coverage (`covers_problem` distinguishes "no ORA in the window" from "log doesn't reach the window"), and seeds `CaseState.ora_timeline`. It renders as a text digest in every agent's cached prompt and an HTML table (per-window badges) in the report Methodology. Fail-open: no alert log / no client / no incidents → the node no-ops and the case proceeds.

**LLM wrapper** (`observa/llm/rate_limited.py`): `make_llm(model, *, slot="agent")` returns `RateLimitedLLM` — a delegating shim that retries anthropic/openai/429 errors with a 60-second sleep (TPM recovery window). `bind_tools` / `with_structured_output` return new wrapped runnables so the retry behavior survives the chain. A third routing branch handles `oci/<model_id>` strings (e.g., `oci/xai.grok-4-fast`) — these go through `observa.llm.oci_factory.build_oci_chat`, which constructs `ChatOCIGenAI` from `llm.providers.oci`. The `slot` kwarg (`agent` | `consolidator` | `cove` | `debate` | `master_chat`) is validated against `observa.llm.oci_capabilities.OCI_CAPS` before instantiating; eager validation at app boot (`observa.llm.validate_config.validate_llm_config`) blocks bad pairings before any case opens.

**Master chat** (`observa/graph/master_chat.py`): separate from the consolidator. After the graph finishes, `SummaryChatScreen` routes analyst questions through `master_chat()` — a tool-use loop binding the same MCP tools so the Master can run new queries / re-read files on demand, always grounded in the blackboard + FinalSummary.

## Stateless by design

Cases are ephemeral. There is no case database, no LangGraph checkpointer, no artifact store, no external-call cache. A case lives only in memory for the duration of the TUI session. `CASE-YYYY-MMDD-NN` IDs exist only for export filenames and logs.

## Key Conventions

- Agents NEVER hard-code SQL — they formulate queries via the LLM based on the problem statement + blackboard + specialty.
- All SQL from agents passes through `sqlcl_guard.guarded_query`, which enforces the allowlist. The allowlist lives in `observa/mcp/allowlist.py` — adding a new permitted view means editing `ALLOWED_EXACT` or `ALLOWED_PREFIXES`.
- `CaseState` (`observa/models.py`) uses `Annotated[list[X], add]` reducers on the blackboard fields (`turn_findings`, `agent_questions`, `chat_log`, `hypotheses`) so concurrent agent nodes append safely.
- Specialists must abstain explicitly (`abstained=True`) when nothing relevant is found in their area. Explicit abstains are valuable signal for other agents.
- TUI screens never hold the MCP client directly — `ObservaApp` owns its lifecycle.

## Environment

- `ORACLE_PWD` — Oracle password (for `setup check` connectivity test; SQLcl itself uses its own auth)
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — LLM provider key
- `~/.oci/config` — OCI provider credentials (only needed when `llm.providers.oci` is configured). Install OCI extras with `uv sync --extra oci`.
- Config file: `config/config.yaml` — note the `turns:` and `mcp:` sections; set `mcp.sqlcl_path` to the SQLcl binary (`sql` on Linux, `sql.exe` on Windows).

## SQLcl setup requirement

Observa talks to Oracle exclusively through SQLcl's MCP server. SQLcl must be installed (≥ 25.x) and the target AWR repository must be registered as a **named saved connection**. The name goes into `mcp.sqlcl_connection_name` in `config/config.yaml`.

**One-time setup** (replace `<pwd>` / `<host>` / `<port>` / `<service>` with your values):

```bash
<sqlcl_path>/sql /nolog
SQL> connect observa/<pwd>@<host>:<port>/<service>
SQL> conn -save observa
SQL> exit
```

Then set in `config/config.yaml`:

```yaml
mcp:
  sqlcl_path: "<path-to-sql.exe-or-sql>"
  sqlcl_connection_name: "observa"
```

Observa's `McpClient.__aenter__` calls SQLcl's `connect` tool with that name immediately after MCP init, then runs a fixed set of `ALTER SESSION` statements to normalize NLS (date/timestamp/numeric formats) so SQLcl's CSV output is unambiguous in locales that use comma decimal separators (pt-BR, de-DE, etc.). Without normalization the timestamp values embed commas that break CSV parsing.

If the configured connection name does not exist, Observa fails fast at startup with an error that lists the connections SQLcl currently has saved.

## AWR repository model

**The target database is an AWR repository, NOT the problem database.** AWR snapshots were imported from a remote production system via `awrload/awrxtrb`. Consequences enforced by the agent base class and the allowlist:

- `V$` / `GV$` views are rejected — they describe the local repository, not the problem.
- Every agent query must filter by `dbid = <case DBID>`; the repository can hold multiple source databases.
- Zero-row results are expected edge cases, not connection failures. Specialists abstain in that case rather than asking the analyst to "restore the connection".
- Topology/version info comes from `DBA_HIST_DATABASE_INSTANCE`, not `V$INSTANCE`.
