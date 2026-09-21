# Observa

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**A terminal cockpit for Oracle Database performance post-mortems.** You describe the incident and point Observa at an AWR repository; a team of specialist LLM agents queries `DBA_HIST_*` over several reflection turns, argues with each other on a shared blackboard, and a Master consolidates everything into a short root-cause summary where **every claim carries an evidence ID you can click and re-read**.

It is built for the case every Oracle DBA knows: *"the database was slow yesterday between 14:00 and 15:30, nobody knows why, here is an AWR export and an alert log."*

📖 **[Full documentation](docs/documentation.html)** — installation, every config parameter, and how a case actually runs.

```
● MCP connected
Turn 1 · 9 findings · 3 abstains so far
▸ Running turn 1

── Turn 1 ───────────────────────────────────────────────────────────────
02:14:25 [wait_time]      HIGH      WAIT_PROFILE_SHIFT_TO_IO — 'db file
         sequential read' rises from 18% to 61% of DB time between good
         snaps 41692-41705 and bad snaps 41820-41833.      [Q-004, Q-007]
02:14:31 [io_storage]     CRITICAL  SINGLE_BLOCK_READ_LATENCY_10X — average
         single-block read latency moved 2.1 ms → 22.4 ms on the datafiles
         backing the scoped index.                         [Q-011, Q-014]
02:14:38 [sql]            HIGH      PLAN_FLIP_TO_NESTED_LOOPS — SQL_ID
         7xkzm2wq4ftvb changed plan hash at snap 41821, multiplying the
         count of those now-slow reads.                           [Q-019]
02:14:51 [memory]         abstained — no SGA/PGA deviation between windows.
02:15:02 [rac]            skipped — single-instance DB (topology probe).
02:15:09 [file_reader]    INFO      ALERT_LOG_QUIET — no ORA- lines inside
         the problem window; log covers 2026-09-18..2026-09-21.   [S-002]
```

<sub>Synthetic data. Observa never ships screenshots of real diagnostic runs — see <a href="#a-note-on-your-data">A note on your data</a>.</sub>

---

## What makes it different

- **Evidence or it didn't happen.** Every SQL query and every file read is recorded in a per-case ledger with a stable ID (`Q-001`, `F-004`, `S-002`). Findings cite those IDs; hallucinated references are dropped at validation time and uncited findings are visibly marked `○ uncited` in the UI and the report.
- **Agents that argue.** Specialists publish findings between turns and read each other's work. A contradiction detector, a hypothesis synthesizer and a mechanical replay-verification pass run *after* consolidation — the summary's confidence is adjusted by how well the evidence actually held up on re-execution.
- **Abstaining is a first-class answer.** A specialist that finds nothing in its area says so explicitly. "The I/O agent looked and found nothing" is real signal for the other agents, not a gap.
- **Locked-down database access.** Agents cannot run arbitrary SQL. Every statement passes a `sqlglot`-backed allowlist (read-only, `DBA_HIST_*` only), must filter by the case DBID, and is row-capped.
- **Nothing is parsed for you.** Diagnostic files are exposed as paths over MCP with `list_files` / `read_file` / `search_file`. Agents grep and seek their way through a 4 GB alert log instead of relying on a brittle parser.

## What it is *not*

- Not a live-database monitoring tool. It reads history, not `V$`.
- Not a replacement for a DBA. It produces a *hypothesis with citations*, which you verify.
- Not stateful. There is no case database, no checkpoint store, no cache between runs — see [Stateless by design](#stateless-by-design).

---

## The AWR repository model

**Observa expects to connect to an AWR *repository*, not to the problem database.** The normal workflow is to extract AWR from the production system with `awrextr.sql` and import it into a separate database with `awrload.sql`.

This shapes everything downstream:

| Consequence | Why |
|---|---|
| `V$` / `GV$` views are rejected by the allowlist | They describe the local repository instance, not the incident |
| Every agent query must filter `dbid = <case DBID>` | One repository can hold many source databases |
| Topology and version come from `DBA_HIST_DATABASE_INSTANCE` | `V$INSTANCE` would describe the wrong machine |
| Zero rows is a normal answer | Specialists abstain instead of reporting a "connection failure" |

> [!IMPORTANT]
> **Oracle Diagnostics Pack licensing.** Querying `DBA_HIST_*` / AWR requires the Oracle Diagnostics Pack, licensed separately from the Oracle Database. Observa neither grants nor checks that entitlement — confirming that your organization is licensed for the data you point it at is your responsibility.

---

## How it works

### 1. Intake

`observa new` opens a Textual TUI. You supply:

- **Problem statement** — what went wrong, in your words.
- **Investigation scope** — what you want answered. The Master is held strictly to this.
- **Known facts** — anything already established (a deploy, a failover, a batch job).
- **DBID** and a **problem snap window**, picked from an interactive timeline of the snapshots actually present in the repository.
- A **baseline window** (auto-suggested as the same hours N days earlier) so agents compare instead of guessing at absolute numbers.
- **Diagnostic files**, each tagged by type: `alertlog`, `trace`, `tfa`, `hanganalyze`, `sqlhc`, `ddl`, `awr_report`. Only paths are handed to the agents.

### 2. The graph

```
init → topology_probe → diff_probe → ora_timeline
     → turn_1 → gate_1 → turn_2 → gate_2 → … → turn_N
     → consolidate → synthesize_hypotheses → detect_contradictions
     → [resolve_contradiction]? → replay_verify → END
```

| Node | What it does |
|---|---|
| `topology_probe` | Deterministic: reads version, RAC/Exadata/Data Guard flags, instance list. Agents whose subsystem isn't present are skipped before burning a single token. |
| `diff_probe` | Deterministic problem-vs-baseline deltas, seeded into every agent's prompt. |
| `ora_timeline` | Extracts ORA-/incident lines from the attached alert logs, labels each as `problem` / `baseline` / `outside`, and reports *coverage* — distinguishing "no errors in the window" from "the log doesn't reach the window". |
| `turn_N` | All active specialists fan out in parallel under a watchdog (`turns.time_budget_seconds`). Timeouts and exceptions become explicit abstains, never silent gaps. |
| `gate_N` | If any agent raised a question, the run pauses and asks *you*. Your answer joins the blackboard for later turns. |
| `consolidate` | The Master reads the full blackboard and emits a `FinalSummary`: at most 3 findings, tied strictly to the investigation scope, with explicit unknowns and calibrated confidence. |
| `replay_verify` | Re-executes the cited evidence and scores how much of it actually supports each claim, then patches confidence downward when it doesn't. |

Between turns, a **refinement** pass decides which agents stay active: `wait_time` and `ash` always run; others survive on high/critical findings or on being cross-referenced by a peer, with a floor of `min_active`.

### 3. Review

The summary screen shows the root cause, findings, unknowns and next steps — then hands you a chat with the Master, which keeps the full case context *and* live MCP tool access, so follow-up questions can trigger genuinely new queries rather than re-phrasings of the summary.

```
│ PRIMARY ROOT CAUSE · confidence 71%
  Single-block read latency on the datafiles backing SALES.IDX_ORDER_DATE rose
  roughly tenfold in the problem window. A concurrent plan flip to a nested-
  loops index range scan multiplied the number of those now-slow reads, which
  is what turned a 38-minute job into a 4-hour one.
  confidence 78% × verification support 0.91 = 71%

│ TOP FINDINGS (3)
  [1] CRITICAL (io_storage)      evidence: Q-011, Q-014
  [2] HIGH     (sql)             evidence: Q-019
  [3] MEDIUM   (segment_object)  ○ uncited

│ UNKNOWNS
  · No OS-level storage metrics were attached, so the firmware patch is
    correlated with the latency change, not proven to cause it.
```

`Ctrl+D` opens the evidence browser at any time: the full ledger of queries and file reads, replayable.

### The specialists

| Agent | Area |
|---|---|
| `wait_time` | Wait Time — always active |
| `ash` | Active Session History — always active |
| `sql` | SQL & Plans |
| `io_storage` | I/O & Storage |
| `memory` | Memory (SGA/PGA) |
| `concurrency` | Locks, enqueues, latches |
| `segment_object` | Segment & object hotspots |
| `infra` | Infrastructure / host |
| `rac` | RAC & Cache Fusion |
| `exadata` | Exadata |
| `dg` | Data Guard |
| `file_reader` | Alert logs, traces, TFA and other attached files |

Agents never hard-code SQL. Each subclasses `ReflectiveAgent` and contributes only a specialty prompt; the base class handles prompt assembly, the tool-use loop, deadlines and structured extraction.

---

## Requirements

- **Python 3.12+** and [uv](https://docs.astral.sh/uv/)
- **Oracle SQLcl 25.x or newer** — Observa talks to Oracle exclusively through SQLcl's MCP server. Requires a JRE, per SQLcl's own requirements.
- **An Oracle database holding imported AWR data**, reachable from a SQLcl *saved connection*.
- **An LLM API key** — Anthropic, OpenAI, or OCI Generative AI.

## Installation

```bash
git clone https://github.com/fmelodb/observa
cd observa
uv sync --group dev

# Optional: OCI Generative AI provider support
uv sync --extra oci
```

## Setup

### 1. Register a SQLcl saved connection

Observa connects by *name*, never by assembling a connect string. Create it once:

```bash
<sqlcl_path>/sql /nolog
SQL> connect observa/<password>@<host>:<port>/<service>
SQL> conn -save observa
SQL> exit
```

If the configured name doesn't exist, Observa fails at startup with an error listing the connections SQLcl does have saved.

On connect, Observa runs a fixed set of `ALTER SESSION` statements to normalize NLS date, timestamp and numeric formats. This is not cosmetic: in a comma-decimal locale (pt-BR, de-DE, …) the default formats embed commas that corrupt SQLcl's CSV output.

### 2. Create your config

```bash
cp config/config.example.yaml config/config.yaml
```

`config/config.yaml` is git-ignored — it holds machine-specific paths and tenancy identifiers. The settings you will almost certainly touch:

```yaml
mcp:
  sqlcl_path: "/opt/sqlcl/bin/sql"   # sql.exe on Windows
  sqlcl_connection_name: "observa"   # the name you saved above
  query_row_limit: 50
  enforce_window: "warn"             # warn | reject | off

llm:
  default_provider: "anthropic"      # anthropic | openai | oci
  providers:
    anthropic:
      model: "claude-sonnet-4-6"
      api_key_env: "ANTHROPIC_API_KEY"

turns:
  count: 2                           # reflection turns
  time_budget_seconds: 120           # per-turn watchdog
  max_concurrent_agents: 4
```

Models are assignable **per slot** — `agent`, `consolidator`, `hypotheses`, `cove`, `master_chat` — so you can run cheap models for the fan-out and a stronger one for consolidation. Prefix a model id with `oci/` to route it through OCI Generative AI (e.g. `oci/xai.grok-4-fast`). Invalid slot/model pairings are rejected at boot, before a case opens.

### 3. Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | one of these | Anthropic API key |
| `OPENAI_API_KEY` | one of these | OpenAI API key |
| `~/.oci/config` | one of these | OCI credentials, when `llm.providers.oci` is in use |
| `ORACLE_PWD` | no | Oracle password, used only by `setup check` connectivity probe |

### 4. Pre-flight

```bash
uv run observa setup check
```

Verifies Python dependencies, the LLM key, the config file, the SQLcl binary, the saved connection, OCI credentials when configured, and Oracle reachability. Resolve every `[FAIL]` before running a case.

---

## Usage

```bash
uv run observa new              # launch the TUI and start a case
uv run observa new --debug      # same, plus a full trace in observa-debug.log
uv run observa config show      # effective config, secrets redacted
uv run observa setup check      # health checks
```

### Output

From the summary screen you can export, into the current directory:

| File | Contents |
|---|---|
| `<CASE-ID>-report.html` | Self-contained report: findings, methodology, ORA timeline, and the full evidence ledger as an appendix |
| `<CASE-ID>-summary.md` | The `FinalSummary` in Markdown |
| `<CASE-ID>-chat.txt` | Transcript of the post-analysis chat |

`CASE-YYYY-MMDD-NN` identifiers exist only to name these files and tag log lines.

---

## Stateless by design

A case lives in memory for the duration of the TUI session and then it's gone. No case database, no LangGraph checkpointer on disk, no artifact store, no cross-run cache. The evidence ledger doubles as a *within-case* cache, keyed on canonical SQL or `(pattern, path)` and namespaced by access path so agent and catalog queries never cross-serve.

This is deliberate. Diagnostic data is often sensitive, and a tool that quietly accumulates a corpus of your customers' AWR internals is a liability, not a feature. If you want durability, export the report.

---

## Development

```bash
uv run pytest                        # all unit tests
uv run pytest tests/unit/mcp/ -v     # scoped run
uv run pyright                       # type check
```

### Layout

```
observa/
  cli.py              — Typer entry point
  config.py           — Settings (pydantic), per-slot model routing
  models.py           — CaseState blackboard, Finding, Hypothesis, FinalSummary
  snap_windows.py     — Snapshot window discovery and baseline suggestion
  agents/
    base.py           — ReflectiveAgent: prompts, tool loop, deadlines, extraction
    wait_time.py …    — 12 specialists, each ~40 lines of specialty prompt
  graph/
    master.py         — LangGraph StateGraph assembly
    turn_controller.py— Parallel fan-out, watchdog, evidence-ref validation
    topology_probe.py — Deterministic topology detection + agent skipping
    diff_probe.py     — Problem-vs-baseline deltas
    ora_timeline.py   — ORA-/incident extraction and window labelling
    refinement.py     — Which agents survive to the next turn
    master_consolidator.py, hypothesis_synth.py,
    contradiction_detector.py, contradiction_resolver.py,
    evidence_replay.py, replay_verify.py
    master_chat.py    — Post-analysis chat, MCP-enabled
  mcp/
    client.py         — Aggregator; owns the evidence ledger
    sqlcl_launcher.py — Spawns SQLcl as an MCP stdio subprocess
    sqlcl_guard.py    — Allowlist + DBID/window enforcement + row caps
    allowlist.py      — The allowlist itself (sqlglot-based)
    files_server.py   — Our own MCP server: list/read/search over case files
    file_scan.py      — Pure regex scan engine
    evidence.py       — Ledger and evidence IDs
  llm/
    rate_limited.py   — Retry wrapper that survives bind_tools / structured output
    oci_factory.py    — OCI Generative AI provider
    validate_config.py— Boot-time slot/model validation
  tui/
    new_case.py, live_run.py, summary_chat.py, widgets/
  export/
    html_report.py    — Jinja2 report renderer
```

The full reference — every configuration parameter, the setup walkthrough and the guardrail rules — is in [`docs/documentation.html`](docs/documentation.html).

### Extending the allowlist

Permitting a new view means editing `ALLOWED_EXACT` or `ALLOWED_PREFIXES` in `observa/mcp/allowlist.py`. Keep it read-only and keep it AWR-resident — anything reflecting the *local* instance defeats the repository model.

---

## A note on your data

Observa reads production diagnostics, so treat everything it touches as customer data.

- **Cases are never persisted.** No database, no checkpoint file, no cache that survives the session. The only things written to disk are the three export files you explicitly ask for.
- **Exports are not redacted.** `<CASE-ID>-report.html` contains the full evidence ledger — SQL text, DBIDs, schema and object names, hostnames, file paths. It is a customer artifact. `CASE-*` is git-ignored for that reason.
- **This repository ships no screenshots of real runs.** Terminal output in the docs is synthetic. A screenshot of a real case captures the DBID, SQL_IDs, hostnames, instance names, schema and index names and TFA trace filenames all at once — `docs/screenshots/` is git-ignored so that a helpful contribution cannot leak one by accident.

## Contributing

Issues and pull requests are welcome at [github.com/fmelodb/observa](https://github.com/fmelodb/observa).

Before opening a PR: `uv run pytest` must pass, and new behaviour should come with a unit test. `uv run pyright` still reports pre-existing errors — almost all of them in test doubles under `tests/` — so treat it as a ratchet: don't add new ones under `observa/`. Changes that touch the allowlist, the evidence ledger, or agent prompts deserve a note in the PR body explaining the reasoning — those are the parts where a plausible-looking change can quietly degrade grounding.

By contributing, you agree that your contributions are licensed under the Apache License 2.0.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for attribution and trademark information.

Oracle, Oracle Database, Exadata, Real Application Clusters, Data Guard, AWR, ASH and SQLcl are trademarks or registered trademarks of Oracle Corporation and/or its affiliates. This project is independent and is not affiliated with, endorsed by, or sponsored by Oracle Corporation. It bundles no Oracle software: SQLcl and the database are obtained from Oracle under Oracle's own terms and invoked as external programs.

**No warranty.** Observa produces LLM-generated hypotheses about production systems. Verify its conclusions before acting on them.
