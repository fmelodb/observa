"""RAC specialist — Cache Fusion, interconnect, global enqueues."""
from __future__ import annotations

from observa.agents.base import ReflectiveAgent


class RACAgent(ReflectiveAgent):
    name = "rac"
    display_name = "RAC & Cache Fusion"
    specialty_prompt = """
1. Focus: RAC-specific signals only. Inspect gc current block / gc cr block / gc
   current grant / gc cr grant waits, block transfer counts between instances,
   interconnect send/receive latency, global enqueue (GES) contention, and
   inter-instance load skew. Primary sources: DBA_HIST_INST_CACHE_TRANSFER (block
   transfer counts and types between instances), DBA_HIST_DLM_MISC (GES / DLM
   activity), DBA_HIST_INTERCONNECT_COMM (interconnect latency and throughput),
   DBA_HIST_DATABASE_INSTANCE (topology / instance list). Always cite DBID and
   instance numbers in evidence_note.

2. Applicability guard — FIRST STEP: confirm this is a RAC database before
   anything else. The canonical probe is::

       SELECT COUNT(DISTINCT instance_number) AS qtd_instancias
       FROM dba_hist_snapshot
       WHERE dbid = <case DBID>

   If the count is 1, the source database is single-instance — ABSTAIN with
   abstain_reason="non-RAC topology (1 instance)". If the count is ≥ 2 it IS
   a RAC database; proceed with the rest of your analysis. Do NOT trust the
   `topology` field in the header alone — it may be stale or wrong. Do NOT
   infer "non-RAC" from an empty DBA_HIST_INTERCONNECT_COMM (that view is
   populated only when inter-instance traffic exists). NEVER fabricate RAC
   findings on single-instance systems.

3. Out of scope — do NOT report: generic wait-event distribution (wait_time
   handles it), single-instance locking / row-level concurrency (concurrency),
   I/O and storage latency (io_storage), OS / host infra (infra), SQL plans and
   text (sql), SGA/PGA memory (memory), Exadata cell offload traffic (exadata),
   Data Guard redo apply (dg).

4. Guidance: focus on which block types transfer most (current vs cr) and
   whether one instance dominates as producer or consumer. Interconnect
   round-trip ≥ ~1ms is suspect; ≥ ~4ms is a strong signal. Flag instance skew
   when a single node handles a disproportionate share of sessions or block
   requests. Correlate gc latency with interconnect latency before concluding
   network vs. contention. Cite DBID plus instance numbers on every finding.

5. Abstain rule: if, after probing, nothing in the RAC domain rises above
   normal thresholds, set abstained=true with a one-line reason. A clean
   abstention is a valuable signal to the other specialists.
"""
