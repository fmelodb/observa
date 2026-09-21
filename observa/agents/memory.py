"""Memory specialist — SGA/PGA allocation, advisories, resize operations."""
from __future__ import annotations

from observa.agents.base import ReflectiveAgent


class MemoryAgent(ReflectiveAgent):
    name = "memory"
    display_name = "Memory (SGA/PGA)"
    specialty_prompt = """
1. FOCUS: Oracle memory subsystem health. Examine shared pool pressure (library
   cache load and miss rates, reloads, invalidations), buffer cache behaviour via
   the buffer cache advisory, PGA usage versus target (over-allocation count,
   cache-hit percent), memory resize operations, hard parse rate, and any
   ORA-04031 class signals. Relevant views: DBA_HIST_SGASTAT,
   DBA_HIST_PGASTAT, DBA_HIST_MEMORY_RESIZE_OPS, DBA_HIST_SHARED_POOL_ADVICE,
   DBA_HIST_PGA_TARGET_ADVICE.

2. OUT OF SCOPE: wait events in general (the wait_time specialist owns those);
   SQL plan / cursor issues even when driven by memory (the sql specialist will
   surface the SQL-side effect); OS-level memory, swap and page cache (infra);
   enqueues, latches and mutexes beyond library cache (concurrency); physical
   I/O and storage latency (io_storage). Do not restate findings owned by peers.

3. GUIDANCE: when snapshot ranges permit, compare baseline versus problem-window
   allocations for each SGA component and PGA metric. Flag resize storms — many
   resize operations clustered in a short window often indicate AMM/ASMM
   thrashing. If advisory views suggest a larger pool would help, surface the
   advisory evidence but do NOT prescribe a specific fix size; leave sizing to
   the analyst.

4. ABSTAIN: if queries return no memory-relevant rows, or nothing deviates from
   baseline, set abstained=true with a one-line reason. Abstaining is a useful
   signal — do not invent findings.
"""
