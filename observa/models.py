"""Core domain models for the multi-turn collaborative architecture.

Shape:
  - Inputs the analyst provides (problem statement, known facts, DBID, files).
  - Blackboard that agents read from and append to across turns.
  - Final summary produced by the master consolidator.

Finding / Hypothesis are the retained shared types from the previous
iteration; they are now consumed by hypothesis synthesis, contradiction
detection/resolution, and evidence-replay verification (the mechanical
verification tail).
"""
from __future__ import annotations

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------

FileType = Literal[
    "alertlog",
    "trace",
    "tfa",
    "hanganalyze",
    "sqlhc",
    "ddl",
    "awr_report",
]


class InputFile(BaseModel):
    """A diagnostic file the analyst handed us. We don't parse it — the agents
    decide whether to read it via the files MCP server."""

    type: FileType
    path: str
    label: str = ""


class SnapWindow(BaseModel):
    """A snap-id range in the AWR repository. Times are None in manual-entry
    mode (MCP offline at intake), in which case per-second rates are skipped."""

    begin_snap: int
    end_snap: int
    begin_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @model_validator(mode="after")
    def _check_range(self) -> "SnapWindow":
        if self.end_snap <= self.begin_snap:
            raise ValueError("end_snap must be greater than begin_snap")
        if self.begin_time is not None and self.end_time is not None:
            if self.end_time <= self.begin_time:
                raise ValueError("end_time must be after begin_time")
        return self

    def duration_seconds(self) -> Optional[float]:
        if self.begin_time is None or self.end_time is None:
            return None
        return (self.end_time - self.begin_time).total_seconds()


# ---------------------------------------------------------------------------
# Shared finding / hypothesis types (kept for the verification tail)
# ---------------------------------------------------------------------------

Severity = Literal["info", "low", "medium", "high", "critical"]


class Finding(BaseModel):
    finding_id: str
    agent: str
    turn: int
    severity: Severity
    code: str
    description: str
    # Verifiable citations: IDs of EvidenceRecords in the case's evidence
    # ledger (Q-### for SQL, F-### for file reads). ``uncited`` is set when
    # no valid ref survived validation — the finding stands but is flagged.
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_note: str = ""
    uncited: bool = False
    related_objects: list[str] = Field(default_factory=list)

    def evidence_text(self) -> str:
        """Human-readable citation line: 'Q-014, F-003 — note'."""
        refs = ", ".join(self.evidence_refs)
        if refs and self.evidence_note:
            return f"{refs} — {self.evidence_note}"
        return refs or self.evidence_note


class Hypothesis(BaseModel):
    hyp_id: str
    statement: str
    prior: float = Field(ge=0, le=1)
    posterior: float = Field(ge=0, le=1)
    supporting_findings: list[str] = Field(default_factory=list)
    contradicting_findings: list[str] = Field(default_factory=list)
    status: Literal["open", "validated", "rejected", "needs_more_data"] = "open"


# ---------------------------------------------------------------------------
# Turn-based blackboard entries
# ---------------------------------------------------------------------------

class AgentQuestion(BaseModel):
    """Question an agent asks the analyst at the end of a turn."""

    agent: str
    turn: int
    question: str


class TurnFindings(BaseModel):
    """What one agent produces in one turn."""

    agent: str
    turn: int
    findings: list[Finding] = Field(default_factory=list)
    questions: list[AgentQuestion] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str = ""
    # Short note on what the agent explored this turn (query count, files read).
    # Kept for transparency; not structured evidence.
    notes: str = ""


class ChatMessage(BaseModel):
    """A message in the turn-gate chat or post-analysis chat."""

    sender: Literal["user", "agent", "master"]
    agent: Optional[str] = None  # populated when sender == "agent"
    turn: Optional[int] = None   # None for post-analysis chat
    text: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Verification tail (Feature 5) — mechanical, evidence-grounded
# ---------------------------------------------------------------------------

VerificationVerdict = Literal["supported", "partial", "unsupported", "contradicted"]


class FindingVerification(BaseModel):
    finding_id: str
    claim: str
    verdict: VerificationVerdict
    reasoning: str
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_truncated: bool = False


class VerificationResult(BaseModel):
    verifications: list[FindingVerification] = Field(default_factory=list)
    support_score: float = Field(ge=0, le=1)
    applied_confidence_factor: float = Field(ge=0, le=1)
    unsupported: list[str] = Field(default_factory=list)


class ContradictionResolution(BaseModel):
    hyp_a: str
    hyp_b: str
    favored_hyp_id: Optional[str] = None
    inconclusive: bool = False
    reasoning: str = ""
    posterior_a: float = Field(ge=0, le=1, default=0.0)
    posterior_b: float = Field(ge=0, le=1, default=0.0)


# ---------------------------------------------------------------------------
# Master consolidation output
# ---------------------------------------------------------------------------

class SummaryFinding(BaseModel):
    """A single entry in the final summary's top-N findings list."""

    agent: str
    severity: Severity
    description: str
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence-record IDs copied verbatim from the source specialist "
            "finding(s) this entry is based on (e.g. \"Q-014\", \"F-003\"). "
            "Never invent an ID."
        ),
    )
    evidence: str = ""


class FinalSummary(BaseModel):
    """Master's consolidated output. Must not contain more than 3 findings."""

    problem_restated: str
    top_findings: list[SummaryFinding] = Field(default_factory=list, max_length=3)
    root_cause: str
    confidence: float = Field(ge=0, le=1)
    unknowns: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CaseState (LangGraph blackboard)
# ---------------------------------------------------------------------------

class CaseState(TypedDict, total=False):
    # Identity — ephemeral, stateless between sessions.
    case_id: str
    created_at: datetime

    # Oracle context (enriched by init_node when MCP reachable).
    dbid: int
    oracle_version: str
    topology: str
    instance_numbers: list[int]

    # Topology probe (Level 1 deterministic skip). Populated by
    # ``topology_probe_node`` before turn 1.
    topology_flags: dict
    agents_skipped: list[str]
    agents_skip_reasons: dict[str, str]

    # Refinement audit (Level 5 — turn-to-turn agent selection).
    # Per-turn record of who actually ran and why others were dropped.
    agents_active_per_turn: dict[int, list[str]]
    agents_refinement_log: dict[int, dict[str, str]]

    # User-supplied inputs.
    problem_statement: str
    investigation_scope: str
    known_facts: list[str]
    input_files: list[InputFile]

    # Problem/baseline windows (Feature 1). baseline_window is None when the
    # analyst dismissed the baseline; problem_window is None only in legacy
    # manual mode with no window entered.
    problem_window: Optional[SnapWindow]
    baseline_window: Optional[SnapWindow]
    # Structured digest from the diff probe (None => probe skipped/failed).
    baseline_diff: Optional[dict]
    baseline_suspect: bool
    # ORA incident timeline (Feature 3). None => no alert log / probe skipped.
    ora_timeline: Optional[dict]

    # Turn control.
    current_turn: int
    total_turns: int

    # Blackboard (parallel-safe — agents append in each turn).
    turn_findings: Annotated[list[TurnFindings], add]
    agent_questions: Annotated[list[AgentQuestion], add]
    chat_log: Annotated[list[ChatMessage], add]

    # Orchestration state used by the verification tail.
    hypotheses: Annotated[list[Hypothesis], add]
    contradictions: list[dict]
    contradiction_resolutions: list[ContradictionResolution]

    # Final results.
    final_summary: Optional[FinalSummary]
    verification: Optional[VerificationResult]
