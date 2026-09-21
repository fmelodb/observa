"""Verification-tail domain models (Feature 5)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from observa.models import (
    ContradictionResolution,
    FindingVerification,
    VerificationResult,
)


def test_finding_verification_roundtrip():
    fv = FindingVerification(
        finding_id="sql-1-abcd1234",
        claim="Full table scan on ORDERS drives DB time",
        verdict="supported",
        reasoning="Q-014 rows show 12M buffer gets on ORDERS",
        evidence_refs=["Q-014"],
        evidence_truncated=False,
    )
    assert fv.verdict == "supported"
    assert fv.evidence_refs == ["Q-014"]


def test_finding_verification_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        FindingVerification(
            finding_id="x", claim="c", verdict="maybe", reasoning="r",
        )


def test_verification_result_score_bounds():
    vr = VerificationResult(
        verifications=[],
        support_score=0.5,
        applied_confidence_factor=1.0,
        unsupported=[],
    )
    assert vr.support_score == 0.5
    with pytest.raises(ValidationError):
        VerificationResult(
            verifications=[], support_score=1.5,
            applied_confidence_factor=1.0, unsupported=[],
        )


def test_contradiction_resolution_defaults():
    cr = ContradictionResolution(
        hyp_a="H-1", hyp_b="H-2", reasoning="evidence favors H-1",
        posterior_a=0.7, posterior_b=0.2,
    )
    assert cr.favored_hyp_id is None
    assert cr.inconclusive is False
