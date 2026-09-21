"""Feature-5 config surface: renamed flags + hypotheses slot."""
from __future__ import annotations

from observa.config import get_settings
from observa.llm.oci_capabilities import SLOT_REQUIREMENTS, compatible_models_for_slot


def test_settings_expose_renamed_flags():
    s = get_settings()
    assert isinstance(s.agents_enable_hypotheses, bool)
    assert isinstance(s.agents_verification_support_threshold, float)
    assert isinstance(s.llm_hypotheses_model, str)
    # Old names are gone.
    assert not hasattr(s, "agents_enable_debate")
    assert not hasattr(s, "agents_debate_rounds")
    assert not hasattr(s, "llm_debate_advocate_model")


def test_hypotheses_slot_registered():
    assert "hypotheses" in SLOT_REQUIREMENTS
    assert "debate" not in SLOT_REQUIREMENTS
    # A structured-only slot has candidates.
    assert compatible_models_for_slot("hypotheses")
