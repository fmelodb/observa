"""Unit tests for the abstain-reason formatter in observa.agents.base."""

from observa.agents.base import _format_abstain_reason


def test_capability_error_message_names_model_and_remediation():
    """Provider rejecting tools must produce an actionable message naming the
    model and pointing at config.yaml."""
    exc = RuntimeError("tools not supported on this endpoint")
    msg = _format_abstain_reason("oci/google.gemini-2.5-flash-lite", exc, "tool_loop")
    assert "oci/google.gemini-2.5-flash-lite" in msg
    assert "config.yaml" in msg
    assert "tools" in msg.lower()


def test_function_calling_marker_also_detected():
    exc = RuntimeError("function calling not supported by model")
    msg = _format_abstain_reason("oci/some.model", exc, "tool_loop")
    assert "config.yaml" in msg


def test_unsupported_response_format_marker_detected():
    exc = RuntimeError("unsupported response_format requested")
    msg = _format_abstain_reason("oci/some.model", exc, "structured_output")
    assert "config.yaml" in msg


def test_generic_error_still_names_model_and_stage():
    """Non-capability errors must still be attributable to a model."""
    exc = TimeoutError("connection timed out")
    msg = _format_abstain_reason("claude-sonnet-4-6", exc, "tool_loop")
    assert "claude-sonnet-4-6" in msg
    assert "tool_loop" in msg
    assert "TimeoutError" in msg


def test_marker_match_is_case_insensitive():
    exc = RuntimeError("Tools Not Supported here")
    msg = _format_abstain_reason("x", exc, "tool_loop")
    assert "config.yaml" in msg
