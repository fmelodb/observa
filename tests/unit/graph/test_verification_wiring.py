"""Graph topology for the verification tail (Feature 5)."""
from __future__ import annotations

from observa.graph.master import build_graph
from observa.models import CaseState, Hypothesis


def test_graph_builds_with_new_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert "synthesize_hypotheses" in nodes
    assert "resolve_contradiction" in nodes
    assert "replay_verify" in nodes
    assert "debate" not in nodes
    assert "cove" not in nodes


def test_route_after_detection_prefers_resolve_when_contradictions():
    from observa.graph.master import _route_after_detection
    assert _route_after_detection({"contradictions": [{"hyp_a": "H-1", "hyp_b": "H-2"}]}) \
        == "resolve_contradiction"
    assert _route_after_detection({"contradictions": []}) == "replay_verify"
