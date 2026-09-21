"""ContradictionDetector — LLM-based pairwise mutual exclusivity checker."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from observa.models import Hypothesis

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """Check pairs of high-confidence hypotheses for mutual exclusivity via LLM."""

    def __init__(self, llm: Any, threshold: float = 0.3) -> None:
        self._llm = llm
        self._threshold = threshold

    async def run(self, hypotheses: list["Hypothesis"]) -> list[dict]:
        """Return contradiction dicts for each mutually-exclusive pair found."""
        eligible = [
            h for h in hypotheses
            if h.posterior > self._threshold and h.status != "rejected"
        ]
        if len(eligible) < 2:
            return []

        contradictions: list[dict] = []
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                try:
                    result = await self._check_pair(eligible[i], eligible[j])
                    if result["mutually_exclusive"]:
                        contradictions.append({
                            "hyp_a": eligible[i].hyp_id,
                            "hyp_b": eligible[j].hyp_id,
                            "explanation": result["explanation"],
                        })
                except Exception as exc:
                    logger.warning(
                        "ContradictionDetector pair check failed (hyp_a=%s, hyp_b=%s): %s",
                        eligible[i].hyp_id,
                        eligible[j].hyp_id,
                        exc,
                    )
        return contradictions

    async def _check_pair(self, hyp_a: "Hypothesis", hyp_b: "Hypothesis") -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = SystemMessage(content=(
            "You are a logical reasoning expert. Given two hypotheses about an Oracle Database "
            "performance issue, determine if they are mutually exclusive — meaning if one is the "
            "true root cause, the other cannot also be the root cause at the same time. "
            'Respond with valid JSON only: {"mutually_exclusive": bool, "explanation": str}'
        ))
        human = HumanMessage(content=(
            f"Hypothesis A (id={hyp_a.hyp_id}): {hyp_a.statement}\n"
            f"Hypothesis B (id={hyp_b.hyp_id}): {hyp_b.statement}\n\n"
            "Are these mutually exclusive root causes?"
        ))
        response = await self._llm.ainvoke([system, human])
        return json.loads(response.content)
