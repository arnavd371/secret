"""
Pure extraction: TutorResponse.ui_metadata (already the real, populated
dict app.agents.tutor_agent and app.agents.fallback build for every
turn - see their own docstrings for the exact keys each path sets) ->
one GuardrailTurnSignals record. No I/O, no model calls; this is the
one function that has to agree, key-for-key, with what those two
modules actually emit, so it's kept small and is exactly what
tests/test_guardrail_metrics.py exercises with synthetic ui_metadata
dicts standing in for real turns.
"""

from __future__ import annotations

from typing import Any

from app.guardrail_metrics.models import GuardrailTurnSignals


def extract_guardrail_signals(turn_id: str, ui_metadata: dict[str, Any]) -> GuardrailTurnSignals:
    fell_back = bool(ui_metadata.get("templated", False))
    fallback_reason = ui_metadata.get("fallback_reason") if fell_back else None

    return GuardrailTurnSignals(
        turn_id=turn_id,
        leak_check_triggered=fallback_reason == "leak_check",
        critic_verdict=ui_metadata.get("critique_verdict"),
        critic_degraded=bool(ui_metadata.get("critic_degraded", False)),
        grounding_score=ui_metadata.get("grounding_score"),
        fell_back_to_template=fell_back,
        fallback_reason=fallback_reason,
    )
