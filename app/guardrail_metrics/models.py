"""
Per-turn guardrail signal record and the aggregate report computed over
many of them. This is the real telemetry shape any deployment's request
handler could populate turn-by-turn (see app.guardrail_metrics.extraction
for how a turn's real TutorResponse.ui_metadata becomes one of these) -
distinct from app.eval's offline golden-scenario harness, which checks
correctness against known-good answers rather than aggregating live
guardrail behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class GuardrailTurnSignals(BaseModel):
    turn_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # True when app.agents.tutor_agent's leak-check
    # (_violates_action_contract) discarded the draft for this turn.
    leak_check_triggered: bool = False

    # The Verifier/Critic's verdict for this turn, when the critic ran
    # at all (None for a turn that never reached the critic - e.g. one
    # discarded earlier by the leak-check or CAS gate).
    critic_verdict: Optional[str] = None
    # True when the critic call itself degraded to its static fallback
    # (Phase 6) rather than completing a real model critique.
    critic_degraded: bool = False

    # The grounding entailment score for this turn, when grounding was
    # actually checked (None when there were no retrieved_chunks to
    # check against at all, as opposed to a checked-and-failed 0.0).
    grounding_score: Optional[float] = None

    # True whenever the turn's final response was a templated fallback
    # rather than approved LLM prose, regardless of which specific gate
    # caused it - the real "fallback rate" signal.
    fell_back_to_template: bool = False
    # Machine-readable cause when fell_back_to_template is True, e.g.
    # "leak_check", "critic_block", "grounding_failed",
    # "model_call_failed", "regeneration_failed". None for an approved
    # (non-templated) turn.
    fallback_reason: Optional[str] = None


class GuardrailMetricsReport(BaseModel):
    total_turns: int
    leak_check_trigger_rate: float
    fallback_rate: float
    critic_degraded_rate: float
    # Verdict value -> fraction of turns where the critic actually ran
    # (not a fraction of all turns - a turn the critic never reached
    # shouldn't dilute the distribution of verdicts among turns it did).
    critic_verdict_distribution: dict[str, float] = Field(default_factory=dict)
    critic_ran_count: int = 0
    # None when no turn in this batch ever had a grounding score to
    # average - an honest "no data" rather than a misleading 0.0.
    average_grounding_score: Optional[float] = None
    grounding_checked_count: int = 0
