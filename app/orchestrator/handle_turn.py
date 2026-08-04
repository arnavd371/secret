"""
The orchestrator: wires Router/Intent -> session state -> decision policy
-> Tutor agent (or hard-gated refusal) -> session state update.

`handle_turn` is the only place these components are composed together.
Every component it calls already owns its own fallback behavior (see
router_agent.py and tutor_agent.py) — the orchestrator's job is sequencing
and state, not catching stray exceptions from components that should have
handled their own failure modes already. The one thing it enforces itself
is the REFUSE hard-gate: that path returns immediately and never touches
the Tutor agent at all.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents import router_agent, tutor_agent
from app.agents.fallback import build_refusal_response
from app.llm.client import ModelRouter
from app.models.contracts import ActionType, Blackboard, DecisionSignals, IntentType
from app.orchestrator.signals import DEFAULT_MASTERY_ESTIMATE, estimate_frustration, estimate_integrity_risk
from app.policy.decision import decide_pedagogical_action
from app.session.state import (
    InMemorySessionStateStore,
    ProblemSessionState,
    SessionStateStore,
    TurnOutcome,
    apply_turn_outcome,
)

logger = logging.getLogger(__name__)


def _infer_turn_outcome(intent: IntentType, problem_changed: bool) -> TurnOutcome:
    """
    Phase 1 has no CAS/answer verification (that's Phase 2), so we cannot
    know whether an attempt was actually correct. In its absence:
      - a problem switch is always a fresh start
      - an explicit hint_request escalates the ladder immediately
      - a practice-intent turn on the same, still-open problem is treated
        as another attempt that hasn't resolved yet (i.e. "incorrect" in
        ladder terms) — this is what drives escalation across repeated
        attempts per the Phase 1 spec
      - anything else (explain/check_work/exam_prep/off_topic) doesn't
        move the ladder
    Callers that already know the real outcome (e.g. a future grading
    signal) should pass turn_outcome explicitly to handle_turn instead of
    relying on this inference.
    """
    if problem_changed:
        return TurnOutcome.NEW_PROBLEM
    if intent == IntentType.HINT_REQUEST:
        return TurnOutcome.HINT_REQUESTED
    if intent == IntentType.PRACTICE:
        return TurnOutcome.ATTEMPTED_INCORRECT
    return TurnOutcome.NO_CHANGE


async def handle_turn(
    raw_input: str,
    session_id: str,
    student_id: str,
    problem_id: Optional[str] = None,
    *,
    router: Optional[ModelRouter] = None,
    session_store: Optional[SessionStateStore] = None,
    mastery_estimate: float = DEFAULT_MASTERY_ESTIMATE,
    turn_outcome: Optional[TurnOutcome] = None,
) -> Blackboard:
    """
    Returns the fully populated Blackboard for the turn (not just the bare
    TutorResponse) so callers/tests can assert on the intermediate
    IntentResult/DecisionSignals/Action as well as the final response —
    the response text itself is at `blackboard.tutor_response.text`.
    """
    router = router or ModelRouter()
    session_store = session_store or InMemorySessionStateStore()

    intent_result = await router_agent.classify_intent(raw_input, router)

    prev_state = await session_store.get(session_id, problem_id)
    problem_changed = problem_id is not None and problem_id != prev_state.problem_id
    current_state = (
        ProblemSessionState(session_id=session_id, problem_id=problem_id) if problem_changed else prev_state
    )

    signals = DecisionSignals(
        intent=intent_result.intent,
        mastery_estimate=mastery_estimate,
        assessment_mode=intent_result.assessment_mode_guess,
        integrity_risk=estimate_integrity_risk(raw_input, intent_result.assessment_mode_guess),
        attempt_count=current_state.attempt_count,
        frustration_signal=estimate_frustration(raw_input),
        hint_ladder_level=current_state.hint_ladder_level,
    )

    action = decide_pedagogical_action(signals)

    blackboard = Blackboard(
        session_id=session_id,
        student_id=student_id,
        problem_id=problem_id,
        raw_input=raw_input,
        intent_result=intent_result,
        decision_signals=signals,
    )
    blackboard.action = action

    # Hard gate: REFUSE short-circuits here. The Tutor agent is never
    # invoked for a REFUSE action, by construction.
    if action.action_type == ActionType.REFUSE:
        blackboard.tutor_response = build_refusal_response(action)
        return blackboard

    blackboard.tutor_response = await tutor_agent.generate(action, raw_input, router)

    outcome = turn_outcome if turn_outcome is not None else _infer_turn_outcome(intent_result.intent, problem_changed)
    new_state = apply_turn_outcome(current_state, problem_id, outcome)
    await session_store.save(new_state)

    return blackboard
