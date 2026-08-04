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

Phase 1 runs a fixed linear sequence rather than the Planner-produced
parallel stage graph of spec §6 (Retriever/CAS/Diagnostician stages don't
exist yet — see the TODOs on Blackboard fields in app/models/contracts.py).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from app.agents import router_agent, tutor_agent
from app.agents.fallback import build_refusal_response
from app.llm.client import ModelRouter
from app.models.contracts import ActionType, Blackboard, DecisionSignals, RawInput
from app.orchestrator.signals import DEFAULT_MASTERY_ESTIMATE, estimate_frustration, estimate_safety_result
from app.policy.decision import decide_pedagogical_action
from app.session.state import InMemorySessionStateStore, ProblemSessionState, SessionStateStore, advance_session_state

logger = logging.getLogger(__name__)


async def handle_turn(
    raw_input: str,
    session_id: str,
    student_id: str,
    problem_id: Optional[str] = None,
    *,
    router: Optional[ModelRouter] = None,
    session_store: Optional[SessionStateStore] = None,
    mastery_estimate: float = DEFAULT_MASTERY_ESTIMATE,
) -> Blackboard:
    """
    Returns the fully populated Blackboard for the turn (not just the bare
    TutorResponse) so callers/tests can assert on the intermediate
    IntentResult/DecisionSignals/Action as well as the final response —
    the response text itself is at `blackboard.final_response.text`.
    """
    router = router or ModelRouter()
    session_store = session_store or InMemorySessionStateStore()

    blackboard = Blackboard(
        turn_id=str(uuid.uuid4()),
        session_id=session_id,
        student_id=student_id,
        raw_input=RawInput(text=raw_input),
    )

    intent_result = await router_agent.classify_intent(raw_input, router)
    blackboard.intent_result = intent_result

    safety_result = estimate_safety_result(raw_input, intent_result.assessment_mode_guess)
    blackboard.safety_result = safety_result

    prev_state = await session_store.get(session_id, problem_id)
    problem_changed = problem_id is not None and problem_id != prev_state.problem_id
    current_state = (
        ProblemSessionState(session_id=session_id, problem_id=problem_id) if problem_changed else prev_state
    )

    signals = DecisionSignals(
        intent=intent_result.intent,
        mastery_estimate=mastery_estimate,
        assessment_mode=intent_result.assessment_mode_guess,
        integrity_risk=safety_result.integrity_risk,
        attempt_count=current_state.attempt_count,
        frustration_signal=estimate_frustration(raw_input),
        hint_ladder_level=current_state.hint_ladder_level,
    )

    action = decide_pedagogical_action(signals)
    blackboard.decision_action = action

    # Hard gate: REFUSE short-circuits here. The Tutor agent is never
    # invoked for a REFUSE action, by construction.
    if action.action_type == ActionType.REFUSE:
        response = build_refusal_response(action)
        blackboard.final_response = response
        return blackboard

    response = await tutor_agent.generate(action, raw_input, router)
    blackboard.draft_response = response
    blackboard.final_response = response

    new_state = advance_session_state(current_state, problem_id, intent_result.intent, action)
    await session_store.save(new_state)

    return blackboard
