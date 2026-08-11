"""
The orchestrator: wires Router/Intent -> session state -> decision policy
-> CAS verification + retrieval (EXPLAIN) / item generation (CHALLENGE) ->
Tutor agent (or hard-gated refusal) -> session state update.

`handle_turn` is the only place these components are composed together.
Every component it calls already owns its own fallback behavior (see
router_agent.py, tutor_agent.py, cas/solver.py, questions/generator.py) —
the orchestrator's job is sequencing and state, not catching stray
exceptions from components that should have handled their own failure
modes already. The one thing it enforces itself is the REFUSE hard-gate:
that path returns immediately and never touches the Tutor agent (or CAS/
retrieval/generation) at all.

Phase 2/3 ground exactly the action types permitted to use that grounding
per spec §1.5/§7.7: EXPLAIN may state a syllabus-specific claim or a final
answer, so it gets CAS verification + retrieval. CHALLENGE poses a new
problem "instead of full solve" — it never states an answer — so it gets
a real, CAS-verified extension item from the Question Generation Engine
instead. QUESTION/HINT/SUPPORTIVE_SCAFFOLD need neither.

The orchestrator still runs a fixed linear sequence rather than the
Planner's parallel stage graph (spec §6) — Planner, Memory, and Diagnosis
agents remain later-phase non-goals (see the TODOs on Blackboard's
stubbed fields in app/models/contracts.py).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from app.agents import router_agent, tutor_agent
from app.agents.fallback import build_refusal_response
from app.cas.extraction import extract_math_task
from app.cas.models import CASResult
from app.cas.solver import run_cas_operation_async
from app.knowledge.retriever import KnowledgeBase, get_default_knowledge_base
from app.knowledge.schemas import RetrievedChunk
from app.llm.client import ModelRouter
from app.models.contracts import ActionType, Blackboard, DecisionSignals, RawInput
from app.orchestrator.signals import DEFAULT_MASTERY_ESTIMATE, estimate_frustration, estimate_safety_result
from app.policy.decision import decide_pedagogical_action
from app.questions.generator import ItemGenerationError, generate_item_async, select_template_for_topic
from app.questions.models import GeneratedItem
from app.session.state import InMemorySessionStateStore, ProblemSessionState, SessionStateStore, advance_session_state

logger = logging.getLogger(__name__)


async def _ground_explain_turn(
    raw_input: str, topic_hint: Optional[str], knowledge_base: KnowledgeBase
) -> tuple[Optional[CASResult], list[RetrievedChunk]]:
    cas_result: Optional[CASResult] = None
    math_task = extract_math_task(raw_input)
    if math_task is not None:
        cas_result = await run_cas_operation_async(
            math_task.operation, math_task.expression, math_task.variable, math_task.at
        )

    retrieved_chunks = knowledge_base.retrieve(raw_input, topic_hint=topic_hint)
    return cas_result, retrieved_chunks


async def _generate_challenge_item(topic_hint: Optional[str]) -> Optional[GeneratedItem]:
    template_id = select_template_for_topic(topic_hint)
    try:
        return await generate_item_async(template_id)
    except ItemGenerationError as exc:
        logger.warning("Question generation failed for template %s (%s); Tutor falls back to a generic extension prompt", template_id, exc)
        return None


async def handle_turn(
    raw_input: str,
    session_id: str,
    student_id: str,
    problem_id: Optional[str] = None,
    *,
    router: Optional[ModelRouter] = None,
    session_store: Optional[SessionStateStore] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
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
    knowledge_base = knowledge_base or get_default_knowledge_base()

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

    # Hard gate: REFUSE short-circuits here. The Tutor agent (and CAS/
    # retrieval/generation) is never invoked for a REFUSE action, by
    # construction.
    if action.action_type == ActionType.REFUSE:
        response = build_refusal_response(action)
        blackboard.final_response = response
        return blackboard

    cas_result: Optional[CASResult] = None
    retrieved_chunks: list[RetrievedChunk] = []
    challenge_item: Optional[GeneratedItem] = None

    if action.action_type == ActionType.EXPLAIN:
        cas_result, retrieved_chunks = await _ground_explain_turn(raw_input, intent_result.topic_hint, knowledge_base)
        blackboard.cas_result = cas_result
        blackboard.retrieved_chunks = retrieved_chunks
    elif action.action_type == ActionType.CHALLENGE:
        challenge_item = await _generate_challenge_item(intent_result.topic_hint)
        blackboard.generated_item = challenge_item

    response = await tutor_agent.generate(
        action,
        raw_input,
        router,
        cas_result=cas_result,
        retrieved_chunks=retrieved_chunks,
        challenge_item=challenge_item,
    )
    blackboard.draft_response = response
    blackboard.final_response = response

    new_state = advance_session_state(current_state, problem_id, intent_result.intent, action)
    await session_store.save(new_state)

    return blackboard
