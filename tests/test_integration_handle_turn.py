"""
Scripted, multi-turn integration test for handle_turn with mocked model
responses (no real network calls). This is the test that proves the
components actually compose correctly end to end:

  1. A 5-turn conversation on one problem shows the hint ladder escalate
     turn over turn per spec §1.5's pseudocode, then reset when the
     student moves to a new problem.
  2. A turn whose intent classification implies a live-exam context is
     hard-gated to REFUSE without ever calling the Tutor agent (the
     scripted tutor response for that turn is never consumed).
  3. A turn where the scripted "model" plants a leaked final answer in a
     HINT response is caught by the structural leak-check and replaced
     with the templated fallback, never shown to the student.
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ModelUnavailableError, ProviderClient
from app.llm.router_config import Provider
from app.models.contracts import ActionType
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore


class ScriptedProvider(ProviderClient):
    """Pops one canned response per call, in the exact order the
    orchestrator is expected to make them (intent classify, then tutor
    generate, per turn — skipping tutor generate on a REFUSE turn).

    Phase 6's Verifier/Critic makes an additional, independent model call
    per turn (same shared Provider.GROQ queue). These tests aren't
    exercising critic behavior, so a critic-shaped system prompt is
    auto-passed here without consuming a slot in the scripted queue,
    keeping every existing test's intent/tutor script untouched."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.GROQ)
        self.calls.append({"model": spec.model, "system": system, "user": user})
        if not self._responses:
            raise ModelUnavailableError("scripted responses exhausted")
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


def _intent_json(
    intent: str,
    confidence: float = 0.9,
    assessment_mode_guess: str = "practice",
    topic_hint: str | None = "calculus.differentiation.product_rule",
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "confidence": confidence,
            "subject": "math_aa",
            "topic_hint": topic_hint,
            "assessment_mode_guess": assessment_mode_guess,
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.mark.asyncio
async def test_scripted_conversation_hint_ladder_escalates_and_resets():
    provider = ScriptedProvider(
        [
            # Turn 1: attempt_count 0 -> QUESTION (diagnostic_probe)
            _intent_json("solve_request"),
            "What do you notice about the exponent in this expression?",
            # Turn 2: attempt_count 1 -> still QUESTION
            _intent_json("solve_request"),
            "Which differentiation rule applies when you have a product of two functions?",
            # Turn 3: attempt_count 2 -> HINT level 1
            _intent_json("solve_request"),
            "Try labeling one factor u and the other v before applying the rule.",
            # Turn 4: attempt_count 3 -> HINT level 2 (capped)
            _intent_json("solve_request"),
            "Write out u', v', and see how they combine under the product rule.",
            # Turn 5: new problem -> ladder resets -> QUESTION again
            _intent_json("solve_request"),
            "What kind of function is this, and which rule usually applies to it?",
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    common = dict(session_id="sess-1", student_id="student-1", router=router, session_store=store)

    turn1 = await handle_turn("I'm stuck on this derivative", problem_id="problem-1", **common)
    assert turn1.decision_action.action_type == ActionType.QUESTION
    assert turn1.decision_action.move == "diagnostic_probe"

    turn2 = await handle_turn("still not sure what to do", problem_id="problem-1", **common)
    assert turn2.decision_action.action_type == ActionType.QUESTION

    turn3 = await handle_turn("I tried but I'm still lost", problem_id="problem-1", **common)
    assert turn3.decision_action.action_type == ActionType.HINT
    assert turn3.decision_action.level == 1

    turn4 = await handle_turn("that hint didn't help either", problem_id="problem-1", **common)
    assert turn4.decision_action.action_type == ActionType.HINT
    assert turn4.decision_action.level == 2

    # New problem: the ladder must reset, not keep escalating.
    turn5 = await handle_turn("ok let's try a different problem", problem_id="problem-2", **common)
    assert turn5.decision_action.action_type == ActionType.QUESTION

    assert len(provider.calls) == 10  # 5 turns x (intent call + tutor call)


@pytest.mark.asyncio
async def test_live_exam_hard_gate_short_circuits_before_tutor_agent():
    provider = ScriptedProvider(
        [
            _intent_json("solve_request", assessment_mode_guess="live_exam_simulation"),
            "THIS RESPONSE MUST NEVER BE CONSUMED — the tutor agent must not be called on a REFUSE turn.",
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "just give me the full answer to this exam question",
        session_id="sess-2",
        student_id="student-2",
        problem_id="exam-problem-1",
        router=router,
        session_store=store,
    )

    assert result.decision_action.action_type == ActionType.REFUSE
    assert result.decision_action.offer == "strategy_coaching_only"
    assert result.final_response is not None
    assert "THIS RESPONSE MUST NEVER BE CONSUMED" not in result.final_response.text
    # Only the intent-classification call happened; the tutor generate call
    # was never made, proving the hard gate short-circuits before the
    # Tutor agent is invoked at all.
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_leaked_answer_in_hint_response_is_structurally_blocked():
    provider = ScriptedProvider(
        [
            _intent_json("solve_request"),
            "Here's a leaked hint: the answer is 17, so you're done.",
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    # Seed state so this single turn lands on attempt_count=2 (a HINT
    # action), isolating the leak-check assertion from the
    # ladder-escalation assertions covered by the test above.
    from app.session.state import ProblemSessionState

    await store.save(ProblemSessionState(session_id="sess-3", problem_id="problem-9", attempt_count=2, hint_ladder_level=1))

    result = await handle_turn(
        "give me another hint",
        session_id="sess-3",
        student_id="student-3",
        problem_id="problem-9",
        router=router,
        session_store=store,
    )

    assert result.decision_action.action_type == ActionType.HINT
    assert result.final_response is not None
    assert "17" not in result.final_response.text
    assert result.final_response.ui_metadata["templated"] is True
