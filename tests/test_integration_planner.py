"""
Integration coverage for Phase 11's Planner: through the real
handle_turn() orchestrator, a graded check_work submission produces a
real, populated Blackboard.execution_plan reflecting the actual
concurrent post-grading stage graph that ran, and every existing
behavior it wraps (mastery write, review record, diagnosis) still works
exactly as before the refactor.
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.memory.store import InMemoryMemoryStore
from app.adaptive.store import InMemoryReviewStateStore
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore

CHAIN_RULE_TOPIC = "calculus.differentiation.chain_rule"


class ScriptedProvider(ProviderClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.ANTHROPIC)
        self.calls.append({"model": spec.model, "system": system, "user": user})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


def _check_work_intent_json(topic_hint: str = CHAIN_RULE_TOPIC) -> str:
    return json.dumps(
        {
            "intent": "check_work",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": topic_hint,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.mark.asyncio
async def test_correct_grading_produces_a_real_execution_plan_without_a_diagnosis_stage():
    provider = ScriptedProvider([_check_work_intent_json()])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()

    result = await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="sess-planner-1",
        student_id="student-planner-1",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
    )

    assert result.mark_result.total_awarded == result.mark_result.total_available
    assert result.execution_plan is not None
    stage_names = {s["name"] for s in result.execution_plan["stages"]}
    assert stage_names == {"mastery_write", "review_record"}  # no diagnosis: nothing wrong
    assert all(s["error"] is None for s in result.execution_plan["stages"])
    assert result.execution_plan["total_duration_ms"] >= 0


@pytest.mark.asyncio
async def test_incorrect_grading_produces_a_plan_including_the_diagnosis_stage():
    provider = ScriptedProvider([_check_work_intent_json(topic_hint="calculus.differentiation.product_rule")])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-planner-2",
        student_id="student-planner-2",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
        student_work="u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)",  # wrong, pattern-matched
    )

    assert result.diagnosis_result is not None
    assert result.diagnosis_result.misconception_id == "MISC-CALC-010"
    stage_names = {s["name"] for s in result.execution_plan["stages"]}
    assert stage_names == {"mastery_write", "review_record", "diagnosis"}


@pytest.mark.asyncio
async def test_all_post_grading_writes_actually_landed_despite_running_concurrently():
    """The real point of the refactor: concurrent execution must not
    silently drop any of the three independent writes."""
    provider = ScriptedProvider([_check_work_intent_json()])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()

    await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="sess-planner-3",
        student_id="student-planner-3",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
    )

    mastery = await memory_store.get_mastery("student-planner-3", CHAIN_RULE_TOPIC)
    review = await review_store.get("student-planner-3", CHAIN_RULE_TOPIC)
    assert mastery is not None and mastery.attempts_total == 1
    assert review is not None and review.reps == 1
