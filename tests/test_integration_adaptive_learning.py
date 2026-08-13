"""
Integration coverage for Phase 9's Adaptive Learning Engine, through the
real handle_turn() orchestrator: a graded check_work submission updates
real FSRS review state, and a later exam_prep turn picks up that real
due-review queue to bind a genuine generated item to a QUESTION action —
or, when nothing is due, falls back to a plain explanation instead of
manufacturing a review out of nothing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.adaptive.scheduler import record_review
from app.adaptive.store import InMemoryReviewStateStore
from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.memory.store import InMemoryMemoryStore
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore

CHAIN_RULE_TOPIC = "calculus.differentiation.chain_rule"


class ScriptedProvider(ProviderClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.GROQ)
        self.calls.append({"model": spec.model, "system": system, "user": user})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


def _intent_json(intent: str, topic_hint=None) -> str:
    return json.dumps(
        {
            "intent": intent,
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": topic_hint,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.mark.asyncio
async def test_exam_prep_with_a_due_review_binds_a_real_generated_item():
    provider = ScriptedProvider(
        [_intent_json("exam_prep"), "This one's due for review: Find the derivative. Try it from memory."]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()

    # Seed a review far enough in the past that it's already overdue by
    # the time handle_turn runs with the real current time.
    long_ago = datetime.now(timezone.utc) - timedelta(days=365)
    await record_review(review_store, "stu-1", CHAIN_RULE_TOPIC, True, now=long_ago)

    result = await handle_turn(
        "help me review for my exam",
        session_id="sess-adapt-1",
        student_id="stu-1",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
    )

    assert result.decision_action.action_type.value == "question"
    assert result.decision_action.move == "retrieval_practice"
    assert result.decision_action.reason == "exam_prep_review_due"
    assert result.generated_item is not None
    assert result.generated_item.correct_answer.value not in result.final_response.text


@pytest.mark.asyncio
async def test_exam_prep_with_nothing_due_falls_back_to_explain():
    provider = ScriptedProvider(
        [_intent_json("exam_prep"), "Sure, what would you like to go over?"]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()  # empty: nothing has ever been reviewed

    result = await handle_turn(
        "help me review for my exam",
        session_id="sess-adapt-2",
        student_id="stu-2",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
    )

    assert result.decision_action.action_type.value == "explain"
    assert result.decision_action.move == "general_response"
    assert result.decision_action.reason == "exam_prep_no_review_due"
    assert result.generated_item is None


@pytest.mark.asyncio
async def test_graded_check_work_updates_real_fsrs_state():
    provider = ScriptedProvider([_intent_json("check_work", topic_hint=CHAIN_RULE_TOPIC)])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()

    result = await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="sess-adapt-3",
        student_id="stu-3",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
    )

    assert result.mark_result.total_awarded == result.mark_result.total_available

    review_state = await review_store.get("stu-3", CHAIN_RULE_TOPIC)
    assert review_state is not None
    assert review_state.reps == 1
    assert review_state.last_reviewed_at is not None
    assert review_state.due_at > review_state.last_reviewed_at


@pytest.mark.asyncio
async def test_incorrect_grading_schedules_a_sooner_review_than_a_correct_one():
    provider = ScriptedProvider(
        [
            _intent_json("check_work", topic_hint=CHAIN_RULE_TOPIC),
            _intent_json("check_work", topic_hint="calculus.differentiation.product_rule"),
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()

    await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="sess-adapt-4",
        student_id="stu-4",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",  # correct
    )
    await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-adapt-4",
        student_id="stu-4",
        router=router,
        session_store=store,
        memory_store=memory_store,
        review_store=review_store,
        # f'(x)g'(x) instead of the product rule: wrong, and pattern-matched
        # for real (no extra model call needed for the diagnosis this
        # triggers).
        student_work="u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)",
    )

    good_state = await review_store.get("stu-4", CHAIN_RULE_TOPIC)
    bad_state = await review_store.get("stu-4", "calculus.differentiation.product_rule")
    assert bad_state.due_at < good_state.due_at
