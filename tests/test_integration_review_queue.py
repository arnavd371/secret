"""
Integration coverage for Phase 17: through the real handle_turn()
orchestrator, a low-confidence/flagged grading and a degraded critic
call both write real entries to the human review queue - signals every
earlier phase already computed for real, now actually surfaced somewhere.
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ModelUnavailableError, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.review_queue.models import ReviewReason
from app.review_queue.store import InMemoryReviewQueueStore
from app.session.state import InMemorySessionStateStore


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


class _CriticAlwaysFailsProvider(ProviderClient):
    """Every non-critic call is scripted normally; the critic call always
    raises, forcing the real degraded-fallback path (Phase 6)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            raise ModelUnavailableError("simulated critic outage")
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


def _check_work_intent_json(topic_hint=None) -> str:
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
async def test_a_grading_with_no_final_answer_is_queued_for_review():
    provider = ScriptedProvider([_check_work_intent_json()])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    review_queue_store = InMemoryReviewQueueStore()

    result = await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="s1",
        student_id="stu-1",
        router=router,
        session_store=store,
        review_queue_store=review_queue_store,
        student_work="I'm not really sure how to start this one",  # no final answer at all
    )

    assert result.mark_result is not None
    pending = await review_queue_store.list_pending("stu-1")
    assert len(pending) == 1
    assert pending[0].reason == ReviewReason.LOW_CONFIDENCE_GRADING


@pytest.mark.asyncio
async def test_a_fully_correct_well_supported_grading_is_never_queued():
    provider = ScriptedProvider([_check_work_intent_json()])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    review_queue_store = InMemoryReviewQueueStore()

    await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="s2",
        student_id="stu-2",
        router=router,
        session_store=store,
        review_queue_store=review_queue_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
    )

    assert await review_queue_store.list_pending("stu-2") == []


@pytest.mark.asyncio
async def test_a_degraded_critic_call_is_queued_for_review():
    intent_json = json.dumps(
        {
            "intent": "concept_explain",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )
    provider = _CriticAlwaysFailsProvider([intent_json, "The chain rule handles composite functions."])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    review_queue_store = InMemoryReviewQueueStore()

    result = await handle_turn(
        "can you remind me how the chain rule works?",
        session_id="s3",
        student_id="stu-3",
        router=router,
        session_store=store,
        review_queue_store=review_queue_store,
    )

    assert result.final_response.ui_metadata["critic_degraded"] is True
    pending = await review_queue_store.list_pending("stu-3")
    assert len(pending) == 1
    assert pending[0].reason == ReviewReason.CRITIC_DEGRADED
