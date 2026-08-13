"""
Integration coverage for Phase 21: through the real handle_turn()
orchestrator, both a normal approved turn and a degraded-critic turn
write a real record to the guardrail metrics store, and that record
aggregates correctly - proving the wiring, not just the pure functions
in isolation (tests/test_guardrail_metrics.py already covers those).
"""

from __future__ import annotations

import json

import pytest

from app.guardrail_metrics.aggregate import compute_guardrail_metrics
from app.guardrail_metrics.store import InMemoryGuardrailMetricsStore
from app.llm.client import LLMCallResult, ModelRouter, ModelUnavailableError, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore


class ScriptedProvider(ProviderClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.GROQ)
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


class _CriticAlwaysFailsProvider(ProviderClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            raise ModelUnavailableError("simulated critic outage")
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


def _concept_explain_intent_json() -> str:
    return json.dumps(
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


@pytest.mark.asyncio
async def test_a_normal_approved_turn_writes_a_real_non_fallback_record():
    provider = ScriptedProvider([_concept_explain_intent_json(), "The chain rule handles composite functions."])
    router = ModelRouter(providers={Provider.GROQ: provider})
    metrics_store = InMemoryGuardrailMetricsStore()

    result = await handle_turn(
        "can you remind me how the chain rule works?",
        session_id="s1",
        student_id="stu-1",
        router=router,
        session_store=InMemorySessionStateStore(),
        guardrail_metrics_store=metrics_store,
    )

    records = await metrics_store.get_all()
    assert len(records) == 1
    assert records[0].turn_id == result.turn_id
    assert records[0].fell_back_to_template is False
    assert records[0].critic_verdict == "pass"
    assert records[0].leak_check_triggered is False


@pytest.mark.asyncio
async def test_a_degraded_critic_turn_is_recorded_as_critic_degraded():
    provider = _CriticAlwaysFailsProvider([_concept_explain_intent_json(), "The chain rule handles composite functions."])
    router = ModelRouter(providers={Provider.GROQ: provider})
    metrics_store = InMemoryGuardrailMetricsStore()

    await handle_turn(
        "can you remind me how the chain rule works?",
        session_id="s2",
        student_id="stu-2",
        router=router,
        session_store=InMemorySessionStateStore(),
        guardrail_metrics_store=metrics_store,
    )

    records = await metrics_store.get_all()
    assert len(records) == 1
    assert records[0].critic_degraded is True


@pytest.mark.asyncio
async def test_default_guardrail_metrics_store_is_used_when_none_is_passed():
    """A caller that doesn't pass a store explicitly still gets real
    recording - the process-wide singleton, same convention as every
    other store in this codebase."""
    from app.guardrail_metrics.store import get_default_guardrail_metrics_store

    provider = ScriptedProvider([_concept_explain_intent_json(), "The chain rule handles composite functions."])
    router = ModelRouter(providers={Provider.GROQ: provider})
    before = len(await get_default_guardrail_metrics_store().get_all())

    await handle_turn(
        "can you remind me how the chain rule works?",
        session_id="s3",
        student_id="stu-3",
        router=router,
        session_store=InMemorySessionStateStore(),
    )

    after = await get_default_guardrail_metrics_store().get_all()
    assert len(after) == before + 1


@pytest.mark.asyncio
async def test_metrics_recorded_across_multiple_real_turns_aggregate_correctly():
    metrics_store = InMemoryGuardrailMetricsStore()

    normal_provider = ScriptedProvider([_concept_explain_intent_json(), "The chain rule handles composite functions."])
    normal_router = ModelRouter(providers={Provider.GROQ: normal_provider})
    await handle_turn(
        "can you remind me how the chain rule works?",
        session_id="s4",
        student_id="stu-4",
        router=normal_router,
        session_store=InMemorySessionStateStore(),
        guardrail_metrics_store=metrics_store,
    )

    degraded_provider = _CriticAlwaysFailsProvider(
        [_concept_explain_intent_json(), "The chain rule handles composite functions."]
    )
    degraded_router = ModelRouter(providers={Provider.GROQ: degraded_provider})
    await handle_turn(
        "can you remind me how the chain rule works?",
        session_id="s5",
        student_id="stu-5",
        router=degraded_router,
        session_store=InMemorySessionStateStore(),
        guardrail_metrics_store=metrics_store,
    )

    report = compute_guardrail_metrics(await metrics_store.get_all())
    assert report.total_turns == 2
    assert report.critic_degraded_rate == 0.5
