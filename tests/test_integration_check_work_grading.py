"""
Integration coverage for Phase 4's check_work -> real grading wiring,
through the orchestrator (mocked model responses; real CAS + grading
underneath, no network calls).
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore


class ScriptedProvider(ProviderClient):
    """Phase 6's Verifier/Critic makes an additional, independent model
    call per turn on the same shared Provider.GROQ queue. These
    tests aren't exercising critic behavior, so a critic-shaped system
    prompt is auto-passed without consuming a slot in the scripted queue."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.GROQ)
        self.calls.append({"model": spec.model, "system": system, "user": user})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


def _check_work_intent_json() -> str:
    return json.dumps(
        {
            "intent": "check_work",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.mark.asyncio
async def test_check_work_with_student_work_bypasses_tutor_llm_and_grades_for_real():
    provider = ScriptedProvider(
        [
            _check_work_intent_json(),
            "THIS SHOULD NEVER BE CONSUMED — grading bypasses the Tutor LLM call entirely",
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    work = "u = x**2, v = sin(x)\nu_prime = 2*x\nv_prime = cos(x)\ntherefore dy/dx = 2*x*sin(x) + x**2*cos(x)"

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-grade-1",
        student_id="student-1",
        router=router,
        session_store=store,
        student_work=work,
    )

    assert result.mark_result is not None
    assert result.mark_result.total_awarded == result.mark_result.total_available == 2
    assert result.final_response.text == result.mark_result.comment
    assert "THIS SHOULD NEVER BE CONSUMED" not in result.final_response.text
    assert result.final_response.ui_metadata["graded"] is True
    # Only the intent-classification call happened; the Tutor generate
    # call was never made, since grading already produced the response.
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_check_work_without_student_work_falls_back_to_tutor_generated_explain():
    """Backward compatibility: omitting student_work must behave exactly
    as it did before Phase 4 — a normal Tutor-generated EXPLAIN turn."""
    provider = ScriptedProvider(
        [
            _check_work_intent_json(),
            "Let's look at where your working diverges from the correct method.",
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-grade-2",
        student_id="student-2",
        router=router,
        session_store=store,
    )

    assert result.mark_result is None
    assert result.final_response.ui_metadata.get("graded") is not True
    assert len(provider.calls) == 2  # intent classify + tutor generate, as before


@pytest.mark.asyncio
async def test_check_work_grades_incorrect_submission_and_localizes_error():
    provider = ScriptedProvider([_check_work_intent_json(), "unused"])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    work = "u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)"  # wrong final answer

    result = await handle_turn(
        "check my work: differentiate x**2 * sin(x)",
        session_id="sess-grade-3",
        student_id="student-3",
        router=router,
        session_store=store,
        student_work=work,
    )

    assert result.mark_result is not None
    assert result.mark_result.total_awarded < result.mark_result.total_available
    assert result.mark_result.first_error_step_index is not None
    assert len(provider.calls) == 1  # tutor generate never consumed


@pytest.mark.asyncio
async def test_check_work_falls_back_when_no_math_task_is_extractable():
    """student_work is present, but raw_input doesn't contain a checkable
    problem statement — grading can't proceed blind, so it falls back to
    the normal Tutor path rather than fabricating a mark scheme."""
    provider = ScriptedProvider(
        [_check_work_intent_json(), "Let's talk through what you're working on."]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "can you check my work on this problem?",  # no extractable operation/expression
        session_id="sess-grade-4",
        student_id="student-4",
        router=router,
        session_store=store,
        student_work="x = 2",
    )

    assert result.mark_result is None
    assert len(provider.calls) == 2
