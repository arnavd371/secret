"""
Integration coverage for Phase 8's misconception diagnosis, through the
real handle_turn() orchestrator: a wrong check_work submission gets
diagnosed and written to the real misconception registry (Phase 5), and
a later turn's real memory context assembly picks it up automatically,
closing the loop Phase 5 explicitly left open.
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.memory.store import InMemoryMemoryStore
from app.orchestrator.handle_turn import handle_turn
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


def _check_work_intent_json(topic_hint: str = "calculus.differentiation.product_rule") -> str:
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
async def test_pattern_matched_misconception_is_written_and_surfaces_next_turn():
    provider = ScriptedProvider([_check_work_intent_json(), _check_work_intent_json()])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    turn1 = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-diag-1",
        student_id="student-diag-1",
        router=router,
        session_store=store,
        memory_store=memory_store,
        student_work="u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)",  # f'g', not the product rule
    )

    assert turn1.diagnosis_result is not None
    assert turn1.diagnosis_result.misconception_id == "MISC-CALC-010"
    assert turn1.diagnosis_result.method.value == "pattern_match"
    assert "recognizable error" in turn1.final_response.text
    # No model call was needed for a pattern match.
    assert len(provider.calls) == 1

    misconceptions = await memory_store.get_misconceptions("student-diag-1")
    assert len(misconceptions) == 1
    assert misconceptions[0].misconception_id == "MISC-CALC-010"
    assert misconceptions[0].occurrences == 1

    turn2 = await handle_turn(
        "can you check my work? differentiate x**2 * cos(x)",
        session_id="sess-diag-1",
        student_id="student-diag-1",
        router=router,
        session_store=store,
        memory_store=memory_store,
        # Correct AND fully worked (an intermediate step shown, so the
        # method mark is awarded too) — this turn must grade as fully
        # correct so no second diagnosis call is needed; it exists only
        # to prove the first turn's misconception carries into this
        # turn's memory context.
        student_work="u = x**2, v = cos(x)\ntherefore dy/dx = -x**2*sin(x) + 2*x*cos(x)",
    )

    assert "MISC-CALC-010" in turn2.student_state_snapshot.active_misconception_ids
    assert "MISC-CALC-010" in turn2.student_state_snapshot.rendered_text


@pytest.mark.asyncio
async def test_correct_submission_never_runs_diagnosis():
    provider = ScriptedProvider([_check_work_intent_json()])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-diag-2",
        student_id="student-diag-2",
        router=router,
        session_store=store,
        memory_store=memory_store,
        student_work="u = x**2, v = sin(x)\nu_prime = 2*x\nv_prime = cos(x)\ntherefore dy/dx = 2*x*sin(x) + x**2*cos(x)",
    )

    assert result.mark_result.total_awarded == result.mark_result.total_available
    assert result.diagnosis_result is None
    misconceptions = await memory_store.get_misconceptions("student-diag-2")
    assert misconceptions == []


@pytest.mark.asyncio
async def test_unmatched_wrong_answer_falls_back_to_model_and_is_written_when_confident():
    diagnosis_response = json.dumps(
        {"misconception_id": "MISC-CALC-010", "confidence": 0.9, "evidence": "resembles f'g' with a sign slip"}
    )
    provider = ScriptedProvider([_check_work_intent_json(), diagnosis_response])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-diag-3",
        student_id="student-diag-3",
        router=router,
        session_store=store,
        memory_store=memory_store,
        student_work="therefore dy/dx = 12345",  # matches no known pattern at all
    )

    assert result.diagnosis_result is not None
    assert result.diagnosis_result.method.value == "model_inference"
    assert len(provider.calls) == 2  # intent classify + misconception_diagnose

    misconceptions = await memory_store.get_misconceptions("student-diag-3")
    assert len(misconceptions) == 1


@pytest.mark.asyncio
async def test_repeat_diagnosis_increments_occurrences_not_duplicate_entries():
    provider = ScriptedProvider([_check_work_intent_json(), _check_work_intent_json()])
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    work = "u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)"
    for _ in range(2):
        await handle_turn(
            "can you check my work? differentiate x**2 * sin(x)",
            session_id="sess-diag-4",
            student_id="student-diag-4",
            router=router,
            session_store=store,
            memory_store=memory_store,
            student_work=work,
        )

    misconceptions = await memory_store.get_misconceptions("student-diag-4")
    assert len(misconceptions) == 1
    assert misconceptions[0].occurrences == 2
