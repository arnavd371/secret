"""
Integration coverage for Phase 2's CAS verification + retrieval, wired
through the real orchestrator (mocked model responses, real SymPy and
real lexical retrieval — no network calls).
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.models.contracts import ActionType
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


def _intent_json(intent: str, topic_hint: str | None = None) -> str:
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
async def test_concept_explain_turn_is_grounded_by_real_cas_and_retrieval():
    provider = ScriptedProvider(
        [
            _intent_json("concept_explain", topic_hint="calculus.differentiation.chain_rule"),
            "Using the power rule, the derivative is 2*x.",
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "differentiate x^2",
        session_id="sess-cas-1",
        student_id="student-1",
        router=router,
        session_store=store,
    )

    assert result.decision_action.action_type == ActionType.EXPLAIN
    # A real math task was extracted and verified by SymPy.
    assert result.cas_result is not None
    assert result.cas_result.status.value == "ok"
    # Real retrieval found the chain rule grounding via the topic_hint.
    assert result.retrieved_chunks
    assert any(c.subtopic_id == "calculus.differentiation.chain_rule" for c in result.retrieved_chunks)
    # The draft agreed with CAS, so it passed through with real citations.
    assert result.final_response.ui_metadata["templated"] is False
    assert result.final_response.citations


@pytest.mark.asyncio
async def test_concept_explain_turn_with_wrong_draft_is_corrected_by_cas():
    provider = ScriptedProvider(
        [
            _intent_json("concept_explain"),
            "Using the power rule, the answer is 5*x.",  # wrong
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "differentiate x^2 for me",
        session_id="sess-cas-2",
        student_id="student-2",
        router=router,
        session_store=store,
    )

    assert result.cas_result is not None
    assert result.cas_result.status.value == "ok"
    assert result.final_response.ui_metadata.get("cas_grounded") is True
    assert "5*x" not in result.final_response.text
    assert "2*x" in result.final_response.text


@pytest.mark.asyncio
async def test_question_action_does_not_run_cas_or_retrieval():
    """solve_request on a fresh attempt yields QUESTION, which never
    states an answer — CAS/retrieval should not run at all for it."""
    provider = ScriptedProvider(
        [
            _intent_json("solve_request"),
            "What do you think the first step is?",
        ]
    )
    router = ModelRouter(providers={Provider.GROQ: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "differentiate x^2, I'm stuck",
        session_id="sess-cas-3",
        student_id="student-3",
        problem_id="problem-1",
        router=router,
        session_store=store,
    )

    assert result.decision_action.action_type == ActionType.QUESTION
    assert result.cas_result is None
    assert result.retrieved_chunks is None
