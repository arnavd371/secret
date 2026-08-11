"""
Integration coverage for Phase 3's CHALLENGE -> Question Generation Engine
wiring, through the real orchestrator (mocked model responses, real item
generation underneath — no network calls).
"""

from __future__ import annotations

import json

import pytest

from app.cas.solver import verify_claim
from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.models.contracts import ActionType
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore, ProblemSessionState


class ScriptedProvider(ProviderClient):
    """Phase 6's Verifier/Critic makes an additional, independent model
    call per turn on the same shared Provider.ANTHROPIC queue. These
    tests aren't exercising critic behavior, so a critic-shaped system
    prompt is auto-passed without consuming a slot in the scripted queue."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.ANTHROPIC)
        self.calls.append({"model": spec.model, "system": system, "user": user})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


def _intent_json(topic_hint: str | None = None) -> str:
    return json.dumps(
        {
            "intent": "solve_request",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": topic_hint,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


async def _seed_high_mastery_first_attempt(store, session_id, problem_id):
    """The mastery-shortcut branch requires attempt_count == 1 exactly
    (spec §1.5 pseudocode) — seed the ladder state accordingly."""
    await store.save(ProblemSessionState(session_id=session_id, problem_id=problem_id, attempt_count=1, hint_ladder_level=0))


@pytest.mark.asyncio
async def test_high_mastery_turn_gets_a_real_verified_extension_item():
    provider = ScriptedProvider(
        [
            _intent_json(topic_hint="calculus.differentiation.chain_rule"),
            "Great work! Here's a tougher one for you to attempt on your own.",
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    await _seed_high_mastery_first_attempt(store, "sess-challenge-1", "problem-1")

    result = await handle_turn(
        "I think I solved it correctly",
        session_id="sess-challenge-1",
        student_id="student-1",
        problem_id="problem-1",
        router=router,
        session_store=store,
        mastery_estimate=0.95,
    )

    assert result.decision_action.action_type == ActionType.CHALLENGE
    assert result.generated_item is not None
    assert result.generated_item.template_id == "AA.SL.CALC.DIFF.CHAIN.T003"
    assert result.generated_item.correct_answer.cas_verified is True
    # The generated item's answer is independently re-checkable by CAS.
    from app.cas.solver import run_cas_operation
    from app.questions.templates import TEMPLATE_BANK

    template = TEMPLATE_BANK[result.generated_item.template_id]
    expression = template.expression_template.format(**result.generated_item.sampled_parameters)
    recomputed = run_cas_operation(template.operation, expression, template.variable)
    assert verify_claim(recomputed, result.generated_item.correct_answer.value)


@pytest.mark.asyncio
async def test_challenge_response_never_leaks_the_generated_items_answer(monkeypatch):
    """Even if the mocked model tries to leak the answer, the final
    response served to the student must never contain it. The generated
    item is pinned via monkeypatch so the scripted "leaking" draft can
    deterministically reference its real answer."""
    from app.questions.generator import generate_item

    fixed_item = generate_item("AA.SL.CALC.DIFF.CHAIN.T003", seed=99)

    async def _fake_generate_item_async(template_id, **kwargs):
        return fixed_item

    monkeypatch.setattr("app.orchestrator.handle_turn.generate_item_async", _fake_generate_item_async)

    leaking_draft = f"Try this: {fixed_item.rendered_stem} The answer is {fixed_item.correct_answer.value}."
    provider = ScriptedProvider([_intent_json(topic_hint="calculus.differentiation.chain_rule"), leaking_draft])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    await _seed_high_mastery_first_attempt(store, "sess-challenge-2", "problem-1")

    result = await handle_turn(
        "I think I solved it correctly",
        session_id="sess-challenge-2",
        student_id="student-2",
        problem_id="problem-1",
        router=router,
        session_store=store,
        mastery_estimate=0.95,
    )

    assert result.decision_action.action_type == ActionType.CHALLENGE
    assert result.generated_item == fixed_item
    assert result.final_response.ui_metadata["templated"] is True
    assert fixed_item.correct_answer.value not in result.final_response.text
