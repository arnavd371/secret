"""
Integration coverage for Phase 13: through the real handle_turn()
orchestrator, a CHALLENGE turn on a topic with a known template still
uses the parametric generator (no extra model call), while a topic with
no known template tries the LLM-authored variant path for real.
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore, ProblemSessionState


async def _seed_high_mastery_first_attempt(store, session_id, problem_id):
    """The mastery-shortcut branch requires attempt_count == 1 exactly
    (spec §1.5 pseudocode) — seed the ladder state accordingly, same
    convention as tests/test_integration_challenge_generation.py."""
    await store.save(ProblemSessionState(session_id=session_id, problem_id=problem_id, attempt_count=1, hint_ladder_level=0))


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


def _intent_json(topic_hint) -> str:
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


@pytest.mark.asyncio
async def test_challenge_on_a_known_topic_never_tries_the_llm_variant_path():
    provider = ScriptedProvider(
        [_intent_json("calculus.differentiation.chain_rule"), "Here's a harder one to try."]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    await _seed_high_mastery_first_attempt(store, "s1", "problem-1")

    result = await handle_turn(
        "I solved it correctly!",
        session_id="s1",
        student_id="stu-1",
        problem_id="problem-1",
        router=router,
        session_store=store,
        mastery_estimate=0.99,
    )

    assert result.decision_action.action_type.value == "challenge"
    assert result.generated_item is not None
    assert result.generated_item.generation_mode == "parametric"
    # Only intent-classify + tutor-generate; no item_variant_author call.
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_challenge_on_an_unknown_topic_tries_the_llm_variant_first():
    variant_response = json.dumps(
        {
            "stem": "Find the derivative of 5x^4 - 2x.",
            "operation": "differentiate",
            "expression": "5*x**4 - 2*x",
            "variable": "x",
            "claimed_answer": "20*x**3 - 2",
        }
    )
    provider = ScriptedProvider(
        [
            _intent_json("statistics.normal_distribution"),  # no known template
            variant_response,  # item_variant_author call
            "Here's a harder one to try.",  # tutor generate
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    await _seed_high_mastery_first_attempt(store, "s2", "problem-2")

    result = await handle_turn(
        "I solved it correctly!",
        session_id="s2",
        student_id="stu-2",
        problem_id="problem-2",
        router=router,
        session_store=store,
        mastery_estimate=0.99,
    )

    assert result.decision_action.action_type.value == "challenge"
    assert result.generated_item is not None
    assert result.generated_item.generation_mode == "llm_variant"
    assert result.generated_item.correct_answer.value == "20*x**3 - 2"
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_llm_variant_failure_falls_back_to_the_default_template():
    provider = ScriptedProvider(
        [
            _intent_json("statistics.normal_distribution"),
            "not valid json",  # item_variant_author attempt 1: rejected
            "still not valid json",  # attempt 2: rejected too
            "Here's a harder one to try.",  # tutor generate, using the template fallback
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    await _seed_high_mastery_first_attempt(store, "s3", "problem-3")

    result = await handle_turn(
        "I solved it correctly!",
        session_id="s3",
        student_id="stu-3",
        problem_id="problem-3",
        router=router,
        session_store=store,
        mastery_estimate=0.99,
    )

    assert result.generated_item is not None
    assert result.generated_item.generation_mode == "parametric"
