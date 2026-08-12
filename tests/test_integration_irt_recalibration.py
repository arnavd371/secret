"""
Integration coverage for Phase 14: through the real handle_turn()
orchestrator, a check_work grading tagged with responding_to_template_id
writes a real response record, and once enough real history exists a
later CHALLENGE turn on the same template gets a recalibrated
difficulty instead of the hand-set template prior.
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.questions.response_log import InMemoryResponseLogStore
from app.session.state import InMemorySessionStateStore, ProblemSessionState

CHAIN_RULE_TEMPLATE = "AA.SL.CALC.DIFF.CHAIN.T003"
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


def _check_work_intent_json() -> str:
    return json.dumps(
        {
            "intent": "check_work",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": CHAIN_RULE_TOPIC,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.mark.asyncio
async def test_graded_response_tagged_with_a_template_id_is_logged():
    provider = ScriptedProvider([_check_work_intent_json()])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    response_log_store = InMemoryResponseLogStore()

    await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="s1",
        student_id="stu-1",
        router=router,
        session_store=store,
        response_log_store=response_log_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
        responding_to_template_id=CHAIN_RULE_TEMPLATE,
    )

    records = await response_log_store.get_all(CHAIN_RULE_TEMPLATE)
    assert len(records) == 1
    assert records[0].correct is True
    assert records[0].student_id == "stu-1"


@pytest.mark.asyncio
async def test_no_response_logged_without_responding_to_template_id():
    provider = ScriptedProvider([_check_work_intent_json()])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    response_log_store = InMemoryResponseLogStore()

    await handle_turn(
        "can you check my work? differentiate (2*x+1)**5",
        session_id="s2",
        student_id="stu-2",
        router=router,
        session_store=store,
        response_log_store=response_log_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
    )

    assert await response_log_store.get_all(CHAIN_RULE_TEMPLATE) == []


@pytest.mark.asyncio
async def test_enough_real_history_recalibrates_a_later_challenge_item():
    provider = ScriptedProvider(
        [_check_work_intent_json() for _ in range(10)]
        + [json.dumps({"intent": "solve_request", "confidence": 0.9, "subject": "math_aa", "topic_hint": CHAIN_RULE_TOPIC, "assessment_mode_guess": "practice", "requires_multimodal_parse": False, "language": "en"})]
        + ["Great work! Here's a tougher one for you."]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    response_log_store = InMemoryResponseLogStore()

    # 10 correct submissions, all tagged to the chain-rule template.
    for i in range(10):
        await handle_turn(
            "can you check my work? differentiate (2*x+1)**5",
            session_id="s3",
            student_id=f"stu-hist-{i}",
            router=router,
            session_store=store,
            response_log_store=response_log_store,
            student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
            responding_to_template_id=CHAIN_RULE_TEMPLATE,
        )

    await store.save(ProblemSessionState(session_id="s3", problem_id="p-final", attempt_count=1, hint_ladder_level=0))
    result = await handle_turn(
        "I solved it correctly!",
        session_id="s3",
        student_id="stu-final",
        problem_id="p-final",
        router=router,
        session_store=store,
        response_log_store=response_log_store,
        mastery_estimate=0.99,
    )

    assert result.generated_item.template_id == CHAIN_RULE_TEMPLATE
    assert result.generated_item.difficulty_estimate.source == "recalibrated"
    assert result.generated_item.difficulty_estimate.b_param < 0  # all-correct history -> easy
