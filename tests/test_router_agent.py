import json

import pytest

from app.agents.router_agent import LOW_CONFIDENCE_THRESHOLD, classify_intent
from app.llm.client import ModelRouter, ModelUnavailableError, MockProvider
from app.llm.router_config import Provider
from app.models.contracts import AssessmentMode, IntentType


def _router_with_json_response(payload: dict) -> ModelRouter:
    return ModelRouter(providers={Provider.GROQ: MockProvider(canned_response=json.dumps(payload))})


class _AlwaysFailsProvider:
    async def generate(self, *, spec, system, user, images=None):
        raise ModelUnavailableError("simulated outage")


@pytest.mark.asyncio
async def test_classify_intent_parses_valid_high_confidence_response():
    router = _router_with_json_response(
        {
            "intent": "solve_request",
            "confidence": 0.92,
            "subject": "math_aa",
            "topic_hint": "calculus.differentiation.chain_rule",
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )
    result = await classify_intent("can you help me with this derivative problem?", router)
    assert result.intent == IntentType.SOLVE_REQUEST
    assert result.confidence == 0.92
    assert result.topic_hint == "calculus.differentiation.chain_rule"


@pytest.mark.asyncio
async def test_classify_intent_low_confidence_falls_back_to_safe_default():
    router = _router_with_json_response(
        {
            "intent": "exam_prep",
            "confidence": 0.3,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "graded_take_home",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )
    result = await classify_intent("uh, do this thing", router)
    assert result.intent == IntentType.CONCEPT_EXPLAIN
    assert result.assessment_mode_guess == AssessmentMode.PRACTICE
    assert result.confidence == 0.3  # preserved for observability


@pytest.mark.asyncio
async def test_classify_intent_falls_back_on_unparseable_response():
    router = ModelRouter(providers={Provider.GROQ: MockProvider(canned_response="not json at all")})
    result = await classify_intent("hello", router)
    assert result.intent == IntentType.CONCEPT_EXPLAIN
    assert result.confidence == LOW_CONFIDENCE_THRESHOLD


@pytest.mark.asyncio
async def test_classify_intent_falls_back_when_provider_unavailable():
    router = ModelRouter(providers={Provider.GROQ: _AlwaysFailsProvider()})
    result = await classify_intent("hello", router)
    assert result.intent == IntentType.CONCEPT_EXPLAIN
