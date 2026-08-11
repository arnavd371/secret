"""
Tests for the model-inference fallback: same MockProvider pattern as
every other router-calling agent, no real network call or API key.
"""

import json

import pytest

from app.diagnostician.model_fallback import diagnose_via_model
from app.diagnostician.models import DiagnosisMethod
from app.llm.client import ModelRouter, ModelUnavailableError, MockProvider
from app.llm.router_config import Provider


def _router_with_canned_response(text: str) -> ModelRouter:
    return ModelRouter(providers={Provider.ANTHROPIC: MockProvider(canned_response=text)})


@pytest.mark.asyncio
async def test_valid_response_with_known_id_is_parsed():
    canned = json.dumps({"misconception_id": "MISC-CALC-010", "confidence": 0.8, "evidence": "looks like f'g'"})
    router = _router_with_canned_response(canned)
    result = await diagnose_via_model(router, "differentiate", "x**2*sin(x)", "x**2*cos(x)+2*x*sin(x)", "2*x*cos(x)")
    assert result.misconception_id == "MISC-CALC-010"
    assert result.confidence == 0.8
    assert result.method == DiagnosisMethod.MODEL_INFERENCE


@pytest.mark.asyncio
async def test_null_misconception_id_is_a_valid_no_diagnosis_result():
    canned = json.dumps({"misconception_id": None, "confidence": 0.0, "evidence": "looks like an arithmetic slip"})
    router = _router_with_canned_response(canned)
    result = await diagnose_via_model(router, "differentiate", "x**2", "2*x", "3*x")
    assert result.misconception_id is None
    assert result.method is None


@pytest.mark.asyncio
async def test_unknown_misconception_id_is_rejected_not_trusted():
    canned = json.dumps({"misconception_id": "MISC-MADE-UP-999", "confidence": 0.9, "evidence": "..."})
    router = _router_with_canned_response(canned)
    result = await diagnose_via_model(router, "differentiate", "x**2", "2*x", "3*x")
    assert result.misconception_id is None


@pytest.mark.asyncio
async def test_confidence_is_clamped_to_valid_range():
    canned = json.dumps({"misconception_id": "MISC-ALG-003", "confidence": 5.0, "evidence": "..."})
    router = _router_with_canned_response(canned)
    result = await diagnose_via_model(router, "solve", "x**2-5*x+6=0", "x=2, x=3", "x=-2, x=-3")
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_unparseable_response_degrades_to_no_diagnosis():
    router = _router_with_canned_response("not valid json at all")
    result = await diagnose_via_model(router, "differentiate", "x**2", "2*x", "3*x")
    assert result.misconception_id is None
    assert result.method is None
    assert "unavailable" in result.evidence


class _AlwaysFailsProvider:
    async def generate(self, *, spec, system, user, images=None):
        raise ModelUnavailableError("simulated diagnostician outage")


@pytest.mark.asyncio
async def test_call_failure_degrades_to_no_diagnosis_instead_of_raising():
    router = ModelRouter(providers={Provider.ANTHROPIC: _AlwaysFailsProvider()})
    result = await diagnose_via_model(router, "differentiate", "x**2", "2*x", "3*x")
    assert result.misconception_id is None
    assert "unavailable" in result.evidence
