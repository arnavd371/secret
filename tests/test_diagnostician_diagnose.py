"""
Tests for the two-tier orchestration: pattern detectors run first (and
must never touch the model router when they find a match), the model
fallback only runs when nothing matched.
"""

import json

import pytest

from app.cas.extraction import MathTask
from app.cas.models import CASOperation, CASResult, CASStatus
from app.diagnostician.diagnose import diagnose_misconception
from app.diagnostician.models import DiagnosisMethod
from app.llm.client import ModelRouter, ModelUnavailableError, MockProvider
from app.llm.router_config import Provider


class _RaisesIfCalledProvider:
    async def generate(self, *, spec, system, user, images=None):
        raise AssertionError("router must not be called when a pattern match is found")


def _cas_result(operation: CASOperation, result_exact: str) -> CASResult:
    return CASResult(status=CASStatus.OK, operation=operation, input_latex="", result_exact=result_exact)


@pytest.mark.asyncio
async def test_pattern_match_short_circuits_before_any_model_call():
    router = ModelRouter(providers={Provider.GROQ: _RaisesIfCalledProvider()})
    task = MathTask(operation=CASOperation.DIFFERENTIATE, expression="(2*x + 1)**5", variable="x")
    cas_result = _cas_result(CASOperation.DIFFERENTIATE, "10*(2*x + 1)**4")

    result = await diagnose_misconception(
        router, task, cas_result, "u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4"
    )

    assert result.misconception_id == "MISC-CALC-014"
    assert result.method == DiagnosisMethod.PATTERN_MATCH
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_falls_back_to_model_when_no_pattern_matches():
    canned = json.dumps({"misconception_id": "MISC-CALC-010", "confidence": 0.75, "evidence": "matches f'g' shape"})
    router = ModelRouter(providers={Provider.GROQ: MockProvider(canned_response=canned)})
    task = MathTask(operation=CASOperation.DIFFERENTIATE, expression="x**2 * sin(x)", variable="x")
    cas_result = _cas_result(CASOperation.DIFFERENTIATE, "2*x*sin(x) + x**2*cos(x)")

    result = await diagnose_misconception(router, task, cas_result, "therefore dy/dx = 999")

    assert result.method == DiagnosisMethod.MODEL_INFERENCE
    assert result.misconception_id == "MISC-CALC-010"


@pytest.mark.asyncio
async def test_unverifiable_cas_result_skips_diagnosis_entirely():
    router = ModelRouter(providers={Provider.GROQ: _RaisesIfCalledProvider()})
    task = MathTask(operation=CASOperation.DIFFERENTIATE, expression="garbage(((", variable="x")
    cas_result = CASResult(status=CASStatus.UNVERIFIABLE, operation=CASOperation.DIFFERENTIATE, input_latex="")

    result = await diagnose_misconception(router, task, cas_result, "therefore dy/dx = 999")

    assert result.misconception_id is None
    assert result.method is None


@pytest.mark.asyncio
async def test_no_final_answer_in_submission_skips_diagnosis_without_calling_model():
    router = ModelRouter(providers={Provider.GROQ: _RaisesIfCalledProvider()})
    task = MathTask(operation=CASOperation.DIFFERENTIATE, expression="x**2 * sin(x)", variable="x")
    cas_result = _cas_result(CASOperation.DIFFERENTIATE, "2*x*sin(x) + x**2*cos(x)")

    result = await diagnose_misconception(router, task, cas_result, "I'm not sure how to start")

    assert result.misconception_id is None
    assert result.method is None
