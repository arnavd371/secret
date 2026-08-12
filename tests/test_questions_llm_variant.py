"""
Tests for LLM-authored item variants: the CAS-verification gate is the
entire point of this module, so most of these tests are specifically
about what happens when the LLM's claim does or doesn't hold up.
"""

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ModelUnavailableError, MockProvider, ProviderClient
from app.llm.router_config import Provider
from app.questions.llm_variant import generate_llm_variant


def _router_with_canned_response(text: str) -> ModelRouter:
    return ModelRouter(providers={Provider.ANTHROPIC: MockProvider(canned_response=text)})


def _variant_json(**overrides) -> str:
    payload = {
        "stem": "Find the derivative of 3x^2.",
        "operation": "differentiate",
        "expression": "3*x**2",
        "variable": "x",
        "claimed_answer": "6*x",
    }
    payload.update(overrides)
    return json.dumps(payload)


class _QueuedProvider(ProviderClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        self.calls.append({"system": system, "user": user})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


@pytest.mark.asyncio
async def test_a_correct_claimed_answer_produces_a_real_cas_verified_item():
    router = _router_with_canned_response(_variant_json())
    item = await generate_llm_variant("calculus.differentiation", router)

    assert item is not None
    assert item.generation_mode == "llm_variant"
    assert item.correct_answer.cas_verified is True
    assert item.correct_answer.value == "6*x"  # the CAS result, not blindly the LLM's own string
    assert item.rendered_stem == "Find the derivative of 3x^2."
    assert item.quality_gate_status == "PASSED"


@pytest.mark.asyncio
async def test_a_wrong_claimed_answer_is_never_served():
    router = _router_with_canned_response(_variant_json(claimed_answer="999"))
    item = await generate_llm_variant("calculus.differentiation", router, max_attempts=1)
    assert item is None


@pytest.mark.asyncio
async def test_correct_answer_uses_the_cas_value_even_when_claim_is_an_equivalent_but_different_form():
    # 6*x and 3*x*2 are the same value; the served answer should be the
    # CAS's own canonical form, not the LLM's (possibly differently
    # formatted) string.
    router = _router_with_canned_response(_variant_json(claimed_answer="2*x*3"))
    item = await generate_llm_variant("calculus.differentiation", router)
    assert item is not None
    assert item.correct_answer.value == "6*x"


@pytest.mark.asyncio
async def test_unparseable_expression_is_rejected():
    router = _router_with_canned_response(_variant_json(expression="not( valid math (("))
    item = await generate_llm_variant("calculus.differentiation", router, max_attempts=1)
    assert item is None


@pytest.mark.asyncio
async def test_malformed_json_response_is_rejected():
    router = _router_with_canned_response("this is not json at all")
    item = await generate_llm_variant("calculus.differentiation", router, max_attempts=1)
    assert item is None


@pytest.mark.asyncio
async def test_unknown_operation_value_is_rejected():
    router = _router_with_canned_response(_variant_json(operation="fabricate_an_answer"))
    item = await generate_llm_variant("calculus.differentiation", router, max_attempts=1)
    assert item is None


@pytest.mark.asyncio
async def test_missing_required_key_is_rejected():
    payload = json.loads(_variant_json())
    del payload["claimed_answer"]
    router = _router_with_canned_response(json.dumps(payload))
    item = await generate_llm_variant("calculus.differentiation", router, max_attempts=1)
    assert item is None


@pytest.mark.asyncio
async def test_a_bad_first_attempt_can_be_recovered_by_a_good_second_attempt():
    provider = _QueuedProvider([_variant_json(claimed_answer="999"), _variant_json()])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    item = await generate_llm_variant("calculus.differentiation", router, max_attempts=2)

    assert item is not None
    assert len(provider.calls) == 2
    # The correction from the failed first attempt was fed back in.
    assert "didn't match" in provider.calls[1]["user"]


@pytest.mark.asyncio
async def test_model_unavailable_returns_none_instead_of_raising():
    class _AlwaysFailsProvider:
        async def generate(self, *, spec, system, user, images=None):
            raise ModelUnavailableError("simulated outage")

    router = ModelRouter(providers={Provider.ANTHROPIC: _AlwaysFailsProvider()})
    item = await generate_llm_variant("calculus.differentiation", router)
    assert item is None


@pytest.mark.asyncio
async def test_llm_variant_item_has_no_distractors():
    """Documented, honest scope limit: no generic distractor generator
    exists for an arbitrary LLM-authored expression."""
    router = _router_with_canned_response(_variant_json())
    item = await generate_llm_variant("calculus.differentiation", router)
    assert item.distractors == []


@pytest.mark.asyncio
async def test_solve_operation_variant_with_a_single_root():
    # verify_claim (app.cas.solver) only compares a single value, not a
    # multi-root set (that set-aware comparison is alignment.py's job in
    # Phase 4's grader) - a solve variant with exactly one root is within
    # its real, documented scope.
    variant = _variant_json(
        stem="Solve 2*x - 6 = 0.", operation="solve", expression="2*x - 6 = 0", claimed_answer="x = 3"
    )
    router = _router_with_canned_response(variant)
    item = await generate_llm_variant("algebra.linear", router)
    assert item is not None
    assert "3" in item.correct_answer.value
