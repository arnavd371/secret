"""
Tests for the Verifier/Critic Agent: real LLM-response parsing, the
static-fallback degradation path, and the checklist logic itself.
"""

import json

import pytest

from app.llm.client import ModelRouter, ModelUnavailableError, MockProvider
from app.llm.router_config import Provider
from app.models.contracts import Action, ActionType
from app.verifier.critic import _latex_well_formed, _static_fallback_critique, critique_draft
from app.verifier.models import CritiqueVerdict


def _router_with_canned_response(text: str) -> ModelRouter:
    return ModelRouter(providers={Provider.ANTHROPIC: MockProvider(canned_response=text)})


class _AlwaysFailsProvider:
    async def generate(self, *, spec, system, user):
        raise ModelUnavailableError("simulated critic outage")


@pytest.mark.asyncio
async def test_critique_draft_parses_pass_verdict():
    router = _router_with_canned_response(json.dumps({"verdict": "pass", "violations": []}))
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    result = await critique_draft("some draft", action, router)
    assert result.verdict == CritiqueVerdict.PASS
    assert result.critic_degraded is False


@pytest.mark.asyncio
async def test_critique_draft_parses_block_verdict_with_violations():
    router = _router_with_canned_response(json.dumps({"verdict": "block", "violations": ["leaked answer"]}))
    action = Action(action_type=ActionType.HINT, level=1, reason="test")
    result = await critique_draft("the answer is 42", action, router)
    assert result.verdict == CritiqueVerdict.BLOCK
    assert result.violations == ["leaked answer"]


@pytest.mark.asyncio
async def test_critique_draft_parses_revise_verdict():
    router = _router_with_canned_response(json.dumps({"verdict": "revise", "violations": ["tone is blunt"]}))
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    result = await critique_draft("draft text", action, router)
    assert result.verdict == CritiqueVerdict.REVISE


@pytest.mark.asyncio
async def test_critique_draft_degrades_to_static_check_on_unparseable_response():
    router = _router_with_canned_response("not json at all")
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    result = await critique_draft("a clean draft", action, router)
    assert result.critic_degraded is True
    assert result.verdict == CritiqueVerdict.PASS  # no leak, no malformed latex


@pytest.mark.asyncio
async def test_critique_draft_degrades_to_static_check_on_provider_outage():
    router = ModelRouter(providers={Provider.ANTHROPIC: _AlwaysFailsProvider()})
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    result = await critique_draft("a clean draft", action, router)
    assert result.critic_degraded is True


@pytest.mark.asyncio
async def test_critique_draft_degrades_gracefully_on_malformed_json_shape():
    """A response that parses as JSON but doesn't have the expected
    fields must still degrade safely, not raise."""
    router = _router_with_canned_response(json.dumps({"unexpected": "shape"}))
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    result = await critique_draft("a clean draft", action, router)
    assert result.critic_degraded is True


# ---------------------------------------------------------------------------
# Static fallback check itself
# ---------------------------------------------------------------------------


def test_static_fallback_blocks_leak_on_non_explain_action():
    action = Action(action_type=ActionType.HINT, level=1, reason="test")
    result = _static_fallback_critique("The answer is 42.", action)
    assert result.verdict == CritiqueVerdict.BLOCK
    assert result.critic_degraded is True


def test_static_fallback_allows_answer_on_explain_action():
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    result = _static_fallback_critique("The answer is 42.", action)
    assert result.verdict == CritiqueVerdict.PASS


def test_static_fallback_blocks_unbalanced_latex():
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    result = _static_fallback_critique(r"Here is \(x^2 unbalanced", action)
    assert result.verdict == CritiqueVerdict.BLOCK


def test_latex_well_formed_checks_balance():
    assert _latex_well_formed(r"\(x^2\) and \[y = 2\]") is True
    assert _latex_well_formed(r"\(x^2 missing close") is False
