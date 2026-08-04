"""
Tests for the Tutor agent's structural enforcement: the action contract
must be respected even when the underlying model draft doesn't respect it.
"""

import pytest

from app.agents import tutor_agent
from app.agents.fallback import get_fallback_response
from app.llm.client import ModelRouter, ModelUnavailableError, MockProvider
from app.llm.router_config import Provider
from app.models.contracts import Action, ActionType


def _router_with_canned_response(text: str) -> ModelRouter:
    return ModelRouter(providers={Provider.ANTHROPIC: MockProvider(canned_response=text)})


class _AlwaysFailsProvider:
    async def generate(self, *, spec, system, user):
        raise ModelUnavailableError("simulated provider outage")

    async def stream(self, *, spec, system, user):
        raise ModelUnavailableError("simulated provider outage")
        yield ""  # pragma: no cover - unreachable, keeps this an async generator


@pytest.mark.asyncio
async def test_generate_returns_clean_draft_untouched():
    router = _router_with_canned_response("Think about which trig identity relates sin and cos here.")
    action = Action(action_type=ActionType.HINT, level=1, reason="test")

    response = await tutor_agent.generate(action, "how do I start this?", router)

    assert "trig identity" in response.text
    assert response.ui_metadata["templated"] is False


@pytest.mark.asyncio
async def test_generate_discards_leaked_answer_on_hint_action():
    router = _router_with_canned_response("Just plug in the values. The answer is 42.")
    action = Action(action_type=ActionType.HINT, level=2, reason="test")

    response = await tutor_agent.generate(action, "give me a hint", router)

    expected_fallback = get_fallback_response(action)
    assert response.text == expected_fallback.text
    assert response.ui_metadata["templated"] is True
    assert "42" not in response.text


@pytest.mark.asyncio
async def test_generate_discards_leaked_answer_on_question_action():
    router = _router_with_canned_response("Therefore, x = 7. What do you think comes next?")
    action = Action(action_type=ActionType.QUESTION, move="socratic_prompt", reason="test")

    response = await tutor_agent.generate(action, "what's next?", router)

    assert response.ui_metadata["templated"] is True
    assert "x = 7" not in response.text


@pytest.mark.asyncio
async def test_generate_allows_final_answer_on_explain_action():
    """EXPLAIN is the one action type allowed to state the answer, so the
    leak-check must not fire for it."""
    router = _router_with_canned_response("The answer is 42, because substituting gives 6*7.")
    action = Action(action_type=ActionType.EXPLAIN, move="concept_explanation", reason="test")

    response = await tutor_agent.generate(action, "can you check my work?", router)

    assert response.ui_metadata["templated"] is False
    assert "42" in response.text


@pytest.mark.asyncio
async def test_generate_falls_back_when_provider_unavailable():
    router = ModelRouter(providers={Provider.ANTHROPIC: _AlwaysFailsProvider()})
    action = Action(action_type=ActionType.HINT, level=3, reason="test")

    response = await tutor_agent.generate(action, "help", router)

    expected_fallback = get_fallback_response(action)
    assert response.text == expected_fallback.text
    assert response.ui_metadata["templated"] is True


@pytest.mark.asyncio
async def test_generate_rejects_refuse_action():
    router = _router_with_canned_response("shouldn't be called")
    action = Action(action_type=ActionType.REFUSE, reason="test")

    with pytest.raises(ValueError):
        await tutor_agent.generate(action, "x", router)


@pytest.mark.asyncio
async def test_stream_response_reassembles_to_the_same_approved_text():
    router = _router_with_canned_response("A clean, non-leaking hint about the next step.")
    action = Action(action_type=ActionType.HINT, level=1, reason="test")

    expected = await tutor_agent.generate(action, "hint please", router)

    chunks = [chunk async for chunk in tutor_agent.stream_response(action, "hint please", router)]
    assert "".join(chunks) == expected.text
