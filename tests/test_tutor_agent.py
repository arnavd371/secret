"""
Tests for the Tutor agent's structural enforcement: the action contract
must be respected even when the underlying model draft doesn't respect it.
"""

import pytest

from app.agents import tutor_agent
from app.agents.fallback import get_fallback_response
from app.cas.solver import differentiate, solve_equation
from app.knowledge.schemas import DocType, RetrievedChunk
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


# ---------------------------------------------------------------------------
# CAS gating (spec §1.4): EXPLAIN/CHALLENGE drafts are checked against a
# ground-truth CASResult when one was computed for the turn.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explain_draft_agreeing_with_cas_passes_through():
    cas_result = differentiate("x**2", "x")  # 2*x
    router = _router_with_canned_response("Applying the power rule, the derivative is 2*x.")
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "what's the derivative of x^2?", router, cas_result=cas_result)

    assert response.ui_metadata["templated"] is False
    assert "2*x" in response.text


@pytest.mark.asyncio
async def test_explain_draft_disagreeing_with_cas_is_replaced_with_grounded_response():
    cas_result = differentiate("x**2", "x")  # 2*x
    router = _router_with_canned_response("Applying the power rule, the answer is 3*x.")  # wrong
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "what's the derivative of x^2?", router, cas_result=cas_result)

    assert response.ui_metadata.get("cas_grounded") is True
    assert response.ui_metadata["templated"] is True
    assert "2*x" in response.text
    assert "3*x" not in response.text or "2*x" in response.text  # the CAS-correct value is what's shown


@pytest.mark.asyncio
async def test_explain_downgrades_to_hint_style_fallback_when_cas_unverifiable():
    cas_result = solve_equation("x - cos(x) = 0", "x")  # no closed form -> unverifiable
    assert cas_result.status.value == "unverifiable"
    router = _router_with_canned_response("The answer is x = 0.739.")  # model guessed anyway
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "solve x - cos(x) = 0", router, cas_result=cas_result)

    assert response.ui_metadata["cas_status"] == "unverifiable"
    assert response.ui_metadata["templated"] is True
    assert "0.739" not in response.text


@pytest.mark.asyncio
async def test_explain_draft_without_a_stated_value_is_not_forced_through_cas_gate():
    """A conceptual explanation that doesn't claim a discrete final value
    has nothing to check against CAS, and should pass through untouched."""
    cas_result = differentiate("x**2", "x")
    router = _router_with_canned_response(
        "The power rule says that to differentiate x raised to a power, you bring the exponent down and reduce it by one."
    )
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "how does the power rule work?", router, cas_result=cas_result)

    assert response.ui_metadata["templated"] is False


@pytest.mark.asyncio
async def test_cas_result_does_not_gate_hint_actions():
    """HINT never states a final answer (the leak-check already forbids
    it), so an incidental cas_result should not trigger CAS gating on a
    HINT action — only EXPLAIN/CHALLENGE are CAS-gated."""
    cas_result = solve_equation("x - cos(x) = 0", "x")  # unverifiable
    router = _router_with_canned_response("Think about which technique applies here.")
    action = Action(action_type=ActionType.HINT, level=1, reason="test")

    response = await tutor_agent.generate(action, "give me a hint", router, cas_result=cas_result)

    assert response.ui_metadata["templated"] is False
    assert "cas_status" not in response.ui_metadata


@pytest.mark.asyncio
async def test_grounded_retrieval_attaches_citations_to_clean_explain_draft():
    chunks = [
        RetrievedChunk(
            chunk_id="FB-AA-5.8",
            doc_type=DocType.FORMULA_BOOKLET_ENTRY,
            subtopic_id="calculus.differentiation.chain_rule",
            citation="Formula booklet, Calculus: Chain rule",
            text="Chain rule: dy/dx = dy/du * du/dx",
            score=0.95,
            authority_tier=1.0,
        )
    ]
    router = _router_with_canned_response("The chain rule lets you differentiate composite functions.")
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(
        action, "explain the chain rule", router, retrieved_chunks=chunks
    )

    assert response.citations == ["Formula booklet, Calculus: Chain rule"]


@pytest.mark.asyncio
async def test_ungrounded_retrieval_attaches_no_citations():
    chunks = [
        RetrievedChunk(
            chunk_id="LO-AA-5.9.1",
            doc_type=DocType.LEARNING_OBJECTIVE,
            subtopic_id="calculus.integration.reverse_power_rule",
            citation="IB DP Mathematics guide, section 5.9",
            text="Integrate using the reverse of differentiation.",
            score=0.1,  # below the grounding threshold
            authority_tier=1.0,
        )
    ]
    router = _router_with_canned_response("Here's a general explanation.")
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain integration", router, retrieved_chunks=chunks)

    assert response.citations == []


@pytest.mark.asyncio
async def test_only_individually_grounded_chunks_are_cited():
    """Regression: `is_grounded` only checks the top-ranked chunk, so a
    weaker chunk further down the same top-k list must not ride along on
    the top chunk's confidence just because *a* chunk cleared the bar."""
    chunks = [
        RetrievedChunk(
            chunk_id="FB-AA-5.8",
            doc_type=DocType.FORMULA_BOOKLET_ENTRY,
            subtopic_id="calculus.differentiation.chain_rule",
            citation="Formula booklet, Calculus: Chain rule",
            text="Chain rule: dy/dx = dy/du * du/dx",
            score=0.95,
            authority_tier=1.0,
        ),
        RetrievedChunk(
            chunk_id="FB-AA-2.5",
            doc_type=DocType.FORMULA_BOOKLET_ENTRY,
            subtopic_id="algebra.quadratics.solving",
            citation="Formula booklet, Algebra: Quadratic formula",
            text="x = (-b +/- sqrt(b^2 - 4ac)) / 2a",
            score=0.2,  # nonzero, but below the grounding threshold
            authority_tier=1.0,
        ),
    ]
    router = _router_with_canned_response("The chain rule lets you differentiate composite functions.")
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain the chain rule", router, retrieved_chunks=chunks)

    assert response.citations == ["Formula booklet, Calculus: Chain rule"]
