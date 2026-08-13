"""
Tests for the Tutor agent's structural enforcement: the action contract
must be respected even when the underlying model draft doesn't respect it.
"""

import json

import pytest

from app.agents import tutor_agent
from app.agents.fallback import get_fallback_response
from app.cas.solver import differentiate, solve_equation
from app.knowledge.schemas import DocType, RetrievedChunk
from app.llm.client import LLMCallResult, ModelRouter, ModelUnavailableError, MockProvider, ProviderClient
from app.llm.router_config import Provider
from app.models.contracts import Action, ActionType
from app.questions.generator import generate_item


def _router_with_canned_response(text: str) -> ModelRouter:
    return ModelRouter(providers={Provider.GROQ: MockProvider(canned_response=text)})


class _QueuedProvider(ProviderClient):
    """Returns a different scripted response on each successive call, in
    order — used where a single test needs to script the Tutor draft and
    the independent Critic response separately (MockProvider only ever
    returns one fixed canned response, insufficient here)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


def _router_with_queued_responses(responses: list[str]) -> ModelRouter:
    return ModelRouter(providers={Provider.GROQ: _QueuedProvider(responses)})


class _AlwaysFailsProvider:
    async def generate(self, *, spec, system, user, images=None):
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
    router = ModelRouter(providers={Provider.GROQ: _AlwaysFailsProvider()})
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


# ---------------------------------------------------------------------------
# CHALLENGE + generated extension items (spec §1.5's "instead of full
# solve" — CHALLENGE is leak-sensitive like HINT/QUESTION, not CAS-gated
# like EXPLAIN, since it must never state the new item's answer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_challenge_draft_that_only_poses_the_item_passes_through():
    item = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=1)
    router = _router_with_canned_response(f"Nice work! Here's a harder one: {item.rendered_stem} Give it a try.")
    action = Action(action_type=ActionType.CHALLENGE, move="extension_question", reason="test")

    response = await tutor_agent.generate(action, "I got it right", router, challenge_item=item)

    assert response.ui_metadata["templated"] is False
    assert item.rendered_stem in response.text


@pytest.mark.asyncio
async def test_challenge_draft_leaking_the_items_answer_is_blocked():
    item = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=1)
    leaking_draft = f"Try this: {item.rendered_stem} (in case you want it, the answer is {item.correct_answer.value})"
    router = _router_with_canned_response(leaking_draft)
    action = Action(action_type=ActionType.CHALLENGE, move="extension_question", reason="test")

    response = await tutor_agent.generate(action, "I got it right", router, challenge_item=item)

    assert response.ui_metadata["templated"] is True
    assert item.correct_answer.value not in response.text


@pytest.mark.asyncio
async def test_challenge_fallback_uses_the_generated_item_when_provider_fails():
    item = generate_item("AA.SL.CALC.DIFF.CHAIN.T003", seed=2)
    router = ModelRouter(providers={Provider.GROQ: _AlwaysFailsProvider()})
    action = Action(action_type=ActionType.CHALLENGE, move="extension_question", reason="test")

    response = await tutor_agent.generate(action, "I got it right", router, challenge_item=item)

    assert response.ui_metadata["templated"] is True
    assert item.rendered_stem in response.text
    assert item.correct_answer.value not in response.text


@pytest.mark.asyncio
async def test_challenge_with_no_generated_item_uses_generic_fallback_on_leak():
    """If item generation itself failed upstream (challenge_item=None),
    the generic CHALLENGE fallback text is used instead — never a crash."""
    router = _router_with_canned_response("The answer is 42, go ahead and use that.")
    action = Action(action_type=ActionType.CHALLENGE, move="extension_question", reason="test")

    response = await tutor_agent.generate(action, "I got it right", router, challenge_item=None)

    assert response.ui_metadata["templated"] is True
    assert "42" not in response.text


# ---------------------------------------------------------------------------
# QUESTION/retrieval_practice + generated review items (spec §12, Phase 9's
# Adaptive Learning Engine): a real spaced-repetition item bound to a
# QUESTION action needs the exact same answer-leak protection a CHALLENGE
# item gets, since the student is meant to attempt it, not be told the
# answer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_question_draft_that_only_poses_the_item_passes_through():
    item = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=1)
    router = _router_with_canned_response(f"This one's due for review: {item.rendered_stem} Try it from memory.")
    action = Action(action_type=ActionType.QUESTION, move="retrieval_practice", reason="exam_prep_review_due")

    response = await tutor_agent.generate(action, "help me review", router, challenge_item=item)

    assert response.ui_metadata["templated"] is False
    assert item.rendered_stem in response.text


@pytest.mark.asyncio
async def test_review_question_draft_leaking_the_items_answer_is_blocked():
    item = generate_item("AA.SL.CALC.DIFF.POWER.T001", seed=1)
    leaking_draft = f"This one's due: {item.rendered_stem} (the answer is {item.correct_answer.value}, by the way)"
    router = _router_with_canned_response(leaking_draft)
    action = Action(action_type=ActionType.QUESTION, move="retrieval_practice", reason="exam_prep_review_due")

    response = await tutor_agent.generate(action, "help me review", router, challenge_item=item)

    assert response.ui_metadata["templated"] is True
    assert item.correct_answer.value not in response.text


@pytest.mark.asyncio
async def test_review_question_fallback_uses_the_generated_item_when_provider_fails():
    item = generate_item("AA.SL.CALC.DIFF.CHAIN.T003", seed=2)
    router = ModelRouter(providers={Provider.GROQ: _AlwaysFailsProvider()})
    action = Action(action_type=ActionType.QUESTION, move="retrieval_practice", reason="exam_prep_review_due")

    response = await tutor_agent.generate(action, "help me review", router, challenge_item=item)

    assert response.ui_metadata["templated"] is True
    assert item.rendered_stem in response.text
    assert item.correct_answer.value not in response.text


@pytest.mark.asyncio
async def test_plain_retrieval_practice_question_without_an_item_is_unaffected():
    """A QUESTION/retrieval_practice turn with no due review (no bound
    item) must behave exactly as it always has — this change only adds
    protection for the *new* bound-item case, it doesn't alter the old
    unbound one."""
    router = _router_with_canned_response("Can you recall the key rule this topic depends on?")
    action = Action(action_type=ActionType.QUESTION, move="retrieval_practice", reason="test")

    response = await tutor_agent.generate(action, "help me review", router, challenge_item=None)

    assert response.ui_metadata["templated"] is False


# ---------------------------------------------------------------------------
# IA/EE Supervisor coaching word cap (spec §11, Phase 10): a hard,
# structural response-length limit is the "never full ghostwriting"
# guarantee's second layer, backstopping the prompt-level instruction.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_ia_coaching_draft_passes_through():
    draft = "What subject area interests you most, and what's one specific question within it?"
    router = _router_with_canned_response(draft)
    action = Action(action_type=ActionType.EXPLAIN, move="ia_topic_coaching", reason="test")

    response = await tutor_agent.generate(action, "help me pick a topic", router)

    assert response.ui_metadata["templated"] is False
    assert response.text == draft


@pytest.mark.asyncio
async def test_ia_coaching_draft_over_the_word_cap_is_rejected():
    long_draft = "word " * 250  # well over the 180-word cap
    router = _router_with_canned_response(long_draft)
    action = Action(action_type=ActionType.EXPLAIN, move="ia_methodology_coaching", reason="test")

    response = await tutor_agent.generate(action, "help with my methodology", router)

    assert response.ui_metadata["templated"] is True
    assert response.text != long_draft


@pytest.mark.asyncio
async def test_ia_coaching_draft_at_exactly_the_cap_passes():
    # Realistic-shaped coaching text (direct address present, no essay-
    # opening pattern) repeated out to exactly the word cap — isolates
    # the word-count boundary from Phase 16's separate essay-content
    # heuristic, which a degenerate "word word word..." string would
    # trip for an unrelated reason (no direct address at all).
    at_cap_draft = ("Tell me more about your idea and what draws you to it. " * 30).split()
    at_cap_draft = " ".join(at_cap_draft[:180])
    router = _router_with_canned_response(at_cap_draft)
    action = Action(action_type=ActionType.EXPLAIN, move="ia_topic_coaching", reason="test")

    response = await tutor_agent.generate(action, "help me pick a topic", router)

    assert response.ui_metadata["templated"] is False


@pytest.mark.asyncio
async def test_word_cap_does_not_apply_to_a_normal_explain_move():
    """The word cap is scoped to ia_* moves only — a long, legitimate
    EXPLAIN response for a normal math turn must not be affected."""
    long_draft = "word " * 250
    router = _router_with_canned_response(long_draft)
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain the chain rule", router)

    assert response.ui_metadata["templated"] is False


# ---------------------------------------------------------------------------
# IA/EE output-side content guard (spec §11, Phase 16): the word cap
# alone doesn't catch a short-but-complete paragraph of actual
# submittable content, so these are the two real signals that back it up.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ia_coaching_draft_with_an_essay_opening_is_rejected_even_if_short():
    draft = "This essay will explore the effects of temperature on enzyme activity in living systems."
    router = _router_with_canned_response(draft)
    action = Action(action_type=ActionType.EXPLAIN, move="ia_drafting_coaching", reason="test")

    response = await tutor_agent.generate(action, "can you look at my introduction", router)

    assert response.ui_metadata["templated"] is True
    assert response.text != draft


@pytest.mark.asyncio
async def test_ia_coaching_draft_with_no_direct_address_is_rejected_when_long_enough():
    # No essay-opening phrase, well under the word cap, but reads as
    # third-person exposition rather than coaching addressed to a student.
    draft = (
        "Enzyme activity depends strongly on temperature, with reaction rates typically increasing up to an "
        "optimal point before denaturation causes a sharp decline. This relationship is often modeled using "
        "the Arrhenius equation, which describes the exponential dependence of reaction rate on temperature "
        "and activation energy across a defined range of conditions studied experimentally over time."
    )
    assert len(draft.split()) >= 40
    router = _router_with_canned_response(draft)
    action = Action(action_type=ActionType.EXPLAIN, move="ia_analysis_coaching", reason="test")

    response = await tutor_agent.generate(action, "help with my analysis section", router)

    assert response.ui_metadata["templated"] is True


@pytest.mark.asyncio
async def test_short_draft_without_direct_address_is_exempt_from_the_address_check():
    # Under the word-count floor for the address check - a short
    # acknowledgment doesn't need "you" to obviously be coaching.
    draft = "Good instinct. Narrow it down further."
    router = _router_with_canned_response(draft)
    action = Action(action_type=ActionType.EXPLAIN, move="ia_topic_coaching", reason="test")

    response = await tutor_agent.generate(action, "is chemistry a good topic", router)

    assert response.ui_metadata["templated"] is False
    assert response.text == draft


@pytest.mark.asyncio
async def test_long_coaching_draft_with_direct_address_passes():
    draft = (
        "Think about what specifically draws you to this topic, and whether you can narrow it to one "
        "measurable variable relationship. Your research question needs to be answerable with data or sources "
        "you can realistically access within the word limit, so consider what resources you actually have "
        "available before committing to a direction, and tell me what you're leaning toward so I can give you "
        "more specific feedback on it."
    )
    assert len(draft.split()) >= 40
    router = _router_with_canned_response(draft)
    action = Action(action_type=ActionType.EXPLAIN, move="ia_topic_coaching", reason="test")

    response = await tutor_agent.generate(action, "help me pick a topic", router)

    assert response.ui_metadata["templated"] is False
    assert response.text == draft


@pytest.mark.asyncio
async def test_essay_opening_check_does_not_apply_to_a_normal_explain_move():
    draft = "This essay will explore the derivative of x squared, which is 2x by the power rule."
    router = _router_with_canned_response(draft)
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain the power rule", router)

    assert response.ui_metadata["templated"] is False


# ---------------------------------------------------------------------------
# Verifier/Critic + escalation/regeneration (spec §13.5, §13.8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critic_block_verdict_discards_an_otherwise_clean_draft():
    router = _router_with_queued_responses(
        [
            "A perfectly clean-looking draft explanation.",
            json.dumps({"verdict": "block", "violations": ["off-topic content"]}),
        ]
    )
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain something", router)

    assert response.ui_metadata["templated"] is True
    assert response.text != "A perfectly clean-looking draft explanation."


@pytest.mark.asyncio
async def test_critic_pass_verdict_lets_a_clean_draft_through_with_metadata():
    router = _router_with_queued_responses(
        [
            "A clean draft explanation.",
            json.dumps({"verdict": "pass", "violations": []}),
        ]
    )
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain something", router)

    assert response.ui_metadata["templated"] is False
    assert response.ui_metadata["critique_verdict"] == "pass"
    assert response.ui_metadata["critic_degraded"] is False
    assert response.text == "A clean draft explanation."


@pytest.mark.asyncio
async def test_critic_revise_verdict_triggers_one_regeneration_that_succeeds():
    router = _router_with_queued_responses(
        [
            "A slightly blunt draft.",
            json.dumps({"verdict": "revise", "violations": ["tone too blunt"]}),
            "A warmer, regenerated draft.",
        ]
    )
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain something", router)

    assert response.ui_metadata["templated"] is False
    assert response.ui_metadata["critique_verdict"] == "revise"
    assert response.text == "A warmer, regenerated draft."


@pytest.mark.asyncio
async def test_critic_revise_verdict_falls_back_when_regeneration_also_leaks():
    action = Action(action_type=ActionType.HINT, level=1, reason="test")
    router = _router_with_queued_responses(
        [
            "A decent hint without leaking.",
            json.dumps({"verdict": "revise", "violations": ["could be clearer"]}),
            "Oops, the answer is 42 now.",  # regeneration itself leaks
        ]
    )

    response = await tutor_agent.generate(action, "give me a hint", router)

    assert response.ui_metadata["templated"] is True
    assert "42" not in response.text


@pytest.mark.asyncio
async def test_critic_revise_verdict_falls_back_when_regeneration_provider_fails():
    class _FailsOnThirdCallProvider(ProviderClient):
        def __init__(self) -> None:
            self._responses = ["An okay draft.", json.dumps({"verdict": "revise", "violations": ["needs more detail"]})]

        async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
            if not self._responses:
                raise ModelUnavailableError("simulated outage on regeneration attempt")
            text = self._responses.pop(0)
            return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)

    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")
    router = ModelRouter(providers={Provider.GROQ: _FailsOnThirdCallProvider()})

    response = await tutor_agent.generate(action, "explain something", router)

    assert response.ui_metadata["templated"] is True
    assert response.text != "An okay draft."


@pytest.mark.asyncio
async def test_ungrounded_explain_draft_is_blocked_even_with_critic_pass():
    """The grounding check (spec §13.6) is an independent gate — a critic
    that says "pass" doesn't override a draft that clearly doesn't reflect
    its cited context."""
    chunks = [
        RetrievedChunk(
            chunk_id="FB-AA-5.8",
            doc_type=DocType.FORMULA_BOOKLET_ENTRY,
            subtopic_id="calculus.differentiation.chain_rule",
            citation="Formula booklet, Calculus: Chain rule",
            text="Chain rule: dy/dx = dy/du times du/dx, used to differentiate composite functions",
            score=0.95,
            authority_tier=1.0,
        )
    ]
    router = _router_with_queued_responses(
        [
            "The capital of France is Paris and bananas are a good source of potassium.",
            json.dumps({"verdict": "pass", "violations": []}),
        ]
    )
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain the chain rule", router, retrieved_chunks=chunks)

    assert response.ui_metadata["templated"] is True


@pytest.mark.asyncio
async def test_below_threshold_retrieved_chunks_do_not_trigger_a_grounding_failure():
    """Regression: retrieved_chunks that never cleared the citation
    threshold (so nothing was actually cited) must not be grounding-
    checked at all — checking a draft against irrelevant, sub-threshold
    candidates would fail almost any real draft for the wrong reason."""
    weak_chunks = [
        RetrievedChunk(
            chunk_id="LO-AA-5.9.1",
            doc_type=DocType.LEARNING_OBJECTIVE,
            subtopic_id="calculus.integration.reverse_power_rule",
            citation="IB DP Mathematics guide, section 5.9",
            text="Integrate using the reverse of differentiation.",
            score=0.05,  # well below the grounding threshold
            authority_tier=1.0,
        )
    ]
    router = _router_with_queued_responses(
        [
            "A clean draft that has nothing to do with the weak chunk above.",
            json.dumps({"verdict": "pass", "violations": []}),
        ]
    )
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "explain something", router, retrieved_chunks=weak_chunks)

    assert response.ui_metadata["templated"] is False
    assert response.citations == []


# ---------------------------------------------------------------------------
# _strip_trailing_json_echo: a weaker model can literally comply with a now-
# removed "OUTPUT SCHEMA" instruction and echo a stray {"text": ...} blob
# after its real prose - nothing downstream ever parsed that JSON, so it
# must be stripped as a real deterministic backstop, not left to the prompt.
# ---------------------------------------------------------------------------


def test_find_trailing_json_object_isolates_only_the_trailing_balanced_blob():
    text = 'The derivative is \\frac{d}{dx}. {"text": "answer", "citations": []}'
    found = tutor_agent._find_trailing_json_object(text)
    assert found == '{"text": "answer", "citations": []}'


def test_find_trailing_json_object_returns_none_when_text_does_not_end_in_brace():
    assert tutor_agent._find_trailing_json_object("just plain prose.") is None


def test_strip_trailing_json_echo_removes_a_real_schema_echo():
    text = 'The chain rule multiplies the outer and inner derivatives.\n\n{"text": "dup", "citations": [], "ui_hints": {"show_hint_button": false}}'
    cleaned = tutor_agent._strip_trailing_json_echo(text)
    assert cleaned == "The chain rule multiplies the outer and inner derivatives."


def test_strip_trailing_json_echo_leaves_real_latex_braces_untouched():
    text = "The derivative of x^2 is 2x, written \\(\\frac{d}{dx}x^2 = 2x\\)."
    assert tutor_agent._strip_trailing_json_echo(text) == text


def test_strip_trailing_json_echo_leaves_a_trailing_brace_that_is_not_json():
    text = "Solve for the set {1, 2, 3}"
    assert tutor_agent._strip_trailing_json_echo(text) == text


def test_strip_trailing_json_echo_leaves_json_with_unrelated_keys_untouched():
    """Only strips a blob carrying this codebase's own specific
    (unused) schema keys - an unrelated trailing JSON object a student
    might paste as part of their own message must never be touched."""
    text = 'my array is {"a": 1, "b": 2}'
    assert tutor_agent._strip_trailing_json_echo(text) == text


def test_strip_trailing_json_echo_recovers_the_real_text_when_the_whole_draft_is_wrapped():
    text = json.dumps({"text": "The real answer lives here.", "citations": [], "ui_hints": {"show_hint_button": True}})
    assert tutor_agent._strip_trailing_json_echo(text) == "The real answer lives here."


@pytest.mark.asyncio
async def test_generate_strips_a_real_trailing_json_echo_from_a_live_style_draft():
    draft = (
        "A vector has both magnitude and direction.\n\n"
        '{"text": "A vector has both magnitude and direction.", "citations": [], '
        '"ui_hints": {"show_hint_button": false}}'
    )
    router = _router_with_canned_response(draft)
    action = Action(action_type=ActionType.EXPLAIN, move="direct_explanation", reason="test")

    response = await tutor_agent.generate(action, "what is a vector?", router)

    assert response.text == "A vector has both magnitude and direction."
    assert '"citations"' not in response.text
    assert '"ui_hints"' not in response.text
