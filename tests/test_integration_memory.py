"""
Integration coverage for Phase 5's memory wiring through the real
orchestrator (mocked model responses; real BKT/IRT math, decay, and
grading underneath, no network calls).
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.models.contracts import ActionType
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore, ProblemSessionState
from app.memory.models import SubtopicMastery
from app.memory.store import InMemoryMemoryStore


class ScriptedProvider(ProviderClient):
    """Phase 6's Verifier/Critic makes an additional, independent model
    call per turn on the same shared Provider.ANTHROPIC queue. These
    tests aren't exercising critic behavior, so a critic-shaped system
    prompt is auto-passed without consuming a slot in the scripted queue."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.ANTHROPIC)
        self.calls.append({"model": spec.model, "system": system, "user": user})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


def _intent_json(intent: str, topic_hint: str | None = None) -> str:
    return json.dumps(
        {
            "intent": intent,
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": topic_hint,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


TOPIC = "calculus.differentiation.chain_rule"


@pytest.mark.asyncio
async def test_correct_grading_persists_a_mastery_update():
    provider = ScriptedProvider([_intent_json("check_work", TOPIC)])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    work = "u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2"

    result = await handle_turn(
        "differentiate (2*x+1)**5",
        session_id="sess-mem-1",
        student_id="student-1",
        router=router,
        session_store=store,
        memory_store=memory_store,
        student_work=work,
    )

    assert result.mark_result is not None
    assert result.mark_result.confidence.value in ("high", "medium")

    mastery = await memory_store.get_mastery("student-1", TOPIC)
    assert mastery is not None
    assert mastery.attempts_total == 1
    assert mastery.attempts_correct == 1
    assert mastery.p_mastery_bkt > 0.10  # climbed from the p_init default


@pytest.mark.asyncio
async def test_low_confidence_grading_does_not_write_mastery():
    """A grading with nothing gradeable (no working, no answer) is LOW
    confidence and must not corrupt the persisted mastery model."""
    provider = ScriptedProvider([_intent_json("check_work", TOPIC)])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    result = await handle_turn(
        "differentiate (2*x+1)**5",
        session_id="sess-mem-2",
        student_id="student-2",
        router=router,
        session_store=store,
        memory_store=memory_store,
        student_work="I don't know how to start",
    )

    assert result.mark_result is not None
    assert result.mark_result.confidence.value == "low"
    assert await memory_store.get_mastery("student-2", TOPIC) is None


@pytest.mark.asyncio
async def test_persisted_high_mastery_drives_challenge_without_explicit_override():
    """The real proof: mastery built up in one turn (or pre-seeded, same
    persisted-state mechanism) drives the decision policy on a later turn
    with no caller-supplied mastery_estimate at all."""
    provider = ScriptedProvider(
        [
            _intent_json("solve_request", TOPIC),
            "Great work, that's correct! Here's a tougher one for you.",
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    await memory_store.save_mastery(
        SubtopicMastery(student_id="student-3", subtopic_id=TOPIC, p_mastery_bkt=0.95, attempts_total=10, attempts_correct=9)
    )
    await store.save(ProblemSessionState(session_id="sess-mem-3", problem_id="problem-1", attempt_count=1, hint_ladder_level=0))

    result = await handle_turn(
        "I solved it!",
        session_id="sess-mem-3",
        student_id="student-3",
        problem_id="problem-1",
        router=router,
        session_store=store,
        memory_store=memory_store,
        # deliberately no mastery_estimate override
    )

    assert result.decision_action.action_type == ActionType.CHALLENGE
    assert result.generated_item is not None


@pytest.mark.asyncio
async def test_explicit_mastery_estimate_overrides_persisted_memory():
    """A caller-supplied mastery_estimate must win over whatever is
    persisted, preserving the override behavior tests/simulations rely on."""
    provider = ScriptedProvider(
        [_intent_json("solve_request", TOPIC), "What do you think the first step is?"]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    # Persisted mastery is high, but the explicit override is low.
    await memory_store.save_mastery(
        SubtopicMastery(student_id="student-4", subtopic_id=TOPIC, p_mastery_bkt=0.95, attempts_total=10, attempts_correct=9)
    )
    await store.save(ProblemSessionState(session_id="sess-mem-4", problem_id="problem-1", attempt_count=1, hint_ladder_level=0))

    result = await handle_turn(
        "I solved it!",
        session_id="sess-mem-4",
        student_id="student-4",
        problem_id="problem-1",
        router=router,
        session_store=store,
        memory_store=memory_store,
        mastery_estimate=0.2,
    )

    assert result.decision_action.action_type != ActionType.CHALLENGE


@pytest.mark.asyncio
async def test_memory_context_is_populated_on_blackboard_when_record_exists():
    provider = ScriptedProvider(
        [_intent_json("concept_explain", TOPIC), "The chain rule lets you differentiate composite functions."]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    await memory_store.save_mastery(
        SubtopicMastery(student_id="student-5", subtopic_id=TOPIC, p_mastery_bkt=0.6, attempts_total=4)
    )

    result = await handle_turn(
        "explain the chain rule",
        session_id="sess-mem-5",
        student_id="student-5",
        router=router,
        session_store=store,
        memory_store=memory_store,
    )

    assert result.student_state_snapshot is not None
    assert result.student_state_snapshot.subtopic_id == TOPIC
    assert TOPIC in result.student_state_snapshot.rendered_text


@pytest.mark.asyncio
async def test_no_topic_hint_skips_memory_write_and_read():
    provider = ScriptedProvider([_intent_json("check_work", None)])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    result = await handle_turn(
        "differentiate (2*x+1)**5",
        session_id="sess-mem-6",
        student_id="student-6",
        router=router,
        session_store=store,
        memory_store=memory_store,
        student_work="u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2",
    )

    assert result.mark_result is not None
    assert result.student_state_snapshot.subtopic_id is None
    # nothing to key a mastery write against without a topic
    assert await memory_store.get_misconceptions("student-6") == []
