"""
Integration coverage proving the Verifier/Critic (Phase 6) is actually
wired into the real orchestrator path, not just unit-tested in isolation.
"""

from __future__ import annotations

import json

import pytest

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore


class QueuedProvider(ProviderClient):
    """Every call, including the critic's, is explicitly scripted in
    order — this file is specifically testing critic behavior, so it does
    not use the other integration tests' auto-pass-the-critic shortcut."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user) -> LLMCallResult:
        self.calls.append({"model": spec.model, "system": system})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


def _intent_json(intent: str = "concept_explain") -> str:
    return json.dumps(
        {
            "intent": intent,
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.mark.asyncio
async def test_critic_block_reaches_student_as_templated_fallback():
    provider = QueuedProvider(
        [
            _intent_json(),
            "A draft that a critic will reject.",
            json.dumps({"verdict": "block", "violations": ["off-topic"]}),
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "explain something to me",
        session_id="sess-critic-1",
        student_id="student-1",
        router=router,
        session_store=store,
    )

    assert result.final_response.ui_metadata["templated"] is True
    assert result.final_response.text != "A draft that a critic will reject."
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_critic_revise_regenerates_and_serves_the_new_draft():
    provider = QueuedProvider(
        [
            _intent_json(),
            "A slightly rough draft.",
            json.dumps({"verdict": "revise", "violations": ["could be warmer"]}),
            "A warmer, regenerated draft.",
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "explain something to me",
        session_id="sess-critic-2",
        student_id="student-2",
        router=router,
        session_store=store,
    )

    assert result.final_response.ui_metadata["templated"] is False
    assert result.final_response.text == "A warmer, regenerated draft."
    assert len(provider.calls) == 4


@pytest.mark.asyncio
async def test_critic_pass_serves_the_original_draft_with_metadata():
    provider = QueuedProvider(
        [
            _intent_json(),
            "A clean, well-grounded draft.",
            json.dumps({"verdict": "pass", "violations": []}),
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "explain something to me",
        session_id="sess-critic-3",
        student_id="student-3",
        router=router,
        session_store=store,
    )

    assert result.final_response.text == "A clean, well-grounded draft."
    assert result.final_response.ui_metadata["critique_verdict"] == "pass"
    assert result.final_response.ui_metadata["critic_degraded"] is False
