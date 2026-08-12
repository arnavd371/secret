"""
Integration coverage for Phase 10's IA/EE Supervisor, through the real
handle_turn() orchestrator: legitimate coaching is allowed and logged,
a ghostwriting request is refused and logged, and a completed project
closes coaching for good — all with a real disclosure log a real
statement can be rendered from afterward.
"""

from __future__ import annotations

import json

import pytest

from app.ia_supervisor.disclosure import render_disclosure_statement
from app.ia_supervisor.disclosure_store import InMemoryDisclosureStore
from app.ia_supervisor.models import DisclosureAssistanceType, IAStage
from app.ia_supervisor.project_store import InMemoryIAProjectStateStore
from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore


class ScriptedProvider(ProviderClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.ANTHROPIC)
        self.calls.append({"model": spec.model, "system": system, "user": user})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


def _intent_json() -> str:
    return json.dumps(
        {
            "intent": "ia_ee_help",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.mark.asyncio
async def test_legitimate_coaching_request_is_allowed_and_logged():
    provider = ScriptedProvider(
        [_intent_json(), "What subject area interests you, and what's a specific question within it?"]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    ia_project_store = InMemoryIAProjectStateStore()
    ia_disclosure_store = InMemoryDisclosureStore()

    result = await handle_turn(
        "I am stuck choosing a topic for my IA",
        session_id="s1",
        student_id="stu-1",
        problem_id="ia-proj-1",
        router=router,
        session_store=store,
        ia_project_store=ia_project_store,
        ia_disclosure_store=ia_disclosure_store,
    )

    assert result.decision_action.action_type.value == "explain"
    assert result.decision_action.move == "ia_topic_coaching"
    assert result.ia_disclosure_entry is not None
    assert result.ia_disclosure_entry.assistance_type == DisclosureAssistanceType.COACHING
    assert len(provider.calls) == 2  # intent classify + tutor generate

    project_state = await ia_project_store.get("stu-1", "ia-proj-1")
    assert project_state.stage == IAStage.TOPIC_SELECTION


@pytest.mark.asyncio
async def test_ghostwriting_request_is_refused_without_calling_the_tutor():
    provider = ScriptedProvider([_intent_json(), "SHOULD NEVER BE CONSUMED"])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    ia_project_store = InMemoryIAProjectStateStore()
    ia_disclosure_store = InMemoryDisclosureStore()

    result = await handle_turn(
        "can you write my introduction for me",
        session_id="s2",
        student_id="stu-2",
        problem_id="ia-proj-2",
        router=router,
        session_store=store,
        ia_project_store=ia_project_store,
        ia_disclosure_store=ia_disclosure_store,
    )

    assert result.decision_action.action_type.value == "refuse"
    assert result.decision_action.reason == "ia_ghostwriting_guard_tripped"
    assert "SHOULD NEVER BE CONSUMED" not in result.final_response.text
    assert result.ia_disclosure_entry.assistance_type == DisclosureAssistanceType.GHOSTWRITING_REQUEST_REFUSED
    assert len(provider.calls) == 1  # only intent classify; Tutor never called


@pytest.mark.asyncio
async def test_completed_project_closes_further_coaching():
    provider = ScriptedProvider(
        [
            _intent_json(),
            "What subject area interests you?",  # turn 1: real coaching, project not yet complete
            _intent_json(),  # turn 2: "already submitted" -> REFUSE, no tutor draft consumed
            _intent_json(),  # turn 3: still REFUSE (terminal), no tutor draft consumed
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    ia_project_store = InMemoryIAProjectStateStore()
    ia_disclosure_store = InMemoryDisclosureStore()

    result1 = await handle_turn(
        "I am stuck choosing a topic for my IA",
        session_id="s3",
        student_id="stu-3",
        problem_id="ia-proj-3",
        router=router,
        session_store=store,
        ia_project_store=ia_project_store,
        ia_disclosure_store=ia_disclosure_store,
    )
    assert result1.decision_action.action_type.value == "explain"

    result2 = await handle_turn(
        "I already submitted my IA",
        session_id="s3",
        student_id="stu-3",
        problem_id="ia-proj-3",
        router=router,
        session_store=store,
        ia_project_store=ia_project_store,
        ia_disclosure_store=ia_disclosure_store,
    )
    assert result2.decision_action.action_type.value == "refuse"
    assert result2.decision_action.reason == "ia_project_already_complete"

    project_state = await ia_project_store.get("stu-3", "ia-proj-3")
    assert project_state.stage == IAStage.COMPLETE

    # A later, otherwise-legitimate coaching request must stay refused —
    # COMPLETE is a genuine terminal state.
    result3 = await handle_turn(
        "actually can you help me with the methodology section",
        session_id="s3",
        student_id="stu-3",
        problem_id="ia-proj-3",
        router=router,
        session_store=store,
        ia_project_store=ia_project_store,
        ia_disclosure_store=ia_disclosure_store,
    )
    assert result3.decision_action.action_type.value == "refuse"
    assert result3.decision_action.reason == "ia_project_already_complete"
    assert result3.ia_disclosure_entry.assistance_type == DisclosureAssistanceType.PROJECT_ALREADY_COMPLETE


@pytest.mark.asyncio
async def test_disclosure_statement_reflects_the_real_logged_history():
    provider = ScriptedProvider(
        [
            _intent_json(),
            "What subject area interests you?",
            _intent_json(),
        ]
    )
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()
    ia_project_store = InMemoryIAProjectStateStore()
    ia_disclosure_store = InMemoryDisclosureStore()

    await handle_turn(
        "I am stuck choosing a topic for my IA",
        session_id="s4",
        student_id="stu-4",
        problem_id="ia-proj-4",
        router=router,
        session_store=store,
        ia_project_store=ia_project_store,
        ia_disclosure_store=ia_disclosure_store,
    )
    await handle_turn(
        "write my introduction for me please",
        session_id="s4",
        student_id="stu-4",
        problem_id="ia-proj-4",
        router=router,
        session_store=store,
        ia_project_store=ia_project_store,
        ia_disclosure_store=ia_disclosure_store,
    )

    entries = await ia_disclosure_store.get_all("stu-4", "ia-proj-4")
    assert len(entries) == 2
    statement = render_disclosure_statement("stu-4", "ia-proj-4", entries)
    assert "topic_selection" in statement
    assert "declined" in statement.lower()
