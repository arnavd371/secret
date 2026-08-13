"""
Tests for the FastAPI gateway (app/main.py): real HTTP requests against
the real app object via httpx's ASGI transport (no network, no running
server process), with the module-level router/stores swapped for test
doubles - the same MockProvider/InMemory* stores every other test in
this codebase already uses, just reached through the HTTP layer instead
of calling handle_turn directly.
"""

from __future__ import annotations

import json

import httpx
import pytest

import app.main as main_module
from app.adaptive.store import InMemoryReviewStateStore
from app.guardrail_metrics.store import InMemoryGuardrailMetricsStore
from app.ia_supervisor.disclosure_store import InMemoryDisclosureStore
from app.ia_supervisor.project_store import InMemoryIAProjectStateStore
from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.memory.store import InMemoryMemoryStore
from app.questions.response_log import InMemoryResponseLogStore
from app.review_queue.store import InMemoryReviewQueueStore
from app.session.state import InMemorySessionStateStore


class ScriptedProvider(ProviderClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.GROQ)
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)


def _concept_explain_intent_json() -> str:
    return json.dumps(
        {
            "intent": "concept_explain",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


@pytest.fixture(autouse=True)
def _swap_in_memory_stores(monkeypatch):
    """Every test gets fresh in-memory stores instead of the real
    module-level SQLite ones - fast, isolated, no shared db file
    between tests, same isolation every other test file in this repo
    already gets."""
    monkeypatch.setattr(main_module, "_session_store", InMemorySessionStateStore())
    monkeypatch.setattr(main_module, "_memory_store", InMemoryMemoryStore())
    monkeypatch.setattr(main_module, "_review_store", InMemoryReviewStateStore())
    monkeypatch.setattr(main_module, "_ia_project_store", InMemoryIAProjectStateStore())
    monkeypatch.setattr(main_module, "_ia_disclosure_store", InMemoryDisclosureStore())
    monkeypatch.setattr(main_module, "_response_log_store", InMemoryResponseLogStore())
    monkeypatch.setattr(main_module, "_review_queue_store", InMemoryReviewQueueStore())
    monkeypatch.setattr(main_module, "_guardrail_metrics_store", InMemoryGuardrailMetricsStore())


async def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=main_module.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_endpoint():
    async with await _client() as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_index_serves_the_real_frontend_html():
    async with await _client() as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Tutor" in response.text


@pytest.mark.asyncio
async def test_turn_endpoint_returns_the_real_structured_response(monkeypatch):
    provider = ScriptedProvider([_concept_explain_intent_json(), "The chain rule handles composite functions."])
    monkeypatch.setattr(
        main_module, "_router", ModelRouter(providers={Provider.GROQ: provider, Provider.ANTHROPIC: provider})
    )

    async with await _client() as client:
        response = await client.post(
            "/turn",
            json={
                "raw_input": "can you remind me how the chain rule works?",
                "session_id": "s1",
                "student_id": "stu-1",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "The chain rule handles composite functions."
    assert body["intent"] == "concept_explain"
    assert body["action_type"] == "explain"
    assert body["ui_metadata"]["templated"] is False
    assert body["mark_result"] is None
    assert body["misconception"] is None


@pytest.mark.asyncio
async def test_turn_endpoint_grades_real_check_work_submissions(monkeypatch):
    intent_json = json.dumps(
        {
            "intent": "check_work",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )
    provider = ScriptedProvider([intent_json])
    monkeypatch.setattr(
        main_module, "_router", ModelRouter(providers={Provider.GROQ: provider, Provider.ANTHROPIC: provider})
    )

    async with await _client() as client:
        response = await client.post(
            "/turn",
            json={
                "raw_input": "can you check my work? differentiate x**2 * sin(x)",
                "session_id": "s2",
                "student_id": "stu-2",
                "student_work": (
                    "u = x**2, v = sin(x)\n"
                    "u_prime = 2*x\n"
                    "v_prime = cos(x)\n"
                    "therefore dy/dx = 2*x*sin(x) + x**2*cos(x)"
                ),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["mark_result"] is not None
    assert body["mark_result"]["total_awarded"] == body["mark_result"]["total_available"]


@pytest.mark.asyncio
async def test_turn_endpoint_rejects_invalid_base64_image(monkeypatch):
    provider = ScriptedProvider([_concept_explain_intent_json()])
    monkeypatch.setattr(
        main_module, "_router", ModelRouter(providers={Provider.GROQ: provider, Provider.ANTHROPIC: provider})
    )

    async with await _client() as client:
        response = await client.post(
            "/turn",
            json={
                "raw_input": "check my work",
                "session_id": "s3",
                "student_id": "stu-3",
                "student_work_image_base64": "not-valid-base64!!!",
            },
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_turn_endpoint_falls_back_gracefully_when_the_model_is_unavailable(monkeypatch):
    class _AlwaysFailsProvider(ProviderClient):
        async def generate(self, *, spec, system, user, images=None):
            from app.llm.client import ModelUnavailableError

            raise ModelUnavailableError("simulated outage")

    failing = _AlwaysFailsProvider()
    monkeypatch.setattr(
        main_module, "_router", ModelRouter(providers={Provider.GROQ: failing, Provider.ANTHROPIC: failing})
    )

    async with await _client() as client:
        response = await client.post(
            "/turn",
            json={"raw_input": "hi", "session_id": "s4", "student_id": "stu-4"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ui_metadata"]["templated"] is True
