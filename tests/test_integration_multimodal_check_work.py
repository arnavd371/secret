"""
Integration coverage for Phase 7's multimodal check_work path: a
photographed submission run through the real ingestion pipeline
(real intake, real PIL preprocessing, real normalization/parsing/
confidence math), with only the math_ocr model call mocked, wired
through the same handle_turn() orchestrator as every other phase.
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.multimodal.models import ConfidenceTier, IntakeRejectionReason
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore


class ScriptedProvider(ProviderClient):
    """Same auto-pass-the-critic convention as every other integration
    test file; the math_ocr call shares this same scripted queue."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.ANTHROPIC)
        self.calls.append({"model": spec.model, "system": system, "user": user, "images": images})
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


def _check_work_intent_json() -> str:
    return json.dumps(
        {
            "intent": "check_work",
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": None,
            "assessment_mode_guess": "practice",
            "requires_multimodal_parse": True,
            "language": "en",
        }
    )


def _png_bytes(width: int = 900, height: int = 700) -> bytes:
    image = Image.new("RGB", (width, height), color=(220, 220, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_high_confidence_photo_is_graded_for_real_without_tutor_llm():
    transcription = (
        "u = x**2, v = sin(x)\nu_prime = 2*x\nv_prime = cos(x)\n"
        "therefore dy/dx = 2*x*sin(x) + x**2*cos(x)"
    )
    provider = ScriptedProvider([_check_work_intent_json(), transcription])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-mm-1",
        student_id="student-1",
        router=router,
        session_store=store,
        student_work_image=_png_bytes(),
    )

    assert result.ingestion_result is not None
    assert result.ingestion_result.confidence.tier == ConfidenceTier.HIGH
    assert result.mark_result is not None
    assert result.mark_result.total_awarded == result.mark_result.total_available == 2
    assert result.final_response.ui_metadata["graded"] is True
    # intent classify + math_ocr consumed, no Tutor generate call
    assert len(provider.calls) == 2
    assert provider.calls[1]["images"] is not None


@pytest.mark.asyncio
async def test_low_confidence_photo_asks_for_confirmation_instead_of_grading():
    provider = ScriptedProvider([_check_work_intent_json(), "x"])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-mm-2",
        student_id="student-2",
        router=router,
        session_store=store,
        student_work_image=_png_bytes(),
    )

    assert result.mark_result is None
    assert result.ingestion_result.confidence.tier == ConfidenceTier.LOW
    assert result.final_response.ui_metadata.get("multimodal_requires_confirmation") is True
    assert "retype" in result.final_response.text or "retake" in result.final_response.text


@pytest.mark.asyncio
async def test_rejected_intake_short_circuits_before_any_ocr_call():
    provider = ScriptedProvider([_check_work_intent_json(), "SHOULD NEVER BE CONSUMED"])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()

    tiny_image = _png_bytes(width=10, height=10)

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-mm-3",
        student_id="student-3",
        router=router,
        session_store=store,
        student_work_image=tiny_image,
    )

    assert result.mark_result is None
    assert result.ingestion_result.rejected is True
    assert result.ingestion_result.rejection_reason == IntakeRejectionReason.DIMENSIONS_TOO_SMALL
    assert "SHOULD NEVER BE CONSUMED" not in result.final_response.text
    # Only the intent-classification call happened; no math_ocr call.
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_explicit_student_work_text_takes_precedence_over_image():
    """If the caller already has typed student_work, the image (and its
    OCR call) must be ignored entirely rather than overwriting good text
    with a re-transcription."""
    provider = ScriptedProvider([_check_work_intent_json(), "SHOULD NEVER BE CONSUMED"])
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    store = InMemorySessionStateStore()

    work = "u = x**2, v = sin(x)\nu_prime = 2*x\nv_prime = cos(x)\ntherefore dy/dx = 2*x*sin(x) + x**2*cos(x)"

    result = await handle_turn(
        "can you check my work? differentiate x**2 * sin(x)",
        session_id="sess-mm-4",
        student_id="student-4",
        router=router,
        session_store=store,
        student_work=work,
        student_work_image=_png_bytes(),
    )

    assert result.ingestion_result is None
    assert result.mark_result is not None
    assert len(provider.calls) == 1
