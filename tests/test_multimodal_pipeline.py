"""
End-to-end tests for the multimodal ingestion pipeline: real intake,
real PIL preprocessing, real normalization/parsing/confidence math, with
only the math_ocr model call mocked (same MockProvider pattern used by
every other integration test in this codebase).
"""

import io

import pytest
from PIL import Image

from app.llm.client import ModelRouter, ModelUnavailableError, MockProvider
from app.llm.router_config import Provider
from app.multimodal.models import ConfidenceTier, IntakeRejectionReason
from app.multimodal.pipeline import ingest_image


def _png_bytes(width: int = 900, height: int = 700) -> bytes:
    image = Image.new("RGB", (width, height), color=(220, 220, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _router_with_canned_response(text: str) -> ModelRouter:
    return ModelRouter(providers={Provider.GROQ: MockProvider(canned_response=text)})


class _AlwaysFailsProvider:
    async def generate(self, *, spec, system, user, images=None):
        raise ModelUnavailableError("simulated vision outage")


@pytest.mark.asyncio
async def test_high_confidence_transcription_is_usable_immediately():
    router = _router_with_canned_response("dy/dx = 5*(2*x + 1)**4 * 2")
    result = await ingest_image(router, _png_bytes())

    assert result.rejected is False
    assert result.confidence.tier == ConfidenceTier.HIGH
    assert result.student_work == "dy/dx = 5*(2*x + 1)**4 * 2"
    assert result.requires_confirmation is False
    assert result.preprocess is not None
    assert result.ocr is not None
    assert result.expression.parseable is True


@pytest.mark.asyncio
async def test_medium_confidence_transcription_requires_confirmation():
    router = _router_with_canned_response("some illegible scrawl here")
    result = await ingest_image(router, _png_bytes())

    assert result.rejected is False
    assert result.confidence.tier == ConfidenceTier.MEDIUM
    assert result.student_work == "some illegible scrawl here"
    assert result.requires_confirmation is True


@pytest.mark.asyncio
async def test_low_confidence_transcription_withholds_student_work():
    router = _router_with_canned_response("x")
    result = await ingest_image(router, _png_bytes())

    assert result.rejected is False
    assert result.confidence.tier == ConfidenceTier.LOW
    assert result.student_work is None
    assert result.requires_confirmation is True
    assert result.notes


@pytest.mark.asyncio
async def test_intake_rejection_short_circuits_before_any_model_call():
    provider = MockProvider(canned_response="should never be reached")
    router = ModelRouter(providers={Provider.GROQ: provider})
    tiny_image = _png_bytes(width=10, height=10)

    result = await ingest_image(router, tiny_image)

    assert result.rejected is True
    assert result.rejection_reason == IntakeRejectionReason.DIMENSIONS_TOO_SMALL
    assert result.preprocess is None
    assert result.ocr is None
    assert len(provider.calls) == 0


@pytest.mark.asyncio
async def test_corrupt_bytes_are_rejected_before_preprocessing():
    router = _router_with_canned_response("should never be reached")
    result = await ingest_image(router, b"not a real image")

    assert result.rejected is True
    assert result.rejection_reason == IntakeRejectionReason.CORRUPT_IMAGE
    assert result.preprocess is None


@pytest.mark.asyncio
async def test_ocr_outage_degrades_gracefully_instead_of_raising():
    router = ModelRouter(providers={Provider.GROQ: _AlwaysFailsProvider()})
    result = await ingest_image(router, _png_bytes())

    assert result.rejected is False
    assert result.ocr is None
    assert result.student_work is None
    assert result.requires_confirmation is True
    assert result.notes
    # preprocessing still ran for real before the model call failed
    assert result.preprocess is not None
