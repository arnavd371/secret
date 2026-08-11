"""
Tests for the math_ocr call surface: uses the same MockProvider pattern
as every other router-calling agent in this codebase, so no real network
call or API key is needed.
"""

import pytest

from app.llm.client import ModelRouter, ModelUnavailableError, MockProvider
from app.llm.router_config import Provider
from app.multimodal.ocr import transcribe_image


def _router_with_canned_response(text: str) -> ModelRouter:
    return ModelRouter(providers={Provider.ANTHROPIC: MockProvider(canned_response=text)})


@pytest.mark.asyncio
async def test_transcribe_image_returns_model_text():
    router = _router_with_canned_response("y = x^2 + 3x - 5")
    result = await transcribe_image(router, b"fake-png-bytes")
    assert result.raw_text == "y = x^2 + 3x - 5"
    assert result.model  # populated from the math_ocr ModelSpec


@pytest.mark.asyncio
async def test_transcribe_image_passes_the_image_bytes_through_to_the_provider():
    provider = MockProvider(canned_response="x = 2")
    router = ModelRouter(providers={Provider.ANTHROPIC: provider})
    await transcribe_image(router, b"specific-image-bytes")
    assert len(provider.calls) == 1
    images = provider.calls[0]["images"]
    assert images is not None
    assert images[0].data == b"specific-image-bytes"


class _AlwaysFailsProvider:
    async def generate(self, *, spec, system, user, images=None):
        raise ModelUnavailableError("simulated vision outage")


@pytest.mark.asyncio
async def test_transcribe_image_propagates_model_unavailable_error():
    router = ModelRouter(providers={Provider.ANTHROPIC: _AlwaysFailsProvider()})
    with pytest.raises(ModelUnavailableError):
        await transcribe_image(router, b"fake-png-bytes")
