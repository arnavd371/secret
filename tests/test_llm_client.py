"""
Tests for GroqProvider: a real httpx-based client against Groq's
OpenAI-compatible chat completions API, mocked at the transport layer
(httpx.MockTransport) rather than the network - the request actually
gets built and parsed by real code, only the socket is faked.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from app.llm.client import GroqProvider, ImageInput, ModelRouter, ModelUnavailableError
from app.llm.router_config import ModelSpec, Provider

_SPEC = ModelSpec(provider=Provider.GROQ, model="llama-3.3-70b-versatile", max_tokens=256, temperature=0.2)


def _client_with_handler(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_response(text: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})

    return handler


@pytest.mark.asyncio
async def test_generate_returns_the_real_parsed_response_text():
    client = _client_with_handler(_ok_response("2*x"))
    provider = GroqProvider(api_key="fake-key", client=client)

    result = await provider.generate(spec=_SPEC, system="you are a tutor", user="differentiate x**2")

    assert result.text == "2*x"
    assert result.model == "llama-3.3-70b-versatile"
    assert result.provider == Provider.GROQ


@pytest.mark.asyncio
async def test_generate_sends_the_real_request_shape():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = GroqProvider(api_key="fake-key", client=_client_with_handler(handler))
    await provider.generate(spec=_SPEC, system="sys prompt", user="user msg")

    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["auth"] == "Bearer fake-key"
    assert captured["body"]["model"] == "llama-3.3-70b-versatile"
    assert captured["body"]["max_tokens"] == 256
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys prompt"}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "user msg"}


@pytest.mark.asyncio
async def test_generate_with_images_sends_real_multimodal_content_blocks():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x = 3"}}]})

    provider = GroqProvider(api_key="fake-key", client=_client_with_handler(handler))
    image = ImageInput(data=b"fake-png-bytes", media_type="image/png")
    await provider.generate(spec=_SPEC, system="sys", user="what does this say?", images=[image])

    blocks = captured["body"]["messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "what does this say?"}
    assert blocks[1]["type"] == "image_url"
    expected_data_uri = f"data:image/png;base64,{base64.b64encode(b'fake-png-bytes').decode('ascii')}"
    assert blocks[1]["image_url"]["url"] == expected_data_uri


@pytest.mark.asyncio
async def test_generate_raises_model_unavailable_when_no_api_key_configured():
    provider = GroqProvider(api_key=None, client=_client_with_handler(_ok_response("irrelevant")))
    with pytest.raises(ModelUnavailableError, match="GROQ_API_KEY"):
        await provider.generate(spec=_SPEC, system="s", user="u")


@pytest.mark.asyncio
async def test_generate_raises_model_unavailable_on_a_real_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    provider = GroqProvider(api_key="bad-key", client=_client_with_handler(handler))
    with pytest.raises(ModelUnavailableError, match="Groq call failed"):
        await provider.generate(spec=_SPEC, system="s", user="u")


@pytest.mark.asyncio
async def test_generate_raises_model_unavailable_on_malformed_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = GroqProvider(api_key="fake-key", client=_client_with_handler(handler))
    with pytest.raises(ModelUnavailableError):
        await provider.generate(spec=_SPEC, system="s", user="u")


@pytest.mark.asyncio
async def test_default_stream_shim_yields_the_full_text_in_one_chunk():
    provider = GroqProvider(api_key="fake-key", client=_client_with_handler(_ok_response("full answer")))
    chunks = [chunk async for chunk in provider.stream(spec=_SPEC, system="s", user="u")]
    assert chunks == ["full answer"]


def test_model_router_registers_both_anthropic_and_groq_by_default():
    router = ModelRouter()
    assert Provider.ANTHROPIC in router._providers
    assert Provider.GROQ in router._providers
