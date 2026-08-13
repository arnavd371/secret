"""
Model router: the single call surface every agent uses instead of talking to
a provider SDK directly.

    result = await router.call(capability="intent_classify", system=..., user=...)

Swapping `intent_classify` from Haiku to some other model, or from Anthropic
to another provider entirely, is a one-line change in router_config.py and
requires touching zero agent code.

`images` (Phase 7, spec §3.2) carries raw image bytes for a vision-capable
capability like `math_ocr` — optional and ignored by every text-only
capability, so this extension doesn't touch any existing call site.
"""

from __future__ import annotations

import abc
import base64
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from app.llm.router_config import ModelSpec, Provider, get_model_spec


class ModelUnavailableError(RuntimeError):
    """Raised when a provider call fails or times out. Orchestrator/agents
    must catch this and fall back to templated responses — it is not a bug
    to see this raised, it is an expected failure mode the system is
    designed around."""


@dataclass
class LLMCallResult:
    text: str
    model: str
    provider: Provider


@dataclass
class ImageInput:
    data: bytes
    media_type: str = "image/png"


class ProviderClient(abc.ABC):
    @abc.abstractmethod
    async def generate(
        self, *, spec: ModelSpec, system: str, user: str, images: Optional[list[ImageInput]] = None
    ) -> LLMCallResult: ...

    async def stream(
        self, *, spec: ModelSpec, system: str, user: str, images: Optional[list[ImageInput]] = None
    ) -> AsyncIterator[str]:
        """Default streaming shim: providers that support real token
        streaming should override this. Falls back to yielding the full
        response in one chunk."""
        result = await self.generate(spec=spec, system=system, user=user, images=images)
        yield result.text


class AnthropicProvider(ProviderClient):
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    async def generate(
        self, *, spec: ModelSpec, system: str, user: str, images: Optional[list[ImageInput]] = None
    ) -> LLMCallResult:
        if not self._api_key:
            raise ModelUnavailableError(
                "ANTHROPIC_API_KEY is not configured; cannot call the Anthropic provider."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the dep installed
            raise ModelUnavailableError("anthropic SDK is not installed") from exc

        try:
            client = anthropic.AsyncAnthropic(api_key=self._api_key)
            content: list[dict[str, Any]] = []
            for image in images or []:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.media_type,
                            "data": base64.b64encode(image.data).decode("ascii"),
                        },
                    }
                )
            content.append({"type": "text", "text": user})

            response = await client.messages.create(
                model=spec.model,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                system=system,
                messages=[{"role": "user", "content": content}],
                timeout=spec.timeout_seconds,
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)
        except Exception as exc:  # noqa: BLE001 - any provider failure becomes ModelUnavailableError
            raise ModelUnavailableError(f"Anthropic call failed: {exc}") from exc


class GroqProvider(ProviderClient):
    """Groq's chat completions API is OpenAI-compatible, so this talks to
    it directly over `httpx` rather than pulling in a dedicated SDK — one
    fewer dependency, and the request/response shape is simple enough
    that a thin wrapper is the honest amount of code for it, matching
    AnthropicProvider's own directness above.

    `client` is injectable (defaults to a lazily-created real
    `httpx.AsyncClient`) purely so tests can supply an `httpx.MockTransport`
    instead of hitting the network — the same dependency-injection seam
    AnthropicProvider gets for free from the `anthropic` SDK's own client
    object.
    """

    _BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, client: Optional[Any] = None) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client = client

    def _content_blocks(self, user: str, images: Optional[list[ImageInput]]) -> Any:
        if not images:
            return user
        blocks: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for image in images:
            data_uri = f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"
            blocks.append({"type": "image_url", "image_url": {"url": data_uri}})
        return blocks

    async def generate(
        self, *, spec: ModelSpec, system: str, user: str, images: Optional[list[ImageInput]] = None
    ) -> LLMCallResult:
        if not self._api_key:
            raise ModelUnavailableError("GROQ_API_KEY is not configured; cannot call the Groq provider.")

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without the dep installed
            raise ModelUnavailableError("httpx is not installed") from exc

        payload = {
            "model": spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._content_blocks(user, images)},
            ],
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        try:
            client = self._client or httpx.AsyncClient()
            response = await client.post(self._BASE_URL, json=payload, headers=headers, timeout=spec.timeout_seconds)
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"] or ""
            return LLMCallResult(text=text, model=spec.model, provider=Provider.GROQ)
        except Exception as exc:  # noqa: BLE001 - any provider failure becomes ModelUnavailableError
            raise ModelUnavailableError(f"Groq call failed: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()


class MockProvider(ProviderClient):
    """Used by tests and offline dev. Responses are supplied by the caller
    up front rather than generated, so tests are deterministic."""

    def __init__(self, canned_response: str = "") -> None:
        self.canned_response = canned_response
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, *, spec: ModelSpec, system: str, user: str, images: Optional[list[ImageInput]] = None
    ) -> LLMCallResult:
        self.calls.append({"spec": spec, "system": system, "user": user, "images": images})
        return LLMCallResult(text=self.canned_response, model=spec.model, provider=Provider.MOCK)


class ModelRouter:
    def __init__(self, providers: Optional[dict[Provider, ProviderClient]] = None) -> None:
        self._providers: dict[Provider, ProviderClient] = providers or {
            # Both are always registered so CAPABILITY_MODEL_MAP can name
            # either provider per capability without the caller having to
            # know which key is active — each provider client itself
            # raises ModelUnavailableError (not an import-time crash) if
            # its own API key isn't set, exactly like every other
            # provider failure this system already falls back around.
            Provider.ANTHROPIC: AnthropicProvider(),
            Provider.GROQ: GroqProvider(),
        }

    async def call(
        self, *, capability: str, system: str, user: str, images: Optional[list[ImageInput]] = None
    ) -> LLMCallResult:
        spec = get_model_spec(capability)
        provider = self._providers.get(spec.provider)
        if provider is None:
            raise ModelUnavailableError(f"No provider client registered for {spec.provider}")
        return await provider.generate(spec=spec, system=system, user=user, images=images)

    async def stream(
        self, *, capability: str, system: str, user: str, images: Optional[list[ImageInput]] = None
    ) -> AsyncIterator[str]:
        spec = get_model_spec(capability)
        provider = self._providers.get(spec.provider)
        if provider is None:
            raise ModelUnavailableError(f"No provider client registered for {spec.provider}")
        async for chunk in provider.stream(spec=spec, system=system, user=user, images=images):
            yield chunk
