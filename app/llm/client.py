"""
Model router: the single call surface every agent uses instead of talking to
a provider SDK directly.

    result = await router.call(capability="intent_classify", system=..., user=...)

Swapping `intent_classify` from Haiku to some other model, or from Anthropic
to another provider entirely, is a one-line change in router_config.py and
requires touching zero agent code.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
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


class ProviderClient(abc.ABC):
    @abc.abstractmethod
    async def generate(self, *, spec: ModelSpec, system: str, user: str) -> LLMCallResult: ...

    async def stream(self, *, spec: ModelSpec, system: str, user: str) -> AsyncIterator[str]:
        """Default streaming shim: providers that support real token
        streaming should override this. Falls back to yielding the full
        response in one chunk."""
        result = await self.generate(spec=spec, system=system, user=user)
        yield result.text


class AnthropicProvider(ProviderClient):
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    async def generate(self, *, spec: ModelSpec, system: str, user: str) -> LLMCallResult:
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
            response = await client.messages.create(
                model=spec.model,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
                timeout=spec.timeout_seconds,
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)
        except Exception as exc:  # noqa: BLE001 - any provider failure becomes ModelUnavailableError
            raise ModelUnavailableError(f"Anthropic call failed: {exc}") from exc


class MockProvider(ProviderClient):
    """Used by tests and offline dev. Responses are supplied by the caller
    up front rather than generated, so tests are deterministic."""

    def __init__(self, canned_response: str = "") -> None:
        self.canned_response = canned_response
        self.calls: list[dict[str, Any]] = []

    async def generate(self, *, spec: ModelSpec, system: str, user: str) -> LLMCallResult:
        self.calls.append({"spec": spec, "system": system, "user": user})
        return LLMCallResult(text=self.canned_response, model=spec.model, provider=Provider.MOCK)


class ModelRouter:
    def __init__(self, providers: Optional[dict[Provider, ProviderClient]] = None) -> None:
        self._providers: dict[Provider, ProviderClient] = providers or {
            Provider.ANTHROPIC: AnthropicProvider(),
        }

    async def call(self, *, capability: str, system: str, user: str) -> LLMCallResult:
        spec = get_model_spec(capability)
        provider = self._providers.get(spec.provider)
        if provider is None:
            raise ModelUnavailableError(f"No provider client registered for {spec.provider}")
        return await provider.generate(spec=spec, system=system, user=user)

    async def stream(self, *, capability: str, system: str, user: str) -> AsyncIterator[str]:
        spec = get_model_spec(capability)
        provider = self._providers.get(spec.provider)
        if provider is None:
            raise ModelUnavailableError(f"No provider client registered for {spec.provider}")
        async for chunk in provider.stream(spec=spec, system=system, user=user):
            yield chunk
