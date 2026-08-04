"""
Tutor agent: generates a response bound to an Action contract.

This is the component the prompt spec is most explicit about NOT letting
be a naive prompt-and-hope wrapper. Two independent layers enforce the
contract:

  1. templates.build_system_prompt(action) — a real, action-specific
     template that tells the model what it may and may not do.
  2. _violates_action_contract(draft, action) — a post-hoc structural
     check run on the accumulated draft BEFORE it is released. If the
     draft looks like it leaked a final answer on a HINT/QUESTION action,
     it is discarded outright and replaced with the templated fallback.
     The model's compliance with (1) is never trusted on its own.

Streaming uses a buffer-then-check strategy: the full draft is assembled
from the provider (real token streaming when the provider supports it),
the structural check runs on the complete text, and only the approved
final text is chunked back out to the caller. The check gates release —
nothing partial that hasn't passed the check is ever emitted.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

from app.agents.fallback import get_fallback_response
from app.agents.templates import build_system_prompt
from app.llm.client import ModelRouter, ModelUnavailableError
from app.llm.router_config import get_model_spec
from app.models.contracts import Action, ActionType, TutorResponse

logger = logging.getLogger(__name__)

# Action types where stating the final answer defeats the pedagogical
# purpose of the action and must be structurally blocked.
_LEAK_SENSITIVE_ACTIONS = (ActionType.HINT, ActionType.QUESTION)

# Heuristic patterns for "this looks like a final numeric/symbolic answer".
# Phase 1 uses regex/heuristics per spec; a real critic model is Phase 6.
_LEAK_PATTERNS = [
    re.compile(r"\bthe answer is\b", re.IGNORECASE),
    re.compile(r"\bfinal answer\b", re.IGNORECASE),
    re.compile(r"\bsolution\s*:\s*[-+]?\d", re.IGNORECASE),
    re.compile(r"\btherefore,?\s+[a-zA-Z]\s*=\s*[-+]?\d", re.IGNORECASE),
    re.compile(r"^\s*[a-zA-Z]\s*=\s*[-+]?\d+(\.\d+)?\s*$", re.MULTILINE),
    re.compile(r"\bequals\s+[-+]?\d+(\.\d+)?\s*$", re.IGNORECASE),
]

_STREAM_CHUNK_SIZE = 40


def _violates_action_contract(draft: str, action: Action) -> bool:
    if action.action_type not in _LEAK_SENSITIVE_ACTIONS:
        return False
    return any(pattern.search(draft) for pattern in _LEAK_PATTERNS)


async def generate(action: Action, raw_input: str, router: ModelRouter) -> TutorResponse:
    """Buffered generation: produce the full draft, run the structural
    check, and return either the approved draft or a templated fallback.
    Never raises — every failure path resolves to a valid TutorResponse."""

    if action.action_type == ActionType.REFUSE:
        raise ValueError("REFUSE must be hard-gated by the orchestrator before reaching the Tutor agent")

    system_prompt = build_system_prompt(action)
    spec = get_model_spec("tutor_generate")

    try:
        result = await asyncio.wait_for(
            router.call(capability="tutor_generate", system=system_prompt, user=raw_input),
            timeout=spec.timeout_seconds,
        )
        draft = result.text
    except (ModelUnavailableError, asyncio.TimeoutError) as exc:
        logger.warning("Tutor agent generation failed (%s); using templated fallback", exc)
        return get_fallback_response(action)

    if not draft or not draft.strip():
        logger.warning("Tutor agent returned empty draft; using templated fallback")
        return get_fallback_response(action)

    if _violates_action_contract(draft, action):
        logger.warning(
            "Tutor agent draft violated action contract for action_type=%s level=%s; discarding",
            action.action_type,
            action.level,
        )
        return get_fallback_response(action)

    return TutorResponse(
        text=draft.strip(),
        citations=[],
        ui_metadata={"action_type": action.action_type.value, "level": action.level, "templated": False},
    )


async def stream_response(action: Action, raw_input: str, router: ModelRouter) -> AsyncIterator[str]:
    """Buffer-then-check streaming: the gate in `generate` runs on the full
    draft first; only the approved text is ever chunked out to the caller."""
    response = await generate(action, raw_input, router)
    text = response.text
    for i in range(0, len(text), _STREAM_CHUNK_SIZE):
        yield text[i : i + _STREAM_CHUNK_SIZE]
