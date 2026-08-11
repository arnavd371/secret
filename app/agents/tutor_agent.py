"""
Tutor agent: generates a response bound to an Action contract.

This is the component the prompt spec is most explicit about NOT letting
be a naive prompt-and-hope wrapper. Independent, layered checks enforce
the contract — none of them depend on the model having listened to the
system prompt, and none of them depend on each other having caught a
problem first:

  1. `_violates_action_contract` (leak-check): a HINT/QUESTION/CHALLENGE
     draft that looks like it states a final numeric/symbolic answer is
     discarded and replaced with the templated fallback. CHALLENGE is
     leak-sensitive, not answer-permitted: per spec §1.5's decision table,
     it poses "challenge/extension question instead of full solve" — the
     student is meant to attempt the new item, not be handed its answer.
  2. CAS gating (spec §1.4): for EXPLAIN — the only action type actually
     permitted to state a final answer — if a `cas_result` was computed
     for this turn and is `unverifiable`, the draft is discarded for a
     "can't verify, let's work through it" fallback. If `cas_result` is
     `ok` and the draft's claimed final value disagrees with it beyond
     the spec's tolerance, the draft is discarded for a response built
     directly from the CAS ground truth instead.
  3. Grounding check (spec §13.6, app/verifier/grounding.py): for a draft
     with retrieved_chunks, flags claims not actually supported by the
     cited context.
  4. Verifier/Critic (spec §13.5, app/verifier/critic.py): an
     *independent second model call* — deliberately not the same call as
     generation (spec §2.8) — that reviews the already-structurally-
     approved draft. "block" discards it for the templated fallback;
     "revise" triggers one bounded regeneration attempt with the
     critique's violations fed back as stricter constraints before
     falling back if that also fails.

CHALLENGE items are generated up front by the real Question Generation
Engine (app/questions/generator.py) — a CAS-verified, quality-gated
`GeneratedItem`, not something the LLM invents. The Tutor's job is only
to phrase the handoff; layers 1 and 4 both backstop it against
accidentally revealing the item's answer.

Streaming uses a buffer-then-check strategy: the full draft is assembled
from the provider (real token streaming when the provider supports it),
all of the above checks run on the complete text, and only the approved
final text is chunked back out to the caller. The checks gate release —
nothing partial that hasn't passed them is ever emitted.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator, Optional

from app.agents.fallback import (
    build_cas_grounded_response,
    build_cas_unverifiable_response,
    get_fallback_response,
)
from app.agents.templates import build_system_prompt
from app.cas.models import CASResult, CASStatus
from app.cas.solver import verify_claim
from app.knowledge.retriever import RETRIEVAL_SCORE_THRESHOLD, is_grounded
from app.knowledge.schemas import RetrievedChunk
from app.llm.client import ModelRouter, ModelUnavailableError
from app.llm.router_config import get_model_spec
from app.memory.models import MemoryReadContext
from app.models.contracts import Action, ActionType, TutorResponse
from app.questions.models import GeneratedItem
from app.verifier.critic import critique_draft
from app.verifier.grounding import check_grounding
from app.verifier.models import CritiqueVerdict

logger = logging.getLogger(__name__)

# Action types where stating the final answer defeats the pedagogical
# purpose of the action and must be structurally blocked.
_LEAK_SENSITIVE_ACTIONS = (ActionType.HINT, ActionType.QUESTION, ActionType.CHALLENGE)

# Action types permitted to state a final answer at all, and therefore the
# only ones CAS-gated against a ground-truth result.
_CAS_GATED_ACTIONS = (ActionType.EXPLAIN,)

# Heuristic patterns for "this looks like a final numeric/symbolic answer".
_LEAK_PATTERNS = [
    re.compile(r"\bthe answer is\b", re.IGNORECASE),
    re.compile(r"\bfinal answer\b", re.IGNORECASE),
    re.compile(r"\bsolution\s*:\s*[-+]?\d", re.IGNORECASE),
    re.compile(r"\btherefore,?\s+[a-zA-Z]\s*=\s*[-+]?\d", re.IGNORECASE),
    re.compile(r"^\s*[a-zA-Z]\s*=\s*[-+]?\d+(\.\d+)?\s*$", re.MULTILINE),
    re.compile(r"\bequals\s+[-+]?\d+(\.\d+)?\s*$", re.IGNORECASE),
]

# Patterns used to extract (not just detect) a claimed final value from an
# EXPLAIN draft, for comparison against the CAS ground truth.
_FINAL_CLAIM_PATTERNS = [
    re.compile(r"(?:the answer is|final answer(?: is)?|equals)\s*[:\-]?\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"^\s*[a-zA-Z]\s*=\s*([^\n]+)$", re.MULTILINE),
]

_STREAM_CHUNK_SIZE = 40


def _violates_action_contract(draft: str, action: Action, challenge_item: Optional[GeneratedItem]) -> bool:
    if action.action_type not in _LEAK_SENSITIVE_ACTIONS:
        return False
    if any(pattern.search(draft) for pattern in _LEAK_PATTERNS):
        return True
    # A CHALLENGE draft that states the generated item's own verified
    # answer is just as much a leak as matching the generic patterns above.
    if action.action_type == ActionType.CHALLENGE and challenge_item is not None:
        answer_fragment = challenge_item.correct_answer.value.strip()
        if answer_fragment and answer_fragment in draft:
            return True
    return False


def _extract_claimed_value(draft: str) -> Optional[str]:
    matches: list[str] = []
    for pattern in _FINAL_CLAIM_PATTERNS:
        matches.extend(match.group(1).strip().rstrip(".") for match in pattern.finditer(draft))
    return matches[-1] if matches else None


def _apply_cas_gate(draft: str, action: Action, cas_result: Optional[CASResult]) -> Optional[TutorResponse]:
    """Returns a replacement TutorResponse if the draft must be discarded
    per the CAS gate, or None if the draft may pass through untouched."""
    if action.action_type not in _CAS_GATED_ACTIONS or cas_result is None:
        return None

    if cas_result.status != CASStatus.OK:
        logger.warning("CAS result unverifiable for action_type=%s; downgrading draft", action.action_type)
        return build_cas_unverifiable_response(action)

    claimed_value = _extract_claimed_value(draft)
    if claimed_value is None:
        # The draft doesn't appear to state a discrete final value (e.g. a
        # pure conceptual explanation) — nothing to check against CAS.
        return None

    if not verify_claim(cas_result, claimed_value):
        logger.warning(
            "Tutor draft's claimed value %r disagreed with CAS result %r; substituting grounded response",
            claimed_value,
            cas_result.result_exact,
        )
        return build_cas_grounded_response(action, cas_result)

    return None


def _citations_for(retrieved_chunks: Optional[list[RetrievedChunk]]) -> list[str]:
    """Only chunks that individually clear the grounding threshold are
    cited — `is_grounded` alone only checks the top-ranked chunk, so a
    weaker chunk further down the same top-k list must not ride along on
    the top chunk's confidence."""
    if not retrieved_chunks or not is_grounded(retrieved_chunks):
        return []
    return [chunk.citation for chunk in retrieved_chunks if chunk.score >= RETRIEVAL_SCORE_THRESHOLD]


async def _call_tutor_model(system_prompt: str, raw_input: str, router: ModelRouter) -> Optional[str]:
    spec = get_model_spec("tutor_generate")
    try:
        result = await asyncio.wait_for(
            router.call(capability="tutor_generate", system=system_prompt, user=raw_input),
            timeout=spec.timeout_seconds,
        )
    except (ModelUnavailableError, asyncio.TimeoutError) as exc:
        logger.warning("Tutor agent generation failed (%s)", exc)
        return None
    draft = result.text
    return draft.strip() if draft and draft.strip() else None


async def _regenerate_with_stricter_constraints(
    action: Action,
    raw_input: str,
    router: ModelRouter,
    violations: list[str],
    *,
    cas_result: Optional[CASResult],
    retrieved_chunks: Optional[list[RetrievedChunk]],
    challenge_item: Optional[GeneratedItem],
    memory_context: Optional[MemoryReadContext],
) -> Optional[str]:
    """Spec §13.8's escalation policy: one bounded regeneration attempt
    with the critique's violations fed back as an explicit stricter
    constraint, re-checked against the exact same structural gates as the
    original draft. Returns None (never a partial/unchecked draft) if the
    regeneration also fails any check — the caller falls back to the
    templated response."""
    system_prompt = build_system_prompt(
        action,
        cas_result=cas_result,
        retrieved_chunks=retrieved_chunks,
        challenge_item=challenge_item,
        memory_context=memory_context,
    )
    system_prompt += (
        "\n\nA prior draft was rejected by an independent reviewer for: "
        + "; ".join(violations)
        + ". Regenerate the response, strictly avoiding these issues."
    )

    draft = await _call_tutor_model(system_prompt, raw_input, router)
    if draft is None:
        return None
    if _violates_action_contract(draft, action, challenge_item):
        return None
    if action.action_type in _CAS_GATED_ACTIONS and cas_result is not None and cas_result.status == CASStatus.OK:
        claimed_value = _extract_claimed_value(draft)
        if claimed_value is not None and not verify_claim(cas_result, claimed_value):
            return None
    return draft


async def generate(
    action: Action,
    raw_input: str,
    router: ModelRouter,
    *,
    cas_result: Optional[CASResult] = None,
    retrieved_chunks: Optional[list[RetrievedChunk]] = None,
    challenge_item: Optional[GeneratedItem] = None,
    memory_context: Optional[MemoryReadContext] = None,
) -> TutorResponse:
    """Buffered generation: produce the full draft, run every structural
    and verifier check, and return either the approved draft or a
    templated/CAS-grounded fallback. Never raises — every failure path
    resolves to a valid TutorResponse."""

    if action.action_type == ActionType.REFUSE:
        raise ValueError("REFUSE must be hard-gated by the orchestrator before reaching the Tutor agent")

    system_prompt = build_system_prompt(
        action,
        cas_result=cas_result,
        retrieved_chunks=retrieved_chunks,
        challenge_item=challenge_item,
        memory_context=memory_context,
    )

    draft = await _call_tutor_model(system_prompt, raw_input, router)
    if draft is None:
        return get_fallback_response(action, challenge_item=challenge_item)

    if _violates_action_contract(draft, action, challenge_item):
        logger.warning(
            "Tutor agent draft violated action contract for action_type=%s level=%s; discarding",
            action.action_type,
            action.level,
        )
        return get_fallback_response(action, challenge_item=challenge_item)

    cas_gated_response = _apply_cas_gate(draft, action, cas_result)
    if cas_gated_response is not None:
        # Already a fully-formed, templated/CAS-grounded response — not
        # freshly-generated LLM prose, so there's nothing for the
        # Verifier/Critic to usefully re-check here.
        return cas_gated_response

    approved_text = draft
    # Only chunks that actually clear the citation threshold represent a
    # real claim of "this response reflects retrieved context" — grounding-
    # checking the draft against candidates that never qualified (and so
    # were never cited) would fail almost any draft for the wrong reason.
    groundable_chunks = retrieved_chunks if retrieved_chunks and is_grounded(retrieved_chunks) else None
    grounding_result = check_grounding(approved_text, groundable_chunks)
    critique = await critique_draft(approved_text, action, router, cas_result=cas_result, retrieved_chunks=retrieved_chunks)

    if critique.verdict == CritiqueVerdict.BLOCK or not grounding_result.grounded:
        logger.warning(
            "Verifier/Critic rejected draft (verdict=%s, grounded=%s, violations=%s)",
            critique.verdict.value,
            grounding_result.grounded,
            critique.violations,
        )
        return get_fallback_response(action, challenge_item=challenge_item)

    if critique.verdict == CritiqueVerdict.REVISE:
        regenerated = await _regenerate_with_stricter_constraints(
            action,
            raw_input,
            router,
            critique.violations,
            cas_result=cas_result,
            retrieved_chunks=retrieved_chunks,
            challenge_item=challenge_item,
            memory_context=memory_context,
        )
        if regenerated is None:
            return get_fallback_response(action, challenge_item=challenge_item)
        approved_text = regenerated
        grounding_result = check_grounding(approved_text, groundable_chunks)
        if not grounding_result.grounded:
            return get_fallback_response(action, challenge_item=challenge_item)

    return TutorResponse(
        text=approved_text,
        citations=_citations_for(retrieved_chunks),
        ui_metadata={
            "action_type": action.action_type.value,
            "level": action.level,
            "templated": False,
            "critique_verdict": critique.verdict.value,
            "critic_degraded": critique.critic_degraded,
            "grounding_score": round(grounding_result.score, 4),
        },
    )


async def stream_response(
    action: Action,
    raw_input: str,
    router: ModelRouter,
    *,
    cas_result: Optional[CASResult] = None,
    retrieved_chunks: Optional[list[RetrievedChunk]] = None,
    challenge_item: Optional[GeneratedItem] = None,
    memory_context: Optional[MemoryReadContext] = None,
) -> AsyncIterator[str]:
    """Buffer-then-check streaming: the gates in `generate` run on the full
    draft first; only the approved text is ever chunked out to the caller."""
    response = await generate(
        action,
        raw_input,
        router,
        cas_result=cas_result,
        retrieved_chunks=retrieved_chunks,
        challenge_item=challenge_item,
        memory_context=memory_context,
    )
    text = response.text
    for i in range(0, len(text), _STREAM_CHUNK_SIZE):
        yield text[i : i + _STREAM_CHUNK_SIZE]
