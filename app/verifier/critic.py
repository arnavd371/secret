"""
Verifier / Critic Agent (spec §2.2, §13.5): an independent second-pass
checklist critique of an already-structurally-approved Tutor draft.

Deliberately a separate model call from generation (spec §2.8: "Never
collapse Verifier/Critic into the same call as the Tutor generation —
independence of the check is required for it to catch generation-time
leakage"). On failure/timeout it degrades to a conservative static check
(regex leak patterns + LaTeX well-formedness) rather than blocking the
turn outright — spec: "apply conservative static checks only... and pass
with a critic_degraded=true flag."
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from app.cas.models import CASResult, CASStatus
from app.knowledge.schemas import RetrievedChunk
from app.llm.client import ModelRouter, ModelUnavailableError
from app.llm.router_config import get_model_spec
from app.models.contracts import Action, ActionType
from app.verifier.models import CritiqueResult, CritiqueVerdict

logger = logging.getLogger(__name__)

# Same spirit as tutor_agent.py's leak patterns, deliberately re-declared
# here rather than imported — this is the *fallback* independent check,
# and keeping it self-contained means a bug in one doesn't silently take
# out the other.
_STATIC_LEAK_PATTERNS = [
    re.compile(r"\bthe answer is\b", re.IGNORECASE),
    re.compile(r"\bfinal answer\b", re.IGNORECASE),
    re.compile(r"\bsolution\s*:\s*[-+]?\d", re.IGNORECASE),
    re.compile(r"\btherefore,?\s+[a-zA-Z]\s*=\s*[-+]?\d", re.IGNORECASE),
    re.compile(r"^\s*[a-zA-Z]\s*=\s*[-+]?\d+(\.\d+)?\s*$", re.MULTILINE),
]

_ANSWER_PERMITTED_ACTIONS = (ActionType.EXPLAIN,)

_CRITIC_SYSTEM_PROMPT = """You are a strict, checklist-driven critic reviewing a math tutoring response \
before it is shown to a student. You do not generate content; you only evaluate the draft against the \
checklist and respond with ONLY a JSON object (no prose, no markdown fences):

{"verdict": "pass"|"revise"|"block", "violations": ["short description", ...]}

CHECKLIST:
- If BOUND ACTION is QUESTION, HINT, or CHALLENGE: the draft must NOT state a final numeric/symbolic answer.
- If BOUND ACTION is EXPLAIN: syllabus-specific claims should be grounded in the retrieved context provided; \
the draft must not contradict the CAS-verified result provided.
- The draft must stay on the bound action and not invent information unsupported by the provided context.
- Tone must be appropriate for a student: encouraging, not dismissive, not off-topic.

Return "block" for a clear, serious violation (e.g. a leaked final answer on a non-EXPLAIN action, or a \
claim that contradicts the CAS-verified result). Return "revise" for a moderate, fixable issue. Return \
"pass" if there are no violations.
"""


def _latex_well_formed(text: str) -> bool:
    return text.count(r"\(") == text.count(r"\)") and text.count(r"\[") == text.count(r"\]")


def _static_fallback_critique(draft: str, action: Action) -> CritiqueResult:
    violations: list[str] = []
    if action.action_type not in _ANSWER_PERMITTED_ACTIONS and any(p.search(draft) for p in _STATIC_LEAK_PATTERNS):
        violations.append("static check: draft appears to state a final answer on a non-EXPLAIN action")
    if not _latex_well_formed(draft):
        violations.append("static check: unbalanced LaTeX delimiters")

    verdict = CritiqueVerdict.BLOCK if violations else CritiqueVerdict.PASS
    return CritiqueResult(verdict=verdict, violations=violations, critic_degraded=True)


def _build_context_summary(action: Action, cas_result: Optional[CASResult], retrieved_chunks: Optional[list[RetrievedChunk]]) -> str:
    lines = [f"BOUND ACTION: {action.action_type.value} (move={action.move}, level={action.level})"]
    if cas_result is None:
        lines.append("CAS-VERIFIED RESULT: none computed for this turn")
    elif cas_result.status != CASStatus.OK:
        lines.append("CAS-VERIFIED RESULT: could not be verified this turn")
    else:
        lines.append(f"CAS-VERIFIED RESULT: {cas_result.operation.value} = {cas_result.result_exact}")

    if retrieved_chunks:
        lines.append("RETRIEVED CONTEXT: " + " | ".join(f"[{c.citation}] {c.text}" for c in retrieved_chunks[:3]))
    else:
        lines.append("RETRIEVED CONTEXT: none")
    return "\n".join(lines)


def _parse_critique_json(text: str) -> Optional[CritiqueResult]:
    try:
        payload = json.loads(text)
        verdict = CritiqueVerdict(payload["verdict"])
        violations = payload.get("violations", [])
        if not isinstance(violations, list):
            return None
        return CritiqueResult(verdict=verdict, violations=[str(v) for v in violations], critic_degraded=False)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Critic agent returned unparseable output: %s", exc)
        return None


async def critique_draft(
    draft: str,
    action: Action,
    router: ModelRouter,
    *,
    cas_result: Optional[CASResult] = None,
    retrieved_chunks: Optional[list[RetrievedChunk]] = None,
) -> CritiqueResult:
    spec = get_model_spec("critic_check")
    context_summary = _build_context_summary(action, cas_result, retrieved_chunks)
    user_prompt = f"{context_summary}\n\nDRAFT RESPONSE:\n{draft}"

    try:
        result = await asyncio.wait_for(
            router.call(capability="critic_check", system=_CRITIC_SYSTEM_PROMPT, user=user_prompt),
            timeout=spec.timeout_seconds,
        )
    except (ModelUnavailableError, asyncio.TimeoutError) as exc:
        logger.warning("Critic agent call failed (%s); degrading to static checks", exc)
        return _static_fallback_critique(draft, action)

    parsed = _parse_critique_json(result.text)
    if parsed is None:
        return _static_fallback_critique(draft, action)
    return parsed
