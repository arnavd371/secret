"""
LLM-authored item variants (spec §9.6): a second generation mode
alongside the parametric templates in generator.py. The LLM proposes a
new problem and a claimed answer; nothing it claims is trusted until the
same CAS oracle every other phase relies on independently verifies it,
exactly the discipline app.cas.solver.verify_claim already applies to a
Tutor draft's stated answer, just applied to item authoring instead of
tutoring. An item only reaches app.questions.models.GeneratedItem with
generation_mode="llm_variant" after that verification passed — there is
no code path that serves an unverified LLM-authored claim.

Real, bounded retry (the spec's own "verifier gating" framing): a claim
that fails CAS verification triggers one regeneration attempt with the
mismatch fed back as an explicit correction, then gives up rather than
looping indefinitely or ever serving an unverified item.

Distractor generation for an arbitrary LLM-authored stem is out of
scope: Phase 3's distractor generators (app/questions/distractors.py)
are template-specific, each keyed to one template's own known
misconception pattern. There's no equivalent generic distractor
generator for an arbitrary LLM-authored expression, so a variant item is
served with an empty distractor list rather than a fabricated one.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from app.cas.models import CASOperation, CASStatus
from app.cas.solver import run_cas_operation_async, verify_claim
from app.llm.client import ModelRouter, ModelUnavailableError
from app.questions.mark_scheme import build_mark_scheme
from app.questions.models import (
    CorrectAnswer,
    DifficultyEstimate,
    GeneratedItem,
    QualityGateReport,
    QualityGateResult,
)

logger = logging.getLogger(__name__)

LLM_VARIANT_TEMPLATE_ID = "LLM-VARIANT"
MAX_AUTHOR_ATTEMPTS = 2

_SYSTEM_PROMPT = """You are authoring a new IB Diploma Mathematics: Analysis and Approaches practice \
problem on a given topic. Respond with ONLY a JSON object (no prose, no markdown fences):

{"stem": "<the problem statement, as shown to a student>", "operation": "differentiate"|"integrate"|"solve"|"simplify"|"evaluate", "expression": "<the exact expression/equation to feed a CAS engine, plain-text syntax like x**2 + 3*x>", "variable": "<the variable, usually x>", "claimed_answer": "<your computed answer, in the same plain-text syntax>"}

The stem must be a genuinely new problem (different numbers/structure from a stock textbook example), \
appropriate for IB AA SL/HL, and solvable via the single stated `operation`. Do not include the answer in \
the stem."""

_REQUIRED_KEYS = ("stem", "operation", "expression", "variable", "claimed_answer")


def _build_user_prompt(topic_hint: Optional[str], correction: Optional[str]) -> str:
    topic_text = topic_hint or "a topic of your choice within IB AA calculus or algebra"
    prompt = f"Author one new practice problem on: {topic_text}."
    if correction:
        prompt += f"\n\nYour previous attempt was rejected: {correction}. Try again with a different problem."
    return prompt


def _parse_variant_response(text: str) -> Optional[dict]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in _REQUIRED_KEYS:
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            return None
    if payload["operation"] not in {op.value for op in CASOperation}:
        return None
    return payload


async def generate_llm_variant(
    topic_hint: Optional[str], router: ModelRouter, *, max_attempts: int = MAX_AUTHOR_ATTEMPTS
) -> Optional[GeneratedItem]:
    """Returns a real, CAS-verified GeneratedItem, or None if the LLM
    couldn't produce a verifiable one within `max_attempts`. Callers
    (app.orchestrator.handle_turn) already know how to fall back to the
    parametric template generator on None, same convention as
    app.questions.generator.ItemGenerationError."""
    correction: Optional[str] = None

    for _ in range(max_attempts):
        try:
            result = await router.call(
                capability="item_variant_author",
                system=_SYSTEM_PROMPT,
                user=_build_user_prompt(topic_hint, correction),
            )
        except ModelUnavailableError as exc:
            logger.warning("LLM item-variant authoring call failed (%s)", exc)
            return None

        parsed = _parse_variant_response(result.text)
        if parsed is None:
            correction = "the response wasn't valid JSON in the required shape"
            continue

        operation = CASOperation(parsed["operation"])
        cas_result = await run_cas_operation_async(operation, parsed["expression"], parsed["variable"])
        if cas_result.status != CASStatus.OK:
            correction = f"the expression {parsed['expression']!r} couldn't be verified by the CAS engine"
            continue

        if not verify_claim(cas_result, parsed["claimed_answer"]):
            correction = (
                f"your claimed answer {parsed['claimed_answer']!r} didn't match the independently "
                f"computed result {cas_result.result_exact!r}"
            )
            continue

        item_id = f"ITEM-LLM-{uuid.uuid4().hex[:12]}"
        quality_gate_report = QualityGateReport(
            item_id=item_id,
            results=[
                QualityGateResult(gate="solvability", passed=True, detail=f"CAS status={cas_result.status.value}"),
                QualityGateResult(
                    gate="claim_verification",
                    passed=True,
                    detail="LLM's claimed answer matched the independently computed CAS result",
                ),
            ],
        )
        return GeneratedItem(
            item_id=item_id,
            template_id=LLM_VARIANT_TEMPLATE_ID,
            template_version=0,
            sampled_parameters={},
            rendered_stem=parsed["stem"].strip(),
            calculator_mode="calculator",
            difficulty_estimate=DifficultyEstimate(b_param=0.0, source="llm_estimated"),
            correct_answer=CorrectAnswer(value=cas_result.result_exact or "", cas_verified=True),
            distractors=[],
            generation_mode="llm_variant",
            quality_gate_report=quality_gate_report,
            mark_scheme=build_mark_scheme(item_id, cas_result),
        )

    logger.warning("LLM item-variant authoring exhausted %d attempts without a verifiable item", max_attempts)
    return None
