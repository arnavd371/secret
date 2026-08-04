"""
Router/Intent agent (spec §2.2): one small/fast model call that classifies
the incoming turn into a structured IntentResult.

The confidence<0.6 fallback is implemented as real code below — the model
is never trusted to "know" it's uncertain and behave accordingly on its own.
Per spec: "On low-confidence classification (confidence < 0.6), default to
concept_explain intent with assessment_mode=practice (safest,
least-invasive default) and flag for human-review sampling." If the call
fails outright (timeout, provider error) or returns something that doesn't
parse into IntentResult, we fall back the same way.
"""

from __future__ import annotations

import asyncio
import json
import logging

from pydantic import ValidationError

from app.llm.client import ModelRouter, ModelUnavailableError
from app.llm.router_config import get_model_spec
from app.models.contracts import AssessmentMode, IntentResult, IntentType

logger = logging.getLogger(__name__)

# Below this confidence the classification is not trusted, and we fall back
# to the safe default action-set rather than let a low-confidence guess
# drive the decision policy. Value pinned to spec §2.2.
LOW_CONFIDENCE_THRESHOLD = 0.6

_DEFAULT_SUBJECT = "math_aa"

_INTENT_SYSTEM_PROMPT = """You are an intent classifier for a math tutoring \
assistant. Read the student's message and respond with ONLY a JSON object \
(no prose, no markdown fences) with exactly these keys:

  intent: one of "solve_request", "check_work", "concept_explain", \
"exam_prep", "ia_ee_help", "general_chat"
  confidence: float between 0 and 1
  subject: string, e.g. "math_aa"
  topic_hint: string or null, e.g. "calculus.differentiation.chain_rule"
  assessment_mode_guess: one of "practice", "homework_ungraded", \
"graded_take_home", "live_exam_simulation", "ia_ee"
  requires_multimodal_parse: boolean, true if the message references an \
image/photo/scan of work
  language: ISO 639-1 code, e.g. "en"
"""


def _safe_default_intent() -> IntentResult:
    return IntentResult(
        intent=IntentType.CONCEPT_EXPLAIN,
        confidence=LOW_CONFIDENCE_THRESHOLD,
        subject=_DEFAULT_SUBJECT,
        topic_hint=None,
        assessment_mode_guess=AssessmentMode.PRACTICE,
        requires_multimodal_parse=False,
        language="en",
    )


def _parse_intent_json(raw_text: str) -> IntentResult | None:
    try:
        payload = json.loads(raw_text)
        return IntentResult(**payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.warning("Router/Intent agent returned unparseable output: %s", exc)
        return None


def _apply_low_confidence_fallback(parsed: IntentResult) -> IntentResult:
    """Confidence < threshold: keep the observed confidence (useful for
    logging/eval) but force the intent onto the safe default action-set
    per spec, rather than trusting a shaky classification."""
    return parsed.model_copy(
        update={
            "intent": IntentType.CONCEPT_EXPLAIN,
            "assessment_mode_guess": AssessmentMode.PRACTICE,
        }
    )


async def classify_intent(raw_input: str, router: ModelRouter) -> IntentResult:
    spec = get_model_spec("intent_classify")
    try:
        result = await asyncio.wait_for(
            router.call(capability="intent_classify", system=_INTENT_SYSTEM_PROMPT, user=raw_input),
            timeout=spec.timeout_seconds,
        )
    except (ModelUnavailableError, asyncio.TimeoutError) as exc:
        logger.warning("Router/Intent agent call failed (%s); using safe default", exc)
        return _safe_default_intent()

    parsed = _parse_intent_json(result.text)
    if parsed is None:
        return _safe_default_intent()

    if parsed.confidence < LOW_CONFIDENCE_THRESHOLD:
        return _apply_low_confidence_fallback(parsed)

    return parsed
