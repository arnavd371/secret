"""
Model-inference fallback (spec §8, "MODEL_INFERENCE" tier): used only
when app.diagnostician.detectors couldn't match the student's wrong
answer to any catalogued pattern for real. Deliberately constrained to
choosing from the fixed catalog (or reporting none) rather than free-
form diagnosis — same "classify against a checklist, don't invent"
posture as the Verifier/Critic (app/verifier/critic.py), which this
module's failure handling deliberately mirrors: an unparseable response
or a call failure degrades to "no diagnosis" rather than blocking or
fabricating one.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.diagnostician.catalog import MISCONCEPTION_CATALOG
from app.diagnostician.models import DiagnosisMethod, DiagnosisResult
from app.llm.client import ModelRouter, ModelUnavailableError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a diagnostic classifier reviewing one incorrect math answer to identify which, \
if any, of a fixed set of known misconceptions caused it. You do not generate tutoring content; you only \
classify, and respond with ONLY a JSON object (no prose, no markdown fences):

{"misconception_id": "<one of the catalog IDs below>" or null, "confidence": <0.0-1.0>, "evidence": "<one short sentence>"}

Return null for misconception_id if the wrong answer doesn't clearly match any catalogued misconception \
(e.g. it looks like an arithmetic slip, or there isn't enough information to tell). Do not guess a \
misconception_id you're not confident about — a low-confidence null is more useful than a wrong guess.

KNOWN MISCONCEPTIONS:
"""


def _build_system_prompt() -> str:
    catalog_lines = "\n".join(f"- {mid}: {desc}" for mid, desc in MISCONCEPTION_CATALOG.items())
    return _SYSTEM_PROMPT + catalog_lines


def _build_user_prompt(
    operation: str, expression: str, correct_value: str, student_value: str
) -> str:
    return (
        f"Operation: {operation}\n"
        f"Problem: {expression}\n"
        f"Correct answer (CAS-verified): {correct_value}\n"
        f"Student's answer: {student_value}\n\n"
        "Which known misconception, if any, produced the student's answer?"
    )


def _parse_response(text: str) -> Optional[DiagnosisResult]:
    try:
        payload = json.loads(text)
        misconception_id = payload.get("misconception_id")
        confidence = float(payload.get("confidence", 0.0))
        evidence = str(payload.get("evidence", ""))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Misconception fallback returned unparseable output: %s", exc)
        return None

    if misconception_id is not None:
        if not isinstance(misconception_id, str) or misconception_id not in MISCONCEPTION_CATALOG:
            logger.warning("Misconception fallback returned an unknown id: %r", misconception_id)
            return None
        confidence = max(0.0, min(1.0, confidence))
        return DiagnosisResult(
            misconception_id=misconception_id,
            confidence=confidence,
            method=DiagnosisMethod.MODEL_INFERENCE,
            evidence=evidence,
        )

    return DiagnosisResult(misconception_id=None, confidence=0.0, method=None, evidence=evidence)


async def diagnose_via_model(
    router: ModelRouter, operation: str, expression: str, correct_value: str, student_value: str
) -> DiagnosisResult:
    """Never raises: a call failure degrades to "no diagnosis" exactly
    like a parse failure does, consistent with every other model call in
    this codebase never blocking the turn on its own outage."""
    try:
        result = await router.call(
            capability="misconception_diagnose",
            system=_build_system_prompt(),
            user=_build_user_prompt(operation, expression, correct_value, student_value),
        )
    except ModelUnavailableError as exc:
        logger.warning("Misconception fallback call failed (%s); no diagnosis", exc)
        return DiagnosisResult(misconception_id=None, confidence=0.0, method=None, evidence=f"diagnosis unavailable: {exc}")

    parsed = _parse_response(result.text)
    if parsed is None:
        return DiagnosisResult(misconception_id=None, confidence=0.0, method=None, evidence="diagnosis unavailable: unparseable model response")
    return parsed
