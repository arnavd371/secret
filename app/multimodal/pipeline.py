"""
End-to-end multimodal ingestion pipeline (spec §3.2): intake -> preprocess
-> OCR -> normalize -> parse -> confidence-gate, wired into one function
so app/orchestrator/handle_turn.py has a single call surface, same
pattern as every other subsystem's `handle_turn` integration point.

Every stage after intake can fail or degrade; the pipeline never raises
out to its caller (matching app/cas/solver.py's convention: "nothing in
this module ever raises out to a caller"). A model-router outage on the
math_ocr call is treated as a graceful degradation, not a fatal error —
the student is asked to type their work instead.
"""

from __future__ import annotations

from app.llm.client import ModelRouter, ModelUnavailableError
from app.multimodal.confidence import score_confidence
from app.multimodal.expression_parse import parse_expression
from app.multimodal.intake import validate_intake
from app.multimodal.latex_normalize import normalize_transcription
from app.multimodal.models import (
    ConfidenceTier,
    IngestionResult,
    IntakeRejectionReason,
)
from app.multimodal.ocr import transcribe_image
from app.multimodal.preprocess import preprocess_image


async def ingest_image(router: ModelRouter, image_bytes: bytes) -> IngestionResult:
    intake = validate_intake(image_bytes)
    if not intake.accepted:
        return IngestionResult(
            intake=intake,
            rejected=True,
            rejection_reason=intake.rejection_reason,
        )

    try:
        preprocess = preprocess_image(image_bytes)
    except Exception as exc:  # noqa: BLE001 - any preprocessing failure degrades, doesn't crash the turn
        return IngestionResult(
            intake=intake,
            rejected=True,
            rejection_reason=IntakeRejectionReason.CORRUPT_IMAGE,
            notes=[f"preprocessing failed: {exc}"],
        )

    try:
        ocr = await transcribe_image(router, preprocess.processed_image_bytes)
    except ModelUnavailableError as exc:
        return IngestionResult(
            intake=intake,
            preprocess=preprocess,
            requires_confirmation=True,
            notes=[f"math_ocr unavailable, ask the student to type their work instead: {exc}"],
        )

    normalized = normalize_transcription(ocr.raw_text)
    expression = parse_expression(normalized.normalized_text)
    confidence = score_confidence(
        raw_ocr_text=ocr.raw_text,
        normalized_text=normalized.normalized_text,
        expression_parseable=expression.parseable,
    )

    if confidence.tier == ConfidenceTier.LOW:
        return IngestionResult(
            intake=intake,
            preprocess=preprocess,
            ocr=ocr,
            normalized=normalized,
            expression=expression,
            confidence=confidence,
            student_work=None,
            requires_confirmation=True,
            notes=["low-confidence transcription: ask the student to confirm, retype, or retake the photo"],
        )

    return IngestionResult(
        intake=intake,
        preprocess=preprocess,
        ocr=ocr,
        normalized=normalized,
        expression=expression,
        confidence=confidence,
        student_work=normalized.normalized_text,
        requires_confirmation=confidence.tier == ConfidenceTier.MEDIUM,
    )
