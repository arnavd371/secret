"""
Typed contracts for the multimodal ingestion pipeline (spec §3.2): a
student photographs or screenshots handwritten/printed work, and the
pipeline turns that image into text the rest of the system (Phase 4's
grader, in particular) can already consume.

Reuses `app.examiner.models.ConfidenceTier` for the pipeline's final
gating decision rather than inventing a parallel HIGH/MEDIUM/LOW enum:
spec §3.2's three-tier threshold gating (auto-accept / confirm-with-
student / reject-and-ask-retype) is the same concept Phase 4 already
uses for grading confidence, just applied to a different measurement.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.examiner.models import ConfidenceTier


class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"


class IntakeRejectionReason(str, Enum):
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPT_IMAGE = "corrupt_image"
    TOO_LARGE_BYTES = "too_large_bytes"
    DIMENSIONS_TOO_SMALL = "dimensions_too_small"
    DIMENSIONS_TOO_LARGE = "dimensions_too_large"


class IntakeResult(BaseModel):
    """Result of the format/size/corruption gate that runs before any
    processing touches the image. Rejections here never reach the model
    router — they're cheap, deterministic, and free."""

    accepted: bool
    format: Optional[ImageFormat] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    size_bytes: int
    rejection_reason: Optional[IntakeRejectionReason] = None


class PreprocessResult(BaseModel):
    """Result of the real PIL preprocessing stage: grayscale, contrast
    normalization, binarization, and resizing to an OCR-friendly
    resolution. `processed_image_bytes` is always PNG-encoded regardless
    of the input format, since PNG is lossless and every downstream
    consumer (the vision model, tests) only needs one format to handle."""

    processed_image_bytes: bytes
    width_px: int
    height_px: int
    grayscale_applied: bool
    contrast_enhanced: bool
    binarized: bool
    resized: bool


class OCRResult(BaseModel):
    """Raw output of the math_ocr vision-model call (spec §3.2, §14.2),
    before any normalization. `raw_text` is the model's untouched
    transcription; everything downstream (LaTeX normalization,
    expression parsing) operates on this."""

    raw_text: str
    model: str


class NormalizedTranscription(BaseModel):
    """`raw_text` after LaTeX cleanup (spec §3.2's normalization step):
    stripped of markdown code fences, common LaTeX-command inconsistencies
    resolved, whitespace collapsed."""

    normalized_text: str
    latex_command_count: int


class ExpressionParseResult(BaseModel):
    """Whether the normalized transcription's mathematical content is
    parseable by the same CAS layer Phase 2 already uses (deliberate
    reuse, not a new parser) — a strong, cheap, real signal of
    transcription quality distinct from the vision model's own
    confidence."""

    parseable: bool
    parsed_expression: Optional[str] = None
    parse_error: Optional[str] = None


class ConfidenceBreakdown(BaseModel):
    """The composite confidence score's components (spec §3.2), kept
    visible rather than collapsed into a single number so a human
    reviewer (or a test) can see which signal pulled the score down."""

    ocr_length_signal: float = Field(ge=0.0, le=1.0)
    expression_parseable_signal: float = Field(ge=0.0, le=1.0)
    latex_well_formed_signal: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    tier: ConfidenceTier


class IngestionResult(BaseModel):
    """The pipeline's single end-to-end output (spec §3.2). `student_work`
    is populated (and safe to feed into Phase 4 grading) only when
    `confidence.tier` is HIGH or MEDIUM; on LOW it stays None and the
    caller must ask the student to confirm or retype instead."""

    intake: IntakeResult
    preprocess: Optional[PreprocessResult] = None
    ocr: Optional[OCRResult] = None
    normalized: Optional[NormalizedTranscription] = None
    expression: Optional[ExpressionParseResult] = None
    confidence: Optional[ConfidenceBreakdown] = None
    student_work: Optional[str] = None
    requires_confirmation: bool = False
    rejected: bool = False
    rejection_reason: Optional[IntakeRejectionReason] = None
    notes: list[str] = Field(default_factory=list)
