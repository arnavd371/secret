"""
Write-gate for the Misconception Diagnostician's output (spec §8),
mirroring app.memory.write_policy's gating of mastery updates: not every
diagnosis is trustworthy enough to persist into the student's registry.
A PATTERN_MATCH is always trusted (confidence is fixed at 1.0 by
construction, since it's an exact symbolic match, not a guess). A
MODEL_INFERENCE diagnosis is trusted only above a real threshold — a
low-confidence model guess must not silently pollute the registry (and,
through it, a future turn's Tutor prompt via the ACTIVE MISCONCEPTIONS
slot).
"""

from __future__ import annotations

from app.diagnostician.models import DiagnosisResult

MODEL_INFERENCE_CONFIDENCE_THRESHOLD = 0.6


def should_write_diagnosis(diagnosis: DiagnosisResult) -> bool:
    if diagnosis.misconception_id is None:
        return False
    return diagnosis.confidence >= MODEL_INFERENCE_CONFIDENCE_THRESHOLD
