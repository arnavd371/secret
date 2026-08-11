"""
Typed contracts for the Misconception Diagnostician Agent (spec §8).

Two detection tiers, both surfaced through the same `DiagnosisResult`
shape so the caller (app/orchestrator/handle_turn.py) doesn't need to
know which one fired:

  - PATTERN_MATCH (app/diagnostician/detectors.py): a real, deterministic
    SymPy check that the student's wrong answer is *exactly* what you'd
    get by applying one specific, named error to the actual problem at
    hand. High trust, confidence fixed at 1.0 — there's no ambiguity
    once the symbolic values line up.
  - MODEL_INFERENCE (app/diagnostician/model_fallback.py): an LLM call
    used only when no pattern matched, for the wider range of real
    mistakes that don't reduce to one of the catalogued exact forms.
    Lower trust by construction — the model reports its own confidence,
    which the write policy gates on.

`misconception_id=None` (with `method=None`) is a real, expected outcome:
no catalogued pattern matched and the fallback model didn't confidently
recognize one either. Not every wrong answer maps to a named
misconception, and this type says so honestly rather than forcing a
guess.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DiagnosisMethod(str, Enum):
    PATTERN_MATCH = "pattern_match"
    MODEL_INFERENCE = "model_inference"


class DiagnosisResult(BaseModel):
    misconception_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    method: Optional[DiagnosisMethod] = None
    evidence: str = ""
