"""
Step segmentation (spec §10.2): splits a student's typed submission into
discrete WorkStep objects. Real, but deliberately simple — line/semicolon
splitting plus a CAS-parseability check, not the "lightweight semantic
segmentation model" the spec describes; that needs training data and a
classifier this phase doesn't have. Handwritten input isn't handled at
all (that's the OCR pipeline, spec §3.2, Phase 7).
"""

from __future__ import annotations

import re
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.examiner.models import StepType, WorkStep

_SPLIT_PATTERN = re.compile(r"[\n;]+")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
_FINAL_ANSWER_MARKERS = ("answer", "therefore", "hence", "so ", "thus")


def _try_normalize(text: str) -> Optional[str]:
    """If the step contains '=', try to parse the right-hand side as an
    expression. Returns None for free-text reasoning steps."""
    if "=" not in text:
        return None
    rhs = text.rsplit("=", 1)[-1].strip()
    try:
        parse_expr(rhs, transformations=_TRANSFORMATIONS)
    except Exception:  # noqa: BLE001
        return None
    return rhs


def _expressions_equivalent(a: str, b: str) -> bool:
    try:
        expr_a = parse_expr(a, transformations=_TRANSFORMATIONS)
        expr_b = parse_expr(b, transformations=_TRANSFORMATIONS)
        return sympy.simplify(expr_a - expr_b) == 0
    except Exception:  # noqa: BLE001
        return False


def _classify(text: str, is_last: bool, normalized_expr: Optional[str], given_expression: Optional[str]) -> StepType:
    if given_expression and normalized_expr and _expressions_equivalent(normalized_expr, given_expression):
        return StepType.RESTATEMENT_OF_GIVEN

    lowered = text.lower()
    looks_like_final_answer = normalized_expr is not None or any(marker in lowered for marker in _FINAL_ANSWER_MARKERS)
    if is_last and looks_like_final_answer:
        return StepType.FINAL_ANSWER

    if normalized_expr is not None:
        return StepType.ALGEBRAIC_MANIPULATION

    return StepType.JUSTIFICATION


def segment_work(raw_text: str, given_expression: Optional[str] = None) -> list[WorkStep]:
    segments = [s.strip() for s in _SPLIT_PATTERN.split(raw_text) if s.strip()]
    steps: list[WorkStep] = []
    for index, text in enumerate(segments):
        is_last = index == len(segments) - 1
        normalized_expr = _try_normalize(text)
        step_type = _classify(text, is_last, normalized_expr, given_expression)
        steps.append(WorkStep(step_index=index, raw_text=text, normalized_expr=normalized_expr, step_type=step_type))
    return steps
