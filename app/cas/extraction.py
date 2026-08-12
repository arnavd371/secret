"""
Best-effort extraction of a CAS-checkable task from raw student text.

This is a deliberately small keyword/regex heuristic, not real NLU or a
LaTeX/OCR pipeline (that's spec §3.2, Phase 7 — Multimodal Ingestion
Pipeline). It exists so Phase 2's CAS verification has something to run
against when a student types a plain-text math request; when nothing
recognizable is found it returns None, and the turn proceeds without CAS
grounding exactly as Phase 1 did. A missed extraction is not a bug — it's
the expected degradation mode until the real pipeline lands.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from app.cas.models import CASOperation


class MathTask(BaseModel):
    operation: CASOperation
    expression: str
    variable: str = "x"
    at: Optional[float] = None
    # Phase 12: paired with `at` (as the lower bound) for a real definite
    # integral. None for every other operation, including an indefinite
    # integral extracted without "from A to B".
    upper_at: Optional[float] = None


_FILLER_SUFFIX = re.compile(r"\s*\b(for me|for you|please)\b[\s.,:;!?]*$", re.IGNORECASE)


def _clean(expr: str) -> str:
    expr = _FILLER_SUFFIX.sub("", expr)
    return expr.strip(" .,:;!?\n")


def _extract_solve(text: str) -> Optional[MathTask]:
    match = re.search(r"solve\s+(.+)", text, re.IGNORECASE)
    if not match:
        return None
    tail = match.group(1)
    variable_match = re.search(r"\bfor\s+([a-zA-Z])\b", tail, re.IGNORECASE)
    variable = variable_match.group(1) if variable_match else "x"
    expr_part = re.sub(r"\bfor\s+[a-zA-Z]\b.*", "", tail, flags=re.IGNORECASE)
    expr_part = _clean(expr_part)
    if "=" not in expr_part:
        return None
    return MathTask(operation=CASOperation.SOLVE, expression=expr_part, variable=variable)


def _extract_differentiate(text: str) -> Optional[MathTask]:
    match = re.search(
        r"(?:differentiate|derivative of|find (?:the )?derivative of)\s+(.+)", text, re.IGNORECASE
    )
    if not match:
        return None
    return MathTask(operation=CASOperation.DIFFERENTIATE, expression=_clean(match.group(1)))


def _extract_definite_integral(text: str) -> Optional[MathTask]:
    """"integrate X from A to B [with respect to VAR]" - checked before
    the general indefinite-integral extractor below, since it's the more
    specific pattern (both share the "integrate" keyword)."""
    match = re.search(
        r"(?:integrate|integral of|find (?:the )?integral of)\s+(.+?)\s+from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    expr_part, lower_str, upper_str = match.groups()
    variable_match = re.search(r"\bwith\s+respect\s+to\s+([a-zA-Z])\b", text, re.IGNORECASE)
    variable = variable_match.group(1) if variable_match else "x"
    return MathTask(
        operation=CASOperation.INTEGRATE,
        expression=_clean(expr_part),
        variable=variable,
        at=float(lower_str),
        upper_at=float(upper_str),
    )


def _extract_integrate(text: str) -> Optional[MathTask]:
    match = re.search(
        r"(?:integrate|integral of|antiderivative of|find (?:the )?integral of)\s+(.+)", text, re.IGNORECASE
    )
    if not match:
        return None
    tail = match.group(1)
    variable_match = re.search(r"\bwith\s+respect\s+to\s+([a-zA-Z])\b", tail, re.IGNORECASE)
    variable = variable_match.group(1) if variable_match else "x"
    expr_part = re.sub(r"\bwith\s+respect\s+to\s+[a-zA-Z]\b.*", "", tail, flags=re.IGNORECASE)
    return MathTask(operation=CASOperation.INTEGRATE, expression=_clean(expr_part), variable=variable)


def _extract_determinant(text: str) -> Optional[MathTask]:
    match = re.search(r"determinant\s+of\s+(\[\[.+\]\])", text, re.IGNORECASE)
    if not match:
        return None
    return MathTask(operation=CASOperation.DETERMINANT, expression=match.group(1).strip())


def _extract_evaluate(text: str) -> Optional[MathTask]:
    match = re.search(
        r"evaluate\s+(.+?)\s+at\s+([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if not match:
        return None
    return MathTask(
        operation=CASOperation.EVALUATE,
        expression=_clean(match.group(1)),
        variable=match.group(2),
        at=float(match.group(3)),
    )


def _extract_simplify(text: str) -> Optional[MathTask]:
    match = re.search(r"simplify\s+(.+)", text, re.IGNORECASE)
    if not match:
        return None
    return MathTask(operation=CASOperation.SIMPLIFY, expression=_clean(match.group(1)))


# Order matters: "solve", "evaluate ... at x=", "determinant of", and the
# definite-integral form are all more specific patterns, checked before
# the more general differentiate/integrate/simplify keywords.
_EXTRACTORS = (
    _extract_solve,
    _extract_evaluate,
    _extract_determinant,
    _extract_definite_integral,
    _extract_differentiate,
    _extract_integrate,
    _extract_simplify,
)


def extract_math_task(raw_input: str) -> Optional[MathTask]:
    for extractor in _EXTRACTORS:
        task = extractor(raw_input)
        if task is not None:
            return task
    return None
