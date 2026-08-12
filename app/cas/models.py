"""
Typed contract for the Math Solver + CAS Tool Agent (spec §2.2).

Field names and shape are pinned to the spec's JSON example exactly:

    {
      "status": "ok",
      "operation": "differentiate",
      "input_latex": "x^2 \\sin(x)",
      "result_exact": "2 x \\sin(x) + x^2 \\cos(x)",
      "result_decimal_at": null,
      "steps": ["product_rule", "d/dx[x^2]=2x", "d/dx[sin(x)]=cos(x)"],
      "domain_notes": []
    }

`result_decimal_at` holds the decimal value when the operation reduces to
a single number evaluated at a point (e.g. `evaluate` at x=2, or a `solve`
whose root is purely numeric); it stays None for a general symbolic
result like a derivative expression.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CASStatus(str, Enum):
    OK = "ok"
    # Per spec §2.2: "On CAS exception/timeout... return status=unverifiable;
    # orchestrator forces response to hint/question tier only, never
    # asserts an unverified final answer."
    UNVERIFIABLE = "unverifiable"


class CASOperation(str, Enum):
    DIFFERENTIATE = "differentiate"
    INTEGRATE = "integrate"
    SOLVE = "solve"
    SIMPLIFY = "simplify"
    EVALUATE = "evaluate"
    # Phase 12 additions (spec's IB AA HL syllabus coverage: matrices,
    # definite integrals, piecewise functions) — real SymPy operations,
    # not new parsing infrastructure bolted onto the five above.
    DETERMINANT = "determinant"
    MATRIX_MULTIPLY = "matrix_multiply"
    PIECEWISE_EVALUATE = "piecewise_evaluate"


class CASResult(BaseModel):
    status: CASStatus
    operation: CASOperation
    input_latex: str
    result_exact: Optional[str] = None
    result_decimal_at: Optional[float] = None
    steps: list[str] = Field(default_factory=list)
    domain_notes: list[str] = Field(default_factory=list)
