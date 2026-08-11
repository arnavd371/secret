"""
Math Solver + CAS Tool Agent (spec §2.2): a deterministic, non-generative
tool wrapping SymPy. This is the ground-truth oracle for every numeric/
symbolic claim the system makes — per spec §1.4: "Every mathematical
result the system is about to assert...is independently computed by the
Math Solver + CAS agent...The LLM-authored narrative is discarded/
regenerated if it disagrees with the CAS result."

Two conventions worth noting, since the spec's JSON example is
illustrative rather than a byte-for-byte contract for these two fields:
  - `input_latex` is real LaTeX (`sympy.latex(...)`), for display.
  - `result_exact` is stored as SymPy's plain, re-parseable `str(...)`
    form (e.g. "2*x*sin(x) + x**2*cos(x)"), not pre-rendered LaTeX, so
    `verify_claim` below can parse and compare it directly rather than
    trying to reverse LaTeX syntax. LaTeX rendering for display happens
    at the prompt-injection layer (see app/agents/templates.py) via
    `sympy.latex(sympy.sympify(result_exact))`.

Every public function here catches its own exceptions and returns a
`status=unverifiable` CASResult rather than raising — per spec: "On CAS
exception/timeout..., orchestrator forces response to hint/question tier
only, never asserts an unverified final answer." Nothing in this module
ever raises out to a caller.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Union

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.cas.models import CASOperation, CASResult, CASStatus

logger = logging.getLogger(__name__)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Default per-operation compute budget. Spec §2.2 latency budget: "p50 60
# ms, p95 500 ms (complex symbolic ops can spike)" — generous relative to
# that so a slow-but-real integral doesn't spuriously fail, but bounded so
# a pathological expression can't hang a request indefinitely.
DEFAULT_TIMEOUT_SECONDS = 3.0

# Numeric tolerance for exact-form comparison, pinned to spec §1.4:
# "abs(llm_value - cas_value) > 1e-6 for exact forms".
DEFAULT_TOLERANCE = 1e-6


def _parse(expression: str):
    return parse_expr(expression, transformations=_TRANSFORMATIONS)


def try_parse_expression(expression: str) -> Optional[sympy.Expr]:
    """Public, non-raising wrapper around `_parse` (spec §2.2's expression
    parser, same transformations: implicit multiplication, `^` as power).
    Phase 7's multimodal pipeline (app/multimodal/expression_parse.py)
    reuses this directly rather than standing up a second math parser —
    "does this parse as a real expression" is exactly the same question
    whether the text came from a chat message or an OCR transcription."""
    if not expression or not expression.strip():
        return None
    try:
        return _parse(expression)
    except Exception:  # noqa: BLE001 - any parse failure means "not parseable"
        return None


def _unverifiable(operation: CASOperation, expression: str, exc: Exception) -> CASResult:
    logger.warning("CAS operation %s failed on %r: %s", operation.value, expression, exc)
    return CASResult(
        status=CASStatus.UNVERIFIABLE,
        operation=operation,
        input_latex=expression,
        domain_notes=[f"CAS could not verify this: {exc}"],
    )


def _describe_differentiation(expr) -> list[str]:
    """Best-effort, non-exhaustive step labeling — not a full pedagogical
    solution-graph derivation (that belongs to the Misconception
    Diagnostician's step-diff algorithm, spec §8.3, a later phase)."""
    notes: list[str] = []

    if expr.is_Mul and len([f for f in expr.args if not f.is_number]) >= 2:
        notes.append("product_rule")

    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Function):
            args = node.args
            if args and not (len(args) == 1 and (args[0].is_Symbol or args[0].is_number)):
                notes.append("chain_rule")
                break
        if isinstance(node, sympy.Pow):
            base = node.args[0]
            if not (base.is_Symbol or base.is_number):
                notes.append("chain_rule")
                break

    if not notes:
        notes.append("standard_differentiation_rules")
    return notes


def differentiate(expression: str, variable: str = "x") -> CASResult:
    try:
        x = sympy.Symbol(variable)
        expr = _parse(expression)
        derivative = sympy.simplify(sympy.diff(expr, x))
        steps = _describe_differentiation(expr) + [f"d/d{variable}[{expr}] = {derivative}"]
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.DIFFERENTIATE,
            input_latex=sympy.latex(expr),
            result_exact=str(derivative),
            steps=steps,
        )
    except Exception as exc:  # noqa: BLE001 - any parse/compute failure degrades to unverifiable
        return _unverifiable(CASOperation.DIFFERENTIATE, expression, exc)


def integrate(expression: str, variable: str = "x") -> CASResult:
    try:
        x = sympy.Symbol(variable)
        expr = _parse(expression)
        integral = sympy.simplify(sympy.integrate(expr, x))
        if integral.has(sympy.zoo) or integral.has(sympy.nan) or integral.has(sympy.Integral):
            raise ValueError("integral not expressible in elementary closed form")
        result_exact = f"{integral} + C"
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.INTEGRATE,
            input_latex=sympy.latex(expr),
            result_exact=result_exact,
            steps=[f"\N{INTEGRAL} {expr} d{variable} = {result_exact}"],
        )
    except Exception as exc:  # noqa: BLE001
        return _unverifiable(CASOperation.INTEGRATE, expression, exc)


def solve_equation(equation: str, variable: str = "x") -> CASResult:
    try:
        x = sympy.Symbol(variable)
        if "=" in equation:
            lhs_str, rhs_str = equation.split("=", 1)
            eq = sympy.Eq(_parse(lhs_str), _parse(rhs_str))
        else:
            eq = sympy.Eq(_parse(equation), 0)

        solutions = sympy.solve(eq, x)
        if not solutions:
            raise ValueError("no solution found for the given equation")

        result_exact = ", ".join(f"{variable} = {sol}" for sol in solutions)
        decimal_at = float(solutions[0].evalf()) if len(solutions) == 1 and solutions[0].is_number else None
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.SOLVE,
            input_latex=sympy.latex(eq),
            result_exact=result_exact,
            result_decimal_at=decimal_at,
            steps=[f"solve {eq} for {variable}"],
        )
    except Exception as exc:  # noqa: BLE001
        return _unverifiable(CASOperation.SOLVE, equation, exc)


def simplify_expr(expression: str) -> CASResult:
    try:
        expr = _parse(expression)
        simplified = sympy.simplify(expr)
        decimal_at = float(simplified.evalf()) if simplified.is_number else None
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.SIMPLIFY,
            input_latex=sympy.latex(expr),
            result_exact=str(simplified),
            result_decimal_at=decimal_at,
            steps=[f"simplify({expr}) = {simplified}"],
        )
    except Exception as exc:  # noqa: BLE001
        return _unverifiable(CASOperation.SIMPLIFY, expression, exc)


def evaluate(expression: str, variable: str = "x", at: Optional[float] = None) -> CASResult:
    try:
        expr = _parse(expression)
        value = expr.subs(sympy.Symbol(variable), at) if at is not None else expr
        decimal_value = float(sympy.N(value))
        step = f"evaluate {expr} at {variable}={at}" if at is not None else f"evaluate {expr}"
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.EVALUATE,
            input_latex=sympy.latex(expr),
            result_exact=str(sympy.nsimplify(value)),
            result_decimal_at=decimal_value,
            steps=[step],
        )
    except Exception as exc:  # noqa: BLE001
        return _unverifiable(CASOperation.EVALUATE, expression, exc)


_DISPATCH = {
    CASOperation.DIFFERENTIATE: lambda expression, variable, at: differentiate(expression, variable),
    CASOperation.INTEGRATE: lambda expression, variable, at: integrate(expression, variable),
    CASOperation.SOLVE: lambda expression, variable, at: solve_equation(expression, variable),
    CASOperation.SIMPLIFY: lambda expression, variable, at: simplify_expr(expression),
    CASOperation.EVALUATE: lambda expression, variable, at: evaluate(expression, variable, at),
}


def _coerce_operation(operation: Union[CASOperation, str]) -> Optional[CASOperation]:
    if isinstance(operation, CASOperation):
        return operation
    try:
        return CASOperation(operation)
    except ValueError:
        return None


def run_cas_operation(
    operation: Union[CASOperation, str], expression: str, variable: str = "x", at: Optional[float] = None
) -> CASResult:
    op = _coerce_operation(operation)
    if op is None:
        # Fabricate a placeholder CASResult purely to carry the
        # unverifiable status/notes — the operation itself isn't one of
        # the enum's valid values, so there's no CASOperation to attach.
        return CASResult(
            status=CASStatus.UNVERIFIABLE,
            operation=CASOperation.SIMPLIFY,
            input_latex=expression,
            domain_notes=[f"CAS could not verify this: unsupported operation {operation!r}"],
        )
    handler = _DISPATCH.get(op)
    if handler is None:  # pragma: no cover - exhaustive over CASOperation
        return _unverifiable(op, expression, ValueError(f"unsupported CAS operation: {op}"))
    return handler(expression, variable, at)


async def run_cas_operation_async(
    operation: Union[CASOperation, str],
    expression: str,
    variable: str = "x",
    at: Optional[float] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> CASResult:
    """Async wrapper: SymPy calls are synchronous/CPU-bound, so they run in
    the default executor and are bounded by a real timeout — a slow
    symbolic operation degrades to `unverifiable` rather than blocking the
    turn indefinitely."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, run_cas_operation, operation, expression, variable, at),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        op = _coerce_operation(operation) or CASOperation.SIMPLIFY
        return _unverifiable(op, expression, TimeoutError(f"CAS operation timed out after {timeout_seconds}s"))


def _strip_variable_assignment(value: str) -> str:
    """"x = 2" -> "2": a claimed final answer is often phrased as an
    assignment, which isn't itself a parseable expression."""
    if "=" in value:
        return value.rsplit("=", 1)[-1].strip()
    return value.strip()


def verify_claim(cas_result: CASResult, claimed_value: str, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """
    Spec §1.4: "The LLM-authored narrative is discarded/regenerated if it
    disagrees with the CAS result beyond a defined tolerance
    (abs(llm_value - cas_value) > 1e-6 for exact forms; symbolic
    equivalence check via sympy.simplify(llm_expr - cas_expr) == 0 for
    algebraic forms)."

    Returns False (never verified) on any parse failure or when the CAS
    result itself isn't `ok` — an unverifiable ground truth can never
    confirm a claim.
    """
    if cas_result.status != CASStatus.OK:
        return False

    try:
        claimed_expr = _parse(_strip_variable_assignment(claimed_value))
    except Exception:  # noqa: BLE001
        return False

    if cas_result.result_decimal_at is not None:
        try:
            claimed_value_complex = complex(claimed_expr.evalf())
        except Exception:  # noqa: BLE001
            return False
        return (
            abs(claimed_value_complex.real - cas_result.result_decimal_at) <= tolerance
            and abs(claimed_value_complex.imag) <= tolerance
        )

    if cas_result.result_exact is None:
        return False

    try:
        cas_expr = _parse(cas_result.result_exact)
        return sympy.simplify(claimed_expr - cas_expr) == 0
    except Exception:  # noqa: BLE001
        return False
