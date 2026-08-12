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
import re
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


def integrate(
    expression: str, variable: str = "x", lower: Optional[float] = None, upper: Optional[float] = None
) -> CASResult:
    """`lower`/`upper` given together (Phase 12) compute a real definite
    integral via SymPy's own (var, lower, upper) form — a genuine number
    (or symbolic closed form), never a "+ C" indefinite result. Either
    bound alone is treated as no bounds at all (an honest fallback rather
    than guessing which one was meant)."""
    try:
        x = sympy.Symbol(variable)
        expr = _parse(expression)

        if lower is not None and upper is not None:
            # Whole-number bounds become sympy.Integer rather than
            # sympy.Float — a bound of exactly 2.0 should still produce
            # an exact result like 8/3, not a floating-point 2.6667...
            # Python's float type can't distinguish "2" from "2.0", so
            # this is the CAS layer's own job, not extraction's.
            exact_lower = sympy.Integer(lower) if float(lower).is_integer() else sympy.Float(lower)
            exact_upper = sympy.Integer(upper) if float(upper).is_integer() else sympy.Float(upper)
            integral = sympy.simplify(sympy.integrate(expr, (x, exact_lower, exact_upper)))
            if integral.has(sympy.zoo) or integral.has(sympy.nan) or integral.has(sympy.Integral):
                raise ValueError("definite integral not expressible in closed form")
            result_exact = str(integral)
            decimal_value = float(integral.evalf()) if integral.is_number else None
            return CASResult(
                status=CASStatus.OK,
                operation=CASOperation.INTEGRATE,
                input_latex=sympy.latex(expr),
                result_exact=result_exact,
                result_decimal_at=decimal_value,
                steps=[f"\N{INTEGRAL} from {lower} to {upper} of {expr} d{variable} = {result_exact}"],
            )

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


_MATRIX_ROW_PATTERN = re.compile(r"\[([^\[\]]*)\]")


def _parse_matrix(text: str) -> sympy.Matrix:
    """"[[1,2],[3,4]]" -> sympy.Matrix([[1,2],[3,4]]). Each entry is
    parsed individually via `_parse` (the same restricted expression
    parser every other CAS function uses) — never a raw `eval` of the
    whole string, so a matrix entry can be a real expression ("2*x") but
    the parse itself stays exactly as safe as any other CAS input."""
    stripped = text.strip()
    if not (stripped.startswith("[[") and stripped.endswith("]]")):
        raise ValueError(f"not a matrix literal (expected \"[[...],[...]]\"): {text!r}")
    inner = stripped[1:-1]
    rows = _MATRIX_ROW_PATTERN.findall(inner)
    if not rows:
        raise ValueError(f"no matrix rows found in {text!r}")
    matrix_rows = [[_parse(entry.strip()) for entry in row.split(",") if entry.strip()] for row in rows]
    row_lengths = {len(row) for row in matrix_rows}
    if len(row_lengths) != 1:
        raise ValueError(f"matrix rows have inconsistent lengths: {row_lengths}")
    return sympy.Matrix(matrix_rows)


def determinant(matrix_text: str) -> CASResult:
    try:
        m = _parse_matrix(matrix_text)
        if m.rows != m.cols:
            raise ValueError(f"determinant requires a square matrix, got {m.rows}x{m.cols}")
        det = sympy.simplify(m.det())
        decimal_value = float(det.evalf()) if det.is_number else None
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.DETERMINANT,
            input_latex=sympy.latex(m),
            result_exact=str(det),
            result_decimal_at=decimal_value,
            steps=[f"det({m.tolist()}) = {det}"],
        )
    except Exception as exc:  # noqa: BLE001
        return _unverifiable(CASOperation.DETERMINANT, matrix_text, exc)


def matrix_multiply(combined_text: str) -> CASResult:
    """`combined_text`: "MATRIX_A * MATRIX_B", e.g.
    "[[1,2],[3,4]] * [[5,6],[7,8]]" — kept as one string (rather than two
    parameters) so this fits the same single-expression dispatch shape
    every other CAS operation uses."""
    try:
        if " * " not in combined_text:
            raise ValueError('expected "MATRIX_A * MATRIX_B", e.g. "[[1,2],[3,4]] * [[5,6],[7,8]]"')
        a_text, b_text = combined_text.split(" * ", 1)
        a = _parse_matrix(a_text.strip())
        b = _parse_matrix(b_text.strip())
        if a.cols != b.rows:
            raise ValueError(f"cannot multiply a {a.rows}x{a.cols} matrix by a {b.rows}x{b.cols} matrix")
        product = sympy.simplify(a * b)
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.MATRIX_MULTIPLY,
            input_latex=f"{sympy.latex(a)} \\times {sympy.latex(b)}",
            result_exact=str(product.tolist()),
            steps=[f"{a.tolist()} × {b.tolist()} = {product.tolist()}"],
        )
    except Exception as exc:  # noqa: BLE001
        return _unverifiable(CASOperation.MATRIX_MULTIPLY, combined_text, exc)


def _parse_condition(cond_text: str):
    """`<`, `<=`, `>`, `>=` parse correctly through the normal `_parse`
    (SymPy overloads them to build real Relational objects). `==`/`!=`
    do not — see evaluate_piecewise's docstring — so they're built
    explicitly via sympy.Eq/sympy.Ne instead."""
    if "!=" in cond_text:
        lhs, rhs = cond_text.split("!=", 1)
        return sympy.Ne(_parse(lhs.strip()), _parse(rhs.strip()))
    if "==" in cond_text:
        lhs, rhs = cond_text.split("==", 1)
        return sympy.Eq(_parse(lhs.strip()), _parse(rhs.strip()))
    return _parse(cond_text)


def evaluate_piecewise(pieces_text: str, variable: str, at: float) -> CASResult:
    """`pieces_text`: semicolon-separated "expr if cond" segments, e.g.
    "x**2 if x < 0; 2*x if x >= 0" — a final segment with no " if " acts
    as the default/otherwise branch. Conditions are parsed with the same
    `_parse` used everywhere else for `<`, `<=`, `>`, `>=`: applied to a
    SymPy symbol, Python's ordering operators already produce real SymPy
    Relational objects. `==`/`!=` are the one exception and are handled
    explicitly below — SymPy overloads Python's `==`/`!=` for structural
    equality checks (used to compare two expressions), not to build a
    symbolic Eq/Ne relational, so parsing "x != 0" through `_parse`
    silently produces the Python bool `True` instead of `Ne(x, 0)`. A
    real, easy-to-miss SymPy gotcha, not a hypothetical one — caught by
    this module's own tests.

    This is a structured-input API, not one with its own free-text
    chat-message extractor in app.cas.extraction — a natural-sounding
    request for a specific piecewise definition is hard to phrase as a
    single regexable sentence, so this is exposed for direct/programmatic
    use (and is exactly as real and CAS-verifiable as every other
    operation here) rather than forcing a brittle NL trigger onto it.
    """
    try:
        x = sympy.Symbol(variable)
        segments = [seg.strip() for seg in pieces_text.split(";") if seg.strip()]
        if not segments:
            raise ValueError("no piecewise segments given")

        piece_tuples = []
        for segment in segments:
            if " if " in segment:
                expr_part, cond_part = segment.split(" if ", 1)
                condition = _parse_condition(cond_part.strip())
            else:
                expr_part = segment
                condition = True
            piece_tuples.append((_parse(expr_part.strip()), condition))

        piecewise_expr = sympy.Piecewise(*piece_tuples)
        value = piecewise_expr.subs(x, at)
        if value.has(sympy.Piecewise) or value.has(sympy.nan) or not value.is_number:
            raise ValueError(f"no piecewise branch's condition matched {variable}={at}")
        decimal_value = float(value.evalf())
        return CASResult(
            status=CASStatus.OK,
            operation=CASOperation.PIECEWISE_EVALUATE,
            input_latex=sympy.latex(piecewise_expr),
            result_exact=str(sympy.nsimplify(value)),
            result_decimal_at=decimal_value,
            steps=[f"evaluate piecewise function at {variable}={at}: {value}"],
        )
    except Exception as exc:  # noqa: BLE001
        return _unverifiable(CASOperation.PIECEWISE_EVALUATE, pieces_text, exc)


_DISPATCH = {
    CASOperation.DIFFERENTIATE: lambda expression, variable, at, upper_at: differentiate(expression, variable),
    CASOperation.INTEGRATE: lambda expression, variable, at, upper_at: integrate(
        expression, variable, lower=at, upper=upper_at
    ),
    CASOperation.SOLVE: lambda expression, variable, at, upper_at: solve_equation(expression, variable),
    CASOperation.SIMPLIFY: lambda expression, variable, at, upper_at: simplify_expr(expression),
    CASOperation.EVALUATE: lambda expression, variable, at, upper_at: evaluate(expression, variable, at),
    CASOperation.DETERMINANT: lambda expression, variable, at, upper_at: determinant(expression),
    CASOperation.MATRIX_MULTIPLY: lambda expression, variable, at, upper_at: matrix_multiply(expression),
    CASOperation.PIECEWISE_EVALUATE: lambda expression, variable, at, upper_at: (
        evaluate_piecewise(expression, variable, at)
        if at is not None
        else _unverifiable(
            CASOperation.PIECEWISE_EVALUATE, expression, ValueError("piecewise_evaluate needs an 'at' value")
        )
    ),
}


def _coerce_operation(operation: Union[CASOperation, str]) -> Optional[CASOperation]:
    if isinstance(operation, CASOperation):
        return operation
    try:
        return CASOperation(operation)
    except ValueError:
        return None


def run_cas_operation(
    operation: Union[CASOperation, str],
    expression: str,
    variable: str = "x",
    at: Optional[float] = None,
    upper_at: Optional[float] = None,
) -> CASResult:
    """`upper_at` (Phase 12) is only meaningful for INTEGRATE — given
    together with `at` (as the lower bound), it makes this a real
    definite integral instead of an indefinite one. Every other
    operation ignores it; a single shared dispatch signature is simpler
    than a special-cased call path just for one operation."""
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
    return handler(expression, variable, at, upper_at)


async def run_cas_operation_async(
    operation: Union[CASOperation, str],
    expression: str,
    variable: str = "x",
    at: Optional[float] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    upper_at: Optional[float] = None,
) -> CASResult:
    """Async wrapper: SymPy calls are synchronous/CPU-bound, so they run in
    the default executor and are bounded by a real timeout — a slow
    symbolic operation degrades to `unverifiable` rather than blocking the
    turn indefinitely."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, run_cas_operation, operation, expression, variable, at, upper_at),
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
