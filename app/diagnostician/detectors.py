"""
Real, deterministic pattern detectors (spec §8, "PATTERN_MATCH" tier).

Unlike Phase 3's distractor generators (app/questions/distractors.py),
which bake a misconception into a *generated* item's known template
parameters, these detectors work backwards from an arbitrary real
problem: given the actual CAS operation/expression/variable extracted
from a check_work turn (whatever a student happened to type), compute
what a specific named error *would* produce for that exact problem, and
check whether the student's stated answer matches that hypothesis
exactly (symbolic equivalence, same tolerance convention as
app.cas.solver.verify_claim). A generated item's distractor and a
diagnosed misconception here can therefore share the same catalog ID
even though they're computed by completely different code paths — one
forward (bait a specific wrong answer), one backward (recognize one).

A match here is high-trust by construction: the student's answer isn't
just "wrong," it's *exactly* the value this specific error produces for
this specific problem, which is strong evidence of which error was made
(as opposed to an unrelated arithmetic slip that happens to be wrong in
some other way).
"""

from __future__ import annotations

from typing import Optional

import sympy

from app.cas.models import CASOperation
from app.cas.solver import try_parse_expression


def _expressions_equivalent(a: str, b: str) -> bool:
    expr_a = try_parse_expression(a)
    expr_b = try_parse_expression(b)
    if expr_a is None or expr_b is None:
        return False
    try:
        return sympy.simplify(expr_a - expr_b) == 0
    except Exception:  # noqa: BLE001 - any comparison failure means "not a match"
        return False


def _strip_assignment(value: str) -> str:
    if "=" in value:
        return value.rsplit("=", 1)[-1].strip()
    return value.strip()


def _split_roots(value: str) -> list[str]:
    return [part.rsplit("=", 1)[-1].strip() for part in value.split(",") if part.strip()]


def _differentiation_hypotheses(expr: sympy.Expr, variable: str) -> list[tuple[str, str]]:
    x = sympy.Symbol(variable)
    correct = sympy.simplify(sympy.diff(expr, x))
    hypotheses: list[tuple[str, str]] = []

    # MISC-CALC-GEN-DROPPED-TERM: a sum of >=2 terms, one dropped entirely.
    if isinstance(expr, sympy.Add) and len(expr.args) >= 2:
        for term in expr.args:
            remainder = expr - term
            wrong = sympy.simplify(sympy.diff(remainder, x))
            if wrong != correct:
                hypotheses.append(("MISC-CALC-GEN-DROPPED-TERM", str(wrong)))

    # MISC-CALC-010: a product of exactly two non-constant factors,
    # differentiated as f'(x)*g'(x) instead of the product rule.
    if isinstance(expr, sympy.Mul):
        factors = [f for f in expr.args if not f.is_number]
        if len(factors) == 2:
            f, g = factors
            wrong = sympy.simplify(sympy.diff(f, x) * sympy.diff(g, x))
            if wrong != correct:
                hypotheses.append(("MISC-CALC-010", str(wrong)))

    # MISC-CALC-014: a genuine composite (base^n with base depending on x
    # non-trivially, i.e. the chain rule actually applies here), power
    # rule applied to the outer function only.
    if isinstance(expr, sympy.Pow):
        base, exponent = expr.args
        if exponent.is_number and base.has(x) and sympy.diff(base, x) != 1:
            wrong = sympy.simplify(exponent * base ** (exponent - 1))
            if wrong != correct:
                hypotheses.append(("MISC-CALC-014", str(wrong)))

    return hypotheses


def _equation_to_zero_expr(equation: str, variable: str) -> Optional[sympy.Expr]:
    if "=" in equation:
        lhs_str, rhs_str = equation.split("=", 1)
        lhs = try_parse_expression(lhs_str)
        rhs = try_parse_expression(rhs_str)
        if lhs is None or rhs is None:
            return None
        return sympy.expand(lhs - rhs)
    return try_parse_expression(equation)


def _solve_hypotheses(zero_expr: sympy.Expr, variable: str) -> list[tuple[str, str]]:
    x = sympy.Symbol(variable)
    try:
        poly = sympy.Poly(zero_expr, x)
    except sympy.PolynomialError:
        return []
    if poly.degree() != 2:
        return []

    a, b, c = poly.all_coeffs()
    discriminant = sympy.simplify(b**2 - 4 * a * c)
    if discriminant.is_number and discriminant < 0:
        return []

    sqrt_disc = sympy.sqrt(discriminant)
    correct_roots = {sympy.nsimplify((-b + sqrt_disc) / (2 * a)), sympy.nsimplify((-b - sqrt_disc) / (2 * a))}
    # MISC-ALG-003: sign error on b (uses +b instead of -b in the quadratic formula).
    wrong_roots = [sympy.nsimplify((b + sqrt_disc) / (2 * a)), sympy.nsimplify((b - sqrt_disc) / (2 * a))]
    if set(wrong_roots) == correct_roots:
        return []  # b == 0: the sign error is indistinguishable from the correct answer here

    wrong_value = ", ".join(f"{variable} = {root}" for root in wrong_roots)
    return [("MISC-ALG-003", wrong_value)]


def generate_hypotheses(operation: CASOperation, expression: str, variable: str) -> list[tuple[str, str]]:
    """Every (misconception_id, wrong_value) this problem's shape could
    plausibly produce. Not every operation has known detectors yet
    (integrate/simplify/evaluate: none built) — an empty list is the
    honest, expected result for those, not a bug."""
    if operation == CASOperation.DIFFERENTIATE:
        expr = try_parse_expression(expression)
        if expr is None:
            return []
        return _differentiation_hypotheses(expr, variable)
    if operation == CASOperation.SOLVE:
        zero_expr = _equation_to_zero_expr(expression, variable)
        if zero_expr is None:
            return []
        return _solve_hypotheses(zero_expr, variable)
    return []


def _values_match(operation: CASOperation, hypothesis_value: str, student_value: str) -> bool:
    if operation == CASOperation.SOLVE:
        hypothesis_roots = _split_roots(hypothesis_value)
        student_roots = _split_roots(student_value)
        if len(hypothesis_roots) != len(student_roots):
            return False
        remaining = list(student_roots)
        for root in hypothesis_roots:
            match = next((r for r in remaining if _expressions_equivalent(root, r)), None)
            if match is None:
                return False
            remaining.remove(match)
        return True
    return _expressions_equivalent(_strip_assignment(hypothesis_value), _strip_assignment(student_value))


def detect_pattern_misconception(
    operation: CASOperation, expression: str, variable: str, student_value: str
) -> Optional[tuple[str, str]]:
    """Returns (misconception_id, evidence) for the first hypothesis that
    exactly matches the student's stated final value, or None if no
    catalogued pattern fits this problem and this wrong answer."""
    for misconception_id, wrong_value in generate_hypotheses(operation, expression, variable):
        if _values_match(operation, wrong_value, student_value):
            evidence = (
                f"student's answer '{student_value}' exactly matches the value '{wrong_value}' "
                f"produced by this specific error on this problem"
            )
            return misconception_id, evidence
    return None
