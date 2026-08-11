"""
Distractor generators (spec §9.2's `distractor_generators`): each produces
a plausible wrong answer by deliberately applying one real misconception
from the catalog (spec §8.2) to the sampled parameters, rather than
perturbing the correct answer randomly. Where possible these reuse the
exact misconception IDs from the spec's own §8.2 catalog, so a served
item's distractor is traceable to a named, catalogued error pattern.
"""

from __future__ import annotations

from typing import Callable, Optional

import sympy

from app.questions.models import Distractor

DistractorFn = Callable[[dict[str, int], str], Optional[Distractor]]


def _power_rule_dropped_linear_term(params: dict[str, int], variable: str) -> Optional[Distractor]:
    """y = a*x^n + b*x -> correct dy/dx = a*n*x^(n-1) + b. A common slip
    drops the derivative of the linear term entirely."""
    a, n = params["a"], params["n"]
    x = sympy.Symbol(variable)
    wrong = sympy.simplify(a * n * x ** (n - 1))
    return Distractor(value=str(wrong), generator_id="DG-POWER-DROPPED-TERM", misconception="MISC-CALC-GEN-DROPPED-TERM")


def _product_rule_wrong_method(params: dict[str, int], variable: str) -> Optional[Distractor]:
    """spec §8.2 MISC-CALC-010: 'Differentiates a product using f'(x)g'(x)
    instead of the product rule.'"""
    n = params["n"]
    x = sympy.Symbol(variable)
    f_prime = sympy.diff(x**n, x)
    g_prime = sympy.diff(sympy.sin(x), x)
    wrong = sympy.simplify(f_prime * g_prime)
    return Distractor(value=str(wrong), generator_id="DG-PRODUCT-WRONG-METHOD", misconception="MISC-CALC-010")


def _chain_rule_outer_only(params: dict[str, int], variable: str) -> Optional[Distractor]:
    """spec §8.2 MISC-CALC-014: 'Applies power rule to outer function in a
    composite but forgets to multiply by derivative of inner function.'
    This is the spec's own worked example (WE-calc-chainrule-001)."""
    a, b, n = params["a"], params["b"], params["n"]
    x = sympy.Symbol(variable)
    inner = a * x + b
    wrong = sympy.simplify(n * inner ** (n - 1))  # missing the inner-derivative factor `a`
    return Distractor(value=str(wrong), generator_id="DG-CHAIN-OUTER-ONLY", misconception="MISC-CALC-014")


def _quadratic_sign_error(params: dict[str, int], variable: str) -> Optional[Distractor]:
    """spec §8.2 MISC-ALG-003: 'Sign error distributing a negative across
    a bracket' — here, using +b instead of -b in the quadratic formula."""
    a, b, c = params["a"], params["b"], params["c"]
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    wrong = sympy.nsimplify((b + sympy.sqrt(discriminant)) / (2 * a))
    return Distractor(value=f"{variable} = {wrong}", generator_id="DG-QUAD-SIGN-ERROR", misconception="MISC-ALG-003")


DISTRACTOR_REGISTRY: dict[str, list[DistractorFn]] = {
    "AA.SL.CALC.DIFF.POWER.T001": [_power_rule_dropped_linear_term],
    "AA.SL.CALC.DIFF.PRODUCT.T002": [_product_rule_wrong_method],
    "AA.SL.CALC.DIFF.CHAIN.T003": [_chain_rule_outer_only],
    "AA.SL.ALG.QUAD.T004": [_quadratic_sign_error],
}


def generate_distractors(template_id: str, params: dict[str, int], variable: str, correct_value: str) -> list[Distractor]:
    generators = DISTRACTOR_REGISTRY.get(template_id, [])
    distractors: list[Distractor] = []
    for generator_fn in generators:
        try:
            distractor = generator_fn(params, variable)
        except Exception:  # noqa: BLE001 - a distractor failing to compute must never break item generation
            continue
        if distractor is not None and distractor.value != correct_value:
            distractors.append(distractor)
    return distractors
