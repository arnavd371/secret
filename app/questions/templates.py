"""
A small, real seed of parametric item templates (spec §9.2), covering the
same handful of topics as the Phase 2 knowledge base seed for cohesion —
not the full syllabus (that's a large, curriculum-team-heavy effort per
spec §16.7, same caveat as the knowledge base).

`difficulty_band` values are hand-set curriculum priors, not IRT-
calibrated from response data (spec §9.7's online recalibration needs a
response history this system doesn't have yet).
"""

from __future__ import annotations

from app.cas.models import CASOperation
from app.questions.models import ItemTemplate, ParamSpec

POWER_RULE_TEMPLATE = ItemTemplate(
    template_id="AA.SL.CALC.DIFF.POWER.T001",
    syllabus_ref=["AA 5.3"],
    scope="SL",
    skill_tags=["power_rule", "differentiation"],
    misconception_hooks=["MISC-CALC-GEN-DROPPED-TERM"],
    calculator_mode="non-calculator",
    difficulty_band=(-1.0, -0.2),
    command_term="Find",
    stem_template=r"Find \frac{{dy}}{{dx}} where y = {a}x^{n} + {b}x.",
    parameters={
        "a": ParamSpec(domain_min=2, domain_max=9),
        "n": ParamSpec(domain_min=2, domain_max=5),
        "b": ParamSpec(domain_min=1, domain_max=9),
    },
    constraints=[],
    operation=CASOperation.DIFFERENTIATE,
    expression_template="{a}*x**{n} + {b}*x",
    variable="x",
)

PRODUCT_RULE_TEMPLATE = ItemTemplate(
    template_id="AA.SL.CALC.DIFF.PRODUCT.T002",
    syllabus_ref=["AA 5.6"],
    scope="SL",
    skill_tags=["product_rule", "differentiation"],
    misconception_hooks=["MISC-CALC-010"],
    calculator_mode="non-calculator",
    difficulty_band=(0.0, 0.6),
    command_term="Find",
    stem_template=r"Find \frac{{dy}}{{dx}} where y = x^{n} \sin(x).",
    parameters={"n": ParamSpec(domain_min=2, domain_max=4)},
    constraints=[],
    operation=CASOperation.DIFFERENTIATE,
    expression_template="x**{n} * sin(x)",
    variable="x",
)

CHAIN_RULE_TEMPLATE = ItemTemplate(
    template_id="AA.SL.CALC.DIFF.CHAIN.T003",
    syllabus_ref=["AA 5.6"],
    scope="SL",
    skill_tags=["chain_rule", "differentiation"],
    misconception_hooks=["MISC-CALC-014"],
    calculator_mode="non-calculator",
    difficulty_band=(0.2, 0.8),
    command_term="Find",
    stem_template=r"Find \frac{{dy}}{{dx}} where y = ({a}x {b_signed})^{n}.",
    parameters={
        "a": ParamSpec(domain_min=2, domain_max=5),
        "b": ParamSpec(domain_min=-3, domain_max=3, exclude=[0]),
        "n": ParamSpec(domain_min=2, domain_max=5),
    },
    constraints=[],
    operation=CASOperation.DIFFERENTIATE,
    expression_template="({a}*x + {b})**{n}",
    variable="x",
)

QUADRATIC_FORMULA_TEMPLATE = ItemTemplate(
    template_id="AA.SL.ALG.QUAD.T004",
    syllabus_ref=["AA 2.5"],
    scope="SL",
    skill_tags=["quadratic_formula", "algebra"],
    misconception_hooks=["MISC-ALG-003"],
    calculator_mode="non-calculator",
    difficulty_band=(-0.5, 0.3),
    command_term="Solve",
    stem_template="Solve {a}x^2 {b_signed}x {c_signed} = 0 for x, giving your answer(s) in exact form.",
    parameters={
        "a": ParamSpec(domain_min=1, domain_max=3),
        "b": ParamSpec(domain_min=-6, domain_max=6, exclude=[0]),
        "c": ParamSpec(domain_min=-6, domain_max=6, exclude=[0]),
    },
    # Real-root constraint: this template is a basic SL solving item, not a
    # complex-numbers item, so reject discriminants that would produce
    # complex roots rather than let the number-friendliness gate reject
    # them indirectly.
    constraints=["a != 0", "b**2 - 4*a*c >= 0"],
    operation=CASOperation.SOLVE,
    expression_template="{a}*x**2 + {b}*x + {c} = 0",
    variable="x",
)

TEMPLATE_BANK: dict[str, ItemTemplate] = {
    t.template_id: t
    for t in [POWER_RULE_TEMPLATE, PRODUCT_RULE_TEMPLATE, CHAIN_RULE_TEMPLATE, QUADRATIC_FORMULA_TEMPLATE]
}
