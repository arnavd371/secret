"""
Regression gate: compares a freshly-run EvalReport against a stored
baseline and fails if any category's pass rate has genuinely dropped.
Deliberately per-category, not just overall - a change that fixes ten
CAS cases while silently breaking two grading cases must not net out to
"looks fine overall".

`tolerance` exists because pass rates are computed over a small, fixed
scenario set: a single flipped case moves a category's rate by a
noticeable fraction. The default (0.0) treats every regression as real
- a caller intentionally accepting noise can raise it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.eval.models import EvalCategory, EvalReport


class CategoryRegression(BaseModel):
    category: str
    baseline_pass_rate: float
    current_pass_rate: float

    @property
    def delta(self) -> float:
        return self.current_pass_rate - self.baseline_pass_rate


class RegressionGateResult(BaseModel):
    passed: bool
    regressions: list[CategoryRegression] = Field(default_factory=list)
    baseline_overall_pass_rate: float
    current_overall_pass_rate: float

    @property
    def summary(self) -> str:
        if self.passed:
            return f"gate passed: overall pass rate {self.current_overall_pass_rate:.1%} (baseline {self.baseline_overall_pass_rate:.1%})"
        names = ", ".join(f"{r.category} ({r.baseline_pass_rate:.1%} -> {r.current_pass_rate:.1%})" for r in self.regressions)
        return f"gate FAILED: regressed categories: {names}"


def check_regression(current: EvalReport, baseline: EvalReport, tolerance: float = 0.0) -> RegressionGateResult:
    regressions: list[CategoryRegression] = []
    for category in EvalCategory:
        baseline_rate = baseline.pass_rate_for(category)
        current_rate = current.pass_rate_for(category)
        if current_rate < baseline_rate - tolerance:
            regressions.append(
                CategoryRegression(
                    category=category.value,
                    baseline_pass_rate=baseline_rate,
                    current_pass_rate=current_rate,
                )
            )

    return RegressionGateResult(
        passed=len(regressions) == 0,
        regressions=regressions,
        baseline_overall_pass_rate=baseline.pass_rate,
        current_overall_pass_rate=current.pass_rate,
    )
