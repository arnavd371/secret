"""
Data model for offline eval reporting. Deliberately separate from
pytest's pass/fail: an eval report is a numeric artifact (per-category
pass rate) that gets persisted as a baseline and diffed release over
release, not just asserted true once in CI.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvalCategory(str, Enum):
    CAS = "cas"
    DECISION_POLICY = "decision_policy"
    GRADING = "grading"


class EvalCaseResult(BaseModel):
    name: str
    category: EvalCategory
    passed: bool
    detail: str = ""


class EvalReport(BaseModel):
    results: list[EvalCaseResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 1.0
        return self.total_passed / self.total

    def category_results(self, category: EvalCategory) -> list[EvalCaseResult]:
        return [r for r in self.results if r.category == category]

    def pass_rate_for(self, category: EvalCategory) -> float:
        cases = self.category_results(category)
        if not cases:
            return 1.0
        return sum(1 for r in cases if r.passed) / len(cases)

    def failed_cases(self) -> list[EvalCaseResult]:
        return [r for r in self.results if not r.passed]

    def category_pass_rates(self) -> dict[str, float]:
        return {category.value: self.pass_rate_for(category) for category in EvalCategory}
