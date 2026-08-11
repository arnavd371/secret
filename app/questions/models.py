"""
Typed contracts for the Question Generation Engine (spec §9), trimmed to
the parametric/template generation mode (§9.2-§9.4, §9.9, §9.12-§9.13).

Declarative template data lives here as validated Pydantic models, mirroring
spec §9.2's Item Template Schema. The distractor-generation *logic* for each
template is plain Python (see app/questions/generator.py's registry) rather
than embedded in these models, since a generation function isn't itself a
JSON-serializable contract the way the template's data is.

Not modeled here (later-phase non-goals, same spirit as Phase 2's TODOs):
  - LLM-authored variants + verifier gating (§9.6)
  - Online IRT recalibration from response history (§9.7) — no response
    history exists yet; irt_prior below is a declared, fixed default, not
    a value learned from data.
  - Mixed-topic/paper-style composition as constrained optimization (§9.10)
  - Adaptive generation targeting weak skills/misconceptions (§9.11) —
    needs the Phase 5 mastery model.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.cas.models import CASOperation


class ParamSpec(BaseModel):
    domain_min: int
    domain_max: int
    exclude: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _domain_is_valid(self) -> "ParamSpec":
        if self.domain_min > self.domain_max:
            raise ValueError("domain_min must be <= domain_max")
        return self


class ItemTemplate(BaseModel):
    template_id: str
    version: int = 1
    syllabus_ref: list[str] = Field(default_factory=list)
    scope: Literal["SL", "HL"] = "SL"
    skill_tags: list[str] = Field(default_factory=list)
    misconception_hooks: list[str] = Field(default_factory=list)
    calculator_mode: Literal["calculator", "non-calculator"] = "non-calculator"
    # (b_min, b_max): declared IRT difficulty band. Spec §9.7's IRT
    # parameters (a, b, c) are normally offline-calibrated from response
    # data; with none available yet, these are hand-set curriculum priors.
    difficulty_band: tuple[float, float] = (-1.0, 1.0)
    command_term: str = "Find"

    stem_template: str
    parameters: dict[str, ParamSpec]
    # Python boolean expressions evaluated against the sampled parameters
    # (and, where relevant, the CAS-computed answer) to reject degenerate
    # samples — e.g. "a != c", "abs(answer) <= 12".
    constraints: list[str] = Field(default_factory=list)

    operation: CASOperation
    expression_template: str
    variable: str = "x"

    max_resample_attempts: int = 200


class Distractor(BaseModel):
    value: str
    generator_id: str
    misconception: Optional[str] = None


class DifficultyEstimate(BaseModel):
    b_param: float
    source: Literal["template_prior"] = "template_prior"


class CorrectAnswer(BaseModel):
    value: str
    form: Literal["exact"] = "exact"
    cas_verified: bool


class QualityGateResult(BaseModel):
    gate: str
    passed: bool
    detail: str


class QualityGateReport(BaseModel):
    item_id: str
    results: list[QualityGateResult]

    @property
    def overall_status(self) -> Literal["PASSED", "FAILED"]:
        return "PASSED" if all(r.passed for r in self.results) else "FAILED"


class MarkSchemeNode(BaseModel):
    id: str
    type: Literal["M", "A"]
    text: str
    marks: int = 1


class MarkScheme(BaseModel):
    item_id: str
    total_marks: int
    nodes: list[MarkSchemeNode]


class GeneratedItem(BaseModel):
    item_id: str
    template_id: str
    template_version: int
    sampled_parameters: dict[str, int]
    rendered_stem: str
    calculator_mode: str
    difficulty_estimate: DifficultyEstimate
    correct_answer: CorrectAnswer
    distractors: list[Distractor] = Field(default_factory=list)
    generation_mode: Literal["parametric"] = "parametric"
    quality_gate_report: Optional[QualityGateReport] = None
    mark_scheme: Optional[MarkScheme] = None

    @property
    def quality_gate_status(self) -> Literal["PASSED", "FAILED", "NOT_RUN"]:
        if self.quality_gate_report is None:
            return "NOT_RUN"
        return self.quality_gate_report.overall_status
