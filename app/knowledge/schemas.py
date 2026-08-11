"""
Typed contracts for the Knowledge Base & Retrieval subsystem (spec §5).

Content-authoring models (`LearningObjective`, `FormulaBookletEntry`,
`WorkedExample`) mirror the spec's §5.2 canonical JSON schemas, trimmed to
the fields Phase 2 actually authors content against. `KnowledgeChunk` is
the internal indexed/retrievable unit each of those normalizes into;
`RetrievedChunk` is what the Retriever hands back, matching spec §2.2's
Retriever Agent output shape (doc_id/type/text/score/authority_tier).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    LEARNING_OBJECTIVE = "learning_objective"
    FORMULA_BOOKLET_ENTRY = "formula_booklet_entry"
    WORKED_EXAMPLE = "worked_example"


class LearningObjective(BaseModel):
    id: str
    subtopic_id: str
    text: str
    level: str
    source: str


class FormulaBookletEntry(BaseModel):
    id: str
    section: str
    level: str
    latex: str
    name: str
    applies_to_subtopics: list[str] = Field(default_factory=list)


class WorkedExampleStep(BaseModel):
    step: int
    latex: str
    explanation: str


class WorkedExample(BaseModel):
    id: str
    subtopic_id: str
    stem_latex: str
    steps: list[WorkedExampleStep]


class KnowledgeChunk(BaseModel):
    """The indexed, retrievable unit. `keywords` supplements the doc's own
    text with synonyms/aliases a student might actually type (e.g. "u v
    prime" for the product rule) — a standard lexical-search technique to
    improve recall without a real embedding model."""

    chunk_id: str
    doc_type: DocType
    subtopic_id: str
    citation: str
    text: str
    keywords: list[str] = Field(default_factory=list)
    # Per spec §5.7's syllabus_authority_score: 1.0 official guide/formula
    # booklet content, 0.7 internally-authored worked examples.
    authority_tier: float = 1.0

    @property
    def searchable_text(self) -> str:
        return " ".join([self.text, *self.keywords])


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_type: DocType
    subtopic_id: str
    citation: str
    text: str
    score: float
    authority_tier: float
