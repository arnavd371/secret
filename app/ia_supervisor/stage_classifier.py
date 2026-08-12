"""
Stage classifier (spec §11): a real, small keyword heuristic mapping what
a student's IA/EE message is actually about to one of the real stages an
IA/EE project goes through — same "deliberately small heuristic, not
real NLU" posture as app.cas.extraction and app.orchestrator.signals.

Checked most-specific-first: COMPLETE requires genuine submission
language (not just "I finished my draft," which means the draft is
done, not the project), so a student asking for feedback on a finished
draft still classifies as REVISION rather than prematurely closing
coaching.
"""

from __future__ import annotations

from typing import Optional

from app.ia_supervisor.models import IAStage

_COMPLETE_MARKERS = (
    "i submitted",
    "i've submitted",
    "ive submitted",
    "already submitted",
    "turned it in",
    "handed it in",
    "my ia is complete",
    "my ee is complete",
    "finished submitting",
)

_REVISION_MARKERS = (
    "revise",
    "revision",
    "feedback on my draft",
    "review my draft",
    "improve my draft",
    "polish my draft",
)

_DRAFTING_MARKERS = (
    "draft",
    "introduction",
    "conclusion",
    "write up",
    "writing up",
    "abstract",
)

_ANALYSIS_MARKERS = (
    "analysis",
    "results",
    "findings",
    "interpret",
    "data analysis",
)

_METHODOLOGY_MARKERS = (
    "methodology",
    "method",
    "data collection",
    "experiment design",
    "procedure",
    "sample size",
)

_RESEARCH_QUESTION_MARKERS = (
    "research question",
    " rq ",
    "rq?",
)

_TOPIC_SELECTION_MARKERS = (
    "topic",
    "subject area",
    "what should i write about",
    "choosing a topic",
    "pick a topic",
)

# Order matters: most specific/terminal first, so overlapping vocabulary
# (e.g. "draft" appearing inside a revision request) resolves to the more
# specific stage rather than the more generic one.
_STAGE_MARKERS: list[tuple[IAStage, tuple[str, ...]]] = [
    (IAStage.COMPLETE, _COMPLETE_MARKERS),
    (IAStage.REVISION, _REVISION_MARKERS),
    (IAStage.DRAFTING, _DRAFTING_MARKERS),
    (IAStage.ANALYSIS, _ANALYSIS_MARKERS),
    (IAStage.METHODOLOGY, _METHODOLOGY_MARKERS),
    (IAStage.RESEARCH_QUESTION, _RESEARCH_QUESTION_MARKERS),
    (IAStage.TOPIC_SELECTION, _TOPIC_SELECTION_MARKERS),
]


def classify_stage(raw_input: str) -> Optional[IAStage]:
    """None means nothing recognizable was mentioned this turn — the
    caller should keep the project's existing stage rather than guessing
    one, same "a missed extraction is not a bug" convention as
    app.cas.extraction.extract_math_task."""
    lowered = f" {raw_input.lower()} "
    for stage, markers in _STAGE_MARKERS:
        if any(marker in lowered for marker in markers):
            return stage
    return None
