"""
Tutor agent system prompt, built from the spec's own §7.7 "Tutor Agent
System Prompt Skeleton" — a single shared skeleton where the BOUND ACTION
line and its associated hard constraints vary by action_type, rather than
a separate mega-prompt per action with branching prose logic.

Using one skeleton (as the spec literally specifies) does not weaken
enforcement: the HARD CONSTRAINTS section is a declarative, structured
instruction block (not "if the user says X, do Y" conditionals), and the
real enforcement of the no-leak / no-unverified-answer rules is the
structural post-hoc checks in tutor_agent.py — the prompt is the first,
weaker layer; the regex/CAS gates are the second, real ones that do not
depend on the model having listened.

RETRIEVED CURRICULUM CONTEXT and CAS-VERIFIED RESULT are real when Phase 2's
Retriever/CAS agents found something for this turn; EXTENSION ITEM is real
when Phase 3's Question Generation Engine produced a CAS-verified item —
for a CHALLENGE action (a harder problem for a high-mastery student), or
for a QUESTION/retrieval_practice action bound to a real due-for-review
subtopic (Phase 9's Adaptive Learning Engine, spec §12); STUDENT MASTERY
CONTEXT / ACTIVE MISCONCEPTIONS are real when Phase 5's Memory Agent has a
persisted record for the turn's subtopic. When any of them didn't apply
this turn, the slot says so explicitly rather than being silently blank,
so the model is told not to assert unsupported claims or invent its own
extension question. The item's answer is protected the same way
regardless of which action bound it: tutor_agent.py's structural leak
check covers both CHALLENGE and QUESTION drafts that state the bound
item's own verified answer.
"""

from __future__ import annotations

from typing import Optional

from app.cas.models import CASResult, CASStatus
from app.knowledge.retriever import RETRIEVAL_SCORE_THRESHOLD, is_grounded
from app.knowledge.schemas import RetrievedChunk
from app.memory.models import MemoryReadContext
from app.models.contracts import Action, ActionType
from app.questions.models import GeneratedItem

_NO_MEMORY_CONTEXT = "(no mastery history for this subtopic yet)"
_NO_MISCONCEPTIONS = "(none recorded)"
_NO_CAS_TASK = "(no CAS verification was run for this turn — no checkable math task was identified)"
_NO_GROUNDING = (
    "(no knowledge-base grounding cleared the confidence threshold for this query — do not state "
    "syllabus-specific facts as certain; hedge, or say you can't confirm the IB-specific convention)"
)
_NO_EXTENSION_ITEM = "(no extension item was generated for this turn)"

_SKELETON = """You are the teaching agent for an IB Diploma {subject} tutoring system, currently working with a \
student on {level} content. You must follow the bound pedagogical action exactly.

BOUND ACTION: {action_type} (move: {move}, hint_level: {hint_level})
STUDENT MASTERY CONTEXT: {memory_read_context}
ACTIVE MISCONCEPTIONS: {active_misconceptions}
RETRIEVED CURRICULUM CONTEXT (cite these; do not state syllabus facts not grounded here):
{retrieved_chunks}
CAS-VERIFIED RESULT (ground truth; your narrative must not contradict this): {cas_result}
EXTENSION ITEM (present this verbatim or lightly reworded as the new problem; its answer is for your \
reference only — never state it): {extension_item}

HARD CONSTRAINTS:
{hard_constraints}
- Output must be valid Markdown with LaTeX delimited by \\( \\) inline / \\[ \\] display.

OUTPUT SCHEMA:
{{ "text": "<markdown/latex response>", "citations": ["doc_id", ...], "ui_hints": {{"show_hint_button": bool}} }}
"""

_QUESTION_OR_HINT_CONSTRAINT = (
    "- If BOUND ACTION is QUESTION or HINT, you MUST NOT reveal the final numeric/symbolic answer, "
    "even if asked directly. Redirect to the next rung of the hint ladder instead.\n"
    "- Never state a mathematical result that disagrees with CAS-VERIFIED RESULT.\n"
    "- Match tone to STUDENT MASTERY CONTEXT: encouraging but not patronizing; specific, not generic praise.\n"
    "- Limit response to the cognitive-load rules: one new concept per turn, max 6 worked-example steps."
)

_EXPLAIN_CONSTRAINT = (
    "- If BOUND ACTION is EXPLAIN, you MAY give a full explanation but must ground every syllabus-specific "
    "claim in the RETRIEVED CURRICULUM CONTEXT and cite it. If RETRIEVED CURRICULUM CONTEXT says no grounding "
    "was found, do not state syllabus-specific conventions as fact.\n"
    "- Never state a mathematical result that disagrees with CAS-VERIFIED RESULT. If CAS-VERIFIED RESULT says "
    "no verification was run or it could not be verified, do not assert a specific final numeric/symbolic answer.\n"
    "- Match tone to STUDENT MASTERY CONTEXT: encouraging but not patronizing; specific, not generic praise.\n"
    "- Limit response to the cognitive-load rules: one new concept per turn, max 6 worked-example steps."
)

_CHALLENGE_CONSTRAINT = (
    "- BOUND ACTION is CHALLENGE: this student has demonstrated high mastery. Present the EXTENSION ITEM "
    "above as the new problem for them to attempt — do not invent your own, and do not simply hand them the "
    "original problem's solution.\n"
    "- You MUST NOT reveal the EXTENSION ITEM's answer, even if asked directly — the student is meant to "
    "attempt it, this is not an EXPLAIN action.\n"
    "- Never state a mathematical result that disagrees with CAS-VERIFIED RESULT."
)

_SUPPORTIVE_SCAFFOLD_CONSTRAINT = (
    "- BOUND ACTION is SUPPORTIVE_SCAFFOLD: the student is showing signs of frustration. Acknowledge that "
    "briefly and warmly (tone: {tone}), then work through a faded worked example broken into a smaller, "
    "more approachable step. Do not lecture. Do not give the final answer unprompted.\n"
    "- Never state a mathematical result that disagrees with CAS-VERIFIED RESULT."
)

_CONSTRAINTS_BY_ACTION = {
    ActionType.QUESTION: _QUESTION_OR_HINT_CONSTRAINT,
    ActionType.HINT: _QUESTION_OR_HINT_CONSTRAINT,
    ActionType.EXPLAIN: _EXPLAIN_CONSTRAINT,
    ActionType.CHALLENGE: _CHALLENGE_CONSTRAINT,
    ActionType.SUPPORTIVE_SCAFFOLD: _SUPPORTIVE_SCAFFOLD_CONSTRAINT,
}


def _format_retrieved_chunks(chunks: Optional[list[RetrievedChunk]]) -> str:
    if not chunks or not is_grounded(chunks):
        return _NO_GROUNDING
    return "\n".join(f"- [{chunk.citation}] {chunk.text}" for chunk in chunks if chunk.score >= RETRIEVAL_SCORE_THRESHOLD)


def _format_cas_result(cas_result: Optional[CASResult]) -> str:
    if cas_result is None:
        return _NO_CAS_TASK
    if cas_result.status != CASStatus.OK:
        return "(CAS could not verify a result for this turn — do not state a specific final numeric/symbolic answer)"
    return f"operation={cas_result.operation.value}, verified result = {cas_result.result_exact}"


def _format_extension_item(item: Optional[GeneratedItem]) -> str:
    if item is None:
        return _NO_EXTENSION_ITEM
    return f"Stem: {item.rendered_stem} | (reference answer, do not state: {item.correct_answer.value})"


def _format_memory_context(memory_context: Optional[MemoryReadContext]) -> str:
    if memory_context is None or not memory_context.rendered_text:
        return _NO_MEMORY_CONTEXT
    return memory_context.rendered_text


def _format_active_misconceptions(memory_context: Optional[MemoryReadContext]) -> str:
    if memory_context is None or not memory_context.active_misconception_ids:
        return _NO_MISCONCEPTIONS
    return ", ".join(memory_context.active_misconception_ids)


def build_system_prompt(
    action: Action,
    *,
    subject: str = "Mathematics: Analysis and Approaches",
    level: str = "SL",
    cas_result: Optional[CASResult] = None,
    retrieved_chunks: Optional[list[RetrievedChunk]] = None,
    challenge_item: Optional[GeneratedItem] = None,
    memory_context: Optional[MemoryReadContext] = None,
) -> str:
    if action.action_type == ActionType.REFUSE:
        raise ValueError(
            "REFUSE actions must never reach the Tutor agent — the "
            "orchestrator is responsible for hard-gating them earlier."
        )

    hard_constraints = _CONSTRAINTS_BY_ACTION[action.action_type]
    if action.action_type == ActionType.SUPPORTIVE_SCAFFOLD:
        hard_constraints = hard_constraints.format(tone=action.tone or "reassuring")

    return _SKELETON.format(
        subject=subject,
        level=level,
        action_type=action.action_type.value.upper(),
        move=action.move or "none",
        hint_level=action.level if action.level is not None else "n/a",
        memory_read_context=_format_memory_context(memory_context),
        active_misconceptions=_format_active_misconceptions(memory_context),
        retrieved_chunks=_format_retrieved_chunks(retrieved_chunks),
        cas_result=_format_cas_result(cas_result),
        extension_item=_format_extension_item(challenge_item),
        hard_constraints=hard_constraints,
    )
