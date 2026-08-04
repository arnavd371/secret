"""
Tutor agent system prompt, built from the spec's own §7.7 "Tutor Agent
System Prompt Skeleton" — a single shared skeleton where the BOUND ACTION
line and its associated hard constraints vary by action_type, rather than
a separate mega-prompt per action with branching prose logic.

Using one skeleton (as the spec literally specifies) does not weaken
enforcement: the HARD CONSTRAINTS section is a declarative, structured
instruction block (not "if the user says X, do Y" conditionals), and the
real enforcement of the no-leak rule is the structural post-hoc check in
tutor_agent.py — the prompt is the first, weaker layer; the regex gate is
the second, real one that does not depend on the model having listened.

Retrieval, CAS verification, memory context, and misconception diagnosis
are later-phase subsystems (§5, §2.2 Math Solver+CAS, §4, §8) that don't
exist yet in Phase 1. Their slots in the skeleton are filled with an
explicit "not yet available" placeholder rather than omitted, so the
skeleton's shape already matches what later phases will populate.
"""

from __future__ import annotations

from app.models.contracts import Action, ActionType

_NOT_YET_AVAILABLE = "(not yet available — Phase 1 has no retrieval/CAS/memory/misconception subsystem)"

_SKELETON = """You are the teaching agent for an IB Diploma {subject} tutoring system, currently working with a \
student on {level} content. You must follow the bound pedagogical action exactly.

BOUND ACTION: {action_type} (move: {move}, hint_level: {hint_level})
STUDENT MASTERY CONTEXT: {memory_read_context}
ACTIVE MISCONCEPTIONS: {active_misconceptions}
RETRIEVED CURRICULUM CONTEXT (cite these; do not state syllabus facts not grounded here):
{retrieved_chunks}
CAS-VERIFIED RESULT (ground truth; your narrative must not contradict this): {cas_result}

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
    "claim in the RETRIEVED CURRICULUM CONTEXT and cite it.\n"
    "- Never state a mathematical result that disagrees with CAS-VERIFIED RESULT.\n"
    "- Match tone to STUDENT MASTERY CONTEXT: encouraging but not patronizing; specific, not generic praise.\n"
    "- Limit response to the cognitive-load rules: one new concept per turn, max 6 worked-example steps."
)

_CHALLENGE_CONSTRAINT = (
    "- BOUND ACTION is CHALLENGE: this student has demonstrated high mastery. Pose a harder extension "
    "question (a variation, edge case, or higher-mark IB-style question) rather than re-explaining basics. "
    "Do not simply hand them the original problem's solution.\n"
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


def build_system_prompt(action: Action, *, subject: str = "Mathematics: Analysis and Approaches", level: str = "SL") -> str:
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
        memory_read_context=_NOT_YET_AVAILABLE,
        active_misconceptions=_NOT_YET_AVAILABLE,
        retrieved_chunks=_NOT_YET_AVAILABLE,
        cas_result=_NOT_YET_AVAILABLE,
        hard_constraints=hard_constraints,
    )
