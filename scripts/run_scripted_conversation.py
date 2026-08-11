"""
Runnable demo of the reasoning core through Phase 9: a scripted, multi-turn
conversation through `handle_turn`, with a mocked model provider so it runs
with no API key and no network access. The Router/Intent classification,
Tutor generation, and math_ocr transcription calls are mocked; the decision
policy, hint ladder, CAS verification (real SymPy), retrieval (real lexical
search over the seed knowledge base), question generation (real,
quality-gated, CAS-verified items), grading (real step segmentation +
alignment + mark awarding), memory (real BKT/IRT mastery updates, decay,
and budgeted context assembly), the Verifier/Critic + grounding check
(Phase 6), the multimodal ingestion pipeline (Phase 7: real intake
validation, real PIL preprocessing, real LaTeX normalization/expression-
parsing/confidence scoring), the Misconception Diagnostician (Phase 8:
real SymPy pattern detection against the actual problem, written straight
into the same memory registry Phase 5 already reads from), and the
Adaptive Learning Engine (Phase 9: real FSRS-lite spaced-repetition
scheduling, a real due-review queue driving which subtopic gets a
generated retrieval-practice item) all run for real on every turn — this
script's mocked provider auto-passes the critic's checklist call so the
narrative isn't interrupted by it, but the "critique" line printed per
turn shows it genuinely ran. Explicit block/revise/regenerate scenarios
are covered in tests/test_integration_critic.py and
tests/test_tutor_agent.py instead of here, to keep this script's queue
bookkeeping manageable.

The chain-rule turns in this script deliberately do NOT pass an explicit
mastery_estimate override: three correct check_work gradings persist real
BKT mastery growth (starting from the p_init default of 0.10), and the
later CHALLENGE turn is driven entirely by that persisted state — proving
the loop actually closes, not just that the override plumbing works.

Run with:
    python scripts/run_scripted_conversation.py
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.adaptive.scheduler import record_review
from app.adaptive.store import InMemoryReviewStateStore
from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.memory.store import InMemoryMemoryStore
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore

CHAIN_RULE_TOPIC = "calculus.differentiation.chain_rule"


def _intent_json(
    intent: str, assessment_mode_guess: str = "practice", topic_hint: str = "calculus.differentiation.product_rule"
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "confidence": 0.9,
            "subject": "math_aa",
            "topic_hint": topic_hint,
            "assessment_mode_guess": assessment_mode_guess,
            "requires_multimodal_parse": False,
            "language": "en",
        }
    )


class ScriptedProvider(ProviderClient):
    """Phase 6's Verifier/Critic makes an additional, independent model
    call per turn on the same shared Provider.ANTHROPIC queue. This demo
    isn't scripting critic behavior, so a critic-shaped system prompt is
    auto-passed without consuming a slot in the scripted queue."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, *, spec, system, user, images=None) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.ANTHROPIC)
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


_CORRECT_CHAIN_RULE_WORK = "u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2"

# A photographed submission's math_ocr call returns this transcription
# (mocked, same as every other model call in this script) — real,
# correct chain-rule working for a fresh problem, run through the real
# ingestion pipeline (intake, PIL preprocessing, normalization,
# expression-parseability, composite confidence scoring).
_PHOTOGRAPHED_WORK_TRANSCRIPTION = "u = 3*x + 2\ntherefore dy/dx = 2*(3*x + 2)*3"

# Phase 8: a wrong submission that exactly matches a catalogued
# misconception (MISC-CALC-010: differentiating a product as f'(x)g'(x)
# instead of applying the product rule) for this specific problem's real
# CAS-verified derivative — not a template-time distractor, a real
# pattern match computed backwards from the actual problem.
_PRODUCT_RULE_WRONG_METHOD_WORK = "u = x**2, v = sin(x)\ntherefore dy/dx = 2*x*cos(x)"


def _fake_photo_bytes() -> bytes:
    """A synthetic PNG standing in for a real phone photo of handwritten
    work. Its pixel content doesn't matter here — the OCR call is mocked
    — but it's a real, valid, intake-acceptable image so the real
    preprocessing stage genuinely runs on it."""
    image = Image.new("RGB", (900, 700), color=(235, 235, 230))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

SCRIPT = [
    _intent_json("solve_request"),
    "What do you notice about the two factors being multiplied here?",
    _intent_json("solve_request"),
    "Which rule applies when you're differentiating a product of two functions?",
    _intent_json("solve_request"),
    "Try labeling one factor u and the other v, then recall the product rule formula.",
    _intent_json("solve_request"),
    "Write out u' and v' separately, then combine them with the product rule.",
    _intent_json("concept_explain"),
    "Using the product rule, the derivative is 2*x*cos(x) - x**2*sin(x).",
    # Three correct check_work gradings on the chain rule, back to back.
    # Grading bypasses the Tutor LLM entirely, so each of these three
    # turns consumes only the intent-classification response, no paired
    # "tutor draft".
    _intent_json("check_work", topic_hint=CHAIN_RULE_TOPIC),
    _intent_json("check_work", topic_hint=CHAIN_RULE_TOPIC),
    _intent_json("check_work", topic_hint=CHAIN_RULE_TOPIC),
    _intent_json("solve_request", topic_hint=CHAIN_RULE_TOPIC),
    "What's the inner function here, and what's the outer function?",
    # No mastery_estimate override on this turn — CHALLENGE must be
    # driven purely by the real mastery persisted from the three
    # gradings above.
    _intent_json("solve_request", topic_hint=CHAIN_RULE_TOPIC),
    "Great work, that's correct! Here's a tougher one for you to try on your own.",
    _intent_json("concept_explain", topic_hint=CHAIN_RULE_TOPIC),
    "The chain rule lets you differentiate composite functions by working from the outside in.",
    # Phase 7: a photographed submission. Grading bypasses the Tutor LLM
    # entirely (same as the three typed check_work turns above), so this
    # turn consumes only the intent-classification response plus the
    # math_ocr transcription response — no paired "tutor draft".
    _intent_json("check_work"),
    _PHOTOGRAPHED_WORK_TRANSCRIPTION,
    # Phase 8: another check_work turn, this one wrong. Grading and
    # diagnosis both bypass the Tutor LLM entirely (the pattern detector
    # is real SymPy, no model call at all), so this turn consumes only
    # the intent-classification response.
    _intent_json("check_work"),
    # Phase 9: an exam_prep turn with a real, seeded-overdue review
    # waiting in the Adaptive Learning Engine's queue. The decision
    # policy's has_due_review branch binds a real generated item to the
    # QUESTION action, same leak protection as a CHALLENGE item.
    _intent_json("exam_prep"),
    "This one's due for review: solve the quadratic. Try it from memory before checking your work.",
    # REFUSE short-circuits before the Tutor agent is ever called, so this
    # last entry is scripted but must be left unconsumed — it stays last
    # in the queue on purpose (see the assertion in the integration test).
    _intent_json("solve_request", assessment_mode_guess="live_exam_simulation"),
    "THIS SHOULD NEVER BE SHOWN — the exam-mode hard gate must refuse before this is reached.",
]

# (raw_input, problem_id, student_work, student_work_image)
TURNS: list[tuple[str, str, Optional[str], Optional[bytes]]] = [
    ("I'm stuck on differentiating x^2 * sin(x)", "problem-1", None, None),
    ("still not sure what to do", "problem-1", None, None),
    ("I tried but I'm still lost", "problem-1", None, None),
    ("that hint didn't help either", "problem-1", None, None),
    ("differentiate x**2 * cos(x)", "problem-2", None, None),
    ("can you check my work? differentiate (2*x+1)**5", "problem-3", _CORRECT_CHAIN_RULE_WORK, None),
    ("can you check my work? differentiate (2*x+1)**5", "problem-3", _CORRECT_CHAIN_RULE_WORK, None),
    ("can you check my work? differentiate (2*x+1)**5", "problem-3", _CORRECT_CHAIN_RULE_WORK, None),
    ("I'm working on differentiating (3x-2)^4 now", "problem-4", None, None),
    ("I solved it correctly!", "problem-4", None, None),
    ("can you remind me how the chain rule works?", "problem-4", None, None),
    (
        "here's a photo of my work, can you check it? differentiate (3*x + 2)**2",
        "problem-5",
        None,
        _fake_photo_bytes(),
    ),
    (
        "can you check my work? differentiate x**2 * sin(x)",
        "problem-6",
        _PRODUCT_RULE_WRONG_METHOD_WORK,
        None,
    ),
    ("can you help me review for my exam?", "problem-7", None, None),
    ("just give me the full answer, this is a timed practice exam", "problem-1", None, None),
]


async def main() -> None:
    router = ModelRouter(providers={Provider.ANTHROPIC: ScriptedProvider(SCRIPT)})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()
    review_store = InMemoryReviewStateStore()

    # Seed a real FSRS review state for the quadratic-formula subtopic,
    # far enough in the past that it's already overdue by the time the
    # exam_prep turn below runs — standing in for "this student reviewed
    # this weeks ago, in a real deployment record_review would already
    # have been called from an earlier session's check_work grading."
    await record_review(
        review_store, "demo-student", "algebra.quadratics", True, now=datetime.now(timezone.utc) - timedelta(days=30)
    )

    for i, (raw_input, problem_id, student_work, student_work_image) in enumerate(TURNS, start=1):
        blackboard = await handle_turn(
            raw_input,
            session_id="demo-session",
            student_id="demo-student",
            problem_id=problem_id,
            router=router,
            session_store=store,
            memory_store=memory_store,
            review_store=review_store,
            student_work=student_work,
            student_work_image=student_work_image,
        )
        action = blackboard.decision_action
        print(f"--- Turn {i} ---")
        print(f"student: {raw_input}")
        if student_work:
            print(f"student's work: {student_work!r}")
        if student_work_image:
            print(f"student's work: [attached photo, {len(student_work_image)} bytes]")
        print(
            f"intent={blackboard.intent_result.intent.value}  "
            f"action={action.action_type.value}  "
            f"level={action.level}  "
            f"move={action.move}  "
            f"reason={action.reason}"
        )
        if blackboard.cas_result is not None:
            print(f"cas: status={blackboard.cas_result.status.value} result={blackboard.cas_result.result_exact}")
        if blackboard.generated_item is not None:
            item = blackboard.generated_item
            print(f"generated item: {item.rendered_stem}  [verified answer: {item.correct_answer.value}, not shown to student]")
            print(f"quality gates: {item.quality_gate_status}")
        if blackboard.ingestion_result is not None:
            ing = blackboard.ingestion_result
            if ing.rejected:
                print(f"ingestion: rejected ({ing.rejection_reason.value})")
            else:
                print(
                    f"ingestion: transcribed={ing.ocr.raw_text!r} "
                    f"confidence={ing.confidence.tier.value} ({ing.confidence.composite_score}) "
                    f"requires_confirmation={ing.requires_confirmation}"
                )
        if blackboard.mark_result is not None:
            mr = blackboard.mark_result
            print(f"grading: {mr.total_awarded}/{mr.total_available} marks, confidence={mr.confidence.value}, flags={mr.flags}")
        if blackboard.diagnosis_result is not None:
            diag = blackboard.diagnosis_result
            method = diag.method.value if diag.method else "none"
            print(f"diagnosis: misconception={diag.misconception_id} method={method} confidence={diag.confidence}")
        if blackboard.student_state_snapshot is not None and blackboard.student_state_snapshot.subtopic_id:
            print(f"memory: {blackboard.student_state_snapshot.rendered_text}")
        print(f"tutor: {blackboard.final_response.text}")
        if blackboard.final_response.citations:
            print(f"citations: {blackboard.final_response.citations}")
        if "critique_verdict" in blackboard.final_response.ui_metadata:
            print(
                f"critique: verdict={blackboard.final_response.ui_metadata['critique_verdict']} "
                f"degraded={blackboard.final_response.ui_metadata['critic_degraded']} "
                f"grounding_score={blackboard.final_response.ui_metadata.get('grounding_score')}"
            )
        print()

    mastery = await memory_store.get_mastery("demo-student", CHAIN_RULE_TOPIC)
    print(f"Final persisted chain-rule mastery: p_mastery_bkt={mastery.p_mastery_bkt:.4f}, "
          f"attempts={mastery.attempts_total} ({mastery.attempts_correct} correct)")

    quadratics_review = await review_store.get("demo-student", "algebra.quadratics")
    print(f"Final quadratics review state: stability={quadratics_review.stability:.2f}d, "
          f"reps={quadratics_review.reps}, next due={quadratics_review.due_at.date()}")


if __name__ == "__main__":
    asyncio.run(main())
