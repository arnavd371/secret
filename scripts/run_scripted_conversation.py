"""
Runnable demo of the reasoning core through Phase 6: a scripted, multi-turn
conversation through `handle_turn`, with a mocked model provider so it runs
with no API key and no network access. The Router/Intent classification and
Tutor generation are mocked; the decision policy, hint ladder, CAS
verification (real SymPy), retrieval (real lexical search over the seed
knowledge base), question generation (real, quality-gated, CAS-verified
items), grading (real step segmentation + alignment + mark awarding),
memory (real BKT/IRT mastery updates, decay, and budgeted context
assembly), and the Verifier/Critic + grounding check (Phase 6) all run for
real on every turn — this script's mocked provider auto-passes the
critic's checklist call so the narrative isn't interrupted by it, but the
"critique" line printed per turn shows it genuinely ran. Explicit
block/revise/regenerate scenarios are covered in
tests/test_integration_critic.py and tests/test_tutor_agent.py instead of
here, to keep this script's queue bookkeeping manageable.

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
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

    async def generate(self, *, spec, system, user) -> LLMCallResult:
        if system.startswith("You are a strict, checklist-driven critic"):
            return LLMCallResult(text='{"verdict": "pass", "violations": []}', model=spec.model, provider=Provider.ANTHROPIC)
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


_CORRECT_CHAIN_RULE_WORK = "u = 2*x + 1\ntherefore dy/dx = 5*(2*x + 1)**4 * 2"

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
    # REFUSE short-circuits before the Tutor agent is ever called, so this
    # last entry is scripted but must be left unconsumed — it stays last
    # in the queue on purpose (see the assertion in the integration test).
    _intent_json("solve_request", assessment_mode_guess="live_exam_simulation"),
    "THIS SHOULD NEVER BE SHOWN — the exam-mode hard gate must refuse before this is reached.",
]

# (raw_input, problem_id, student_work)
TURNS: list[tuple[str, str, Optional[str]]] = [
    ("I'm stuck on differentiating x^2 * sin(x)", "problem-1", None),
    ("still not sure what to do", "problem-1", None),
    ("I tried but I'm still lost", "problem-1", None),
    ("that hint didn't help either", "problem-1", None),
    ("differentiate x**2 * cos(x)", "problem-2", None),
    ("can you check my work? differentiate (2*x+1)**5", "problem-3", _CORRECT_CHAIN_RULE_WORK),
    ("can you check my work? differentiate (2*x+1)**5", "problem-3", _CORRECT_CHAIN_RULE_WORK),
    ("can you check my work? differentiate (2*x+1)**5", "problem-3", _CORRECT_CHAIN_RULE_WORK),
    ("I'm working on differentiating (3x-2)^4 now", "problem-4", None),
    ("I solved it correctly!", "problem-4", None),
    ("can you remind me how the chain rule works?", "problem-4", None),
    ("just give me the full answer, this is a timed practice exam", "problem-1", None),
]


async def main() -> None:
    router = ModelRouter(providers={Provider.ANTHROPIC: ScriptedProvider(SCRIPT)})
    store = InMemorySessionStateStore()
    memory_store = InMemoryMemoryStore()

    for i, (raw_input, problem_id, student_work) in enumerate(TURNS, start=1):
        blackboard = await handle_turn(
            raw_input,
            session_id="demo-session",
            student_id="demo-student",
            problem_id=problem_id,
            router=router,
            session_store=store,
            memory_store=memory_store,
            student_work=student_work,
        )
        action = blackboard.decision_action
        print(f"--- Turn {i} ---")
        print(f"student: {raw_input}")
        if student_work:
            print(f"student's work: {student_work!r}")
        print(
            f"intent={blackboard.intent_result.intent.value}  "
            f"action={action.action_type.value}  "
            f"level={action.level}  "
            f"move={action.move}"
        )
        if blackboard.cas_result is not None:
            print(f"cas: status={blackboard.cas_result.status.value} result={blackboard.cas_result.result_exact}")
        if blackboard.generated_item is not None:
            item = blackboard.generated_item
            print(f"generated item: {item.rendered_stem}  [verified answer: {item.correct_answer.value}, not shown to student]")
            print(f"quality gates: {item.quality_gate_status}")
        if blackboard.mark_result is not None:
            mr = blackboard.mark_result
            print(f"grading: {mr.total_awarded}/{mr.total_available} marks, confidence={mr.confidence.value}, flags={mr.flags}")
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


if __name__ == "__main__":
    asyncio.run(main())
