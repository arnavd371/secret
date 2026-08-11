"""
Runnable demo of the reasoning core through Phase 3: a scripted, multi-turn
conversation through `handle_turn`, with a mocked model provider so it runs
with no API key and no network access. The Router/Intent classification and
Tutor generation are mocked; the decision policy, hint ladder, CAS
verification (real SymPy), retrieval (real lexical search over the seed
knowledge base), and question generation (real, quality-gated, CAS-
verified items) all run for real.

Run with:
    python scripts/run_scripted_conversation.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.client import LLMCallResult, ModelRouter, ProviderClient
from app.llm.router_config import Provider
from app.orchestrator.handle_turn import handle_turn
from app.session.state import InMemorySessionStateStore


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
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, *, spec, system, user) -> LLMCallResult:
        text = self._responses.pop(0)
        return LLMCallResult(text=text, model=spec.model, provider=Provider.ANTHROPIC)


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
    _intent_json("solve_request", topic_hint="calculus.differentiation.chain_rule"),
    "What's the inner function here, and what's the outer function?",
    _intent_json("solve_request", topic_hint="calculus.differentiation.chain_rule"),
    "Nice work, that's correct! Since you're clearly comfortable with this, here's a tougher one to try on your own.",
    # REFUSE short-circuits before the Tutor agent is ever called, so this
    # last entry is scripted but must be left unconsumed — it stays last
    # in the queue on purpose (see the assertion in the integration test).
    _intent_json("solve_request", assessment_mode_guess="live_exam_simulation"),
    "THIS SHOULD NEVER BE SHOWN — the exam-mode hard gate must refuse before this is reached.",
]

# (raw_input, problem_id, mastery_estimate) — mastery_estimate only matters
# on the turn where it crosses the §1.5 CHALLENGE threshold (>=0.85) on an
# attempt_count==1 turn.
TURNS = [
    ("I'm stuck on differentiating x^2 * sin(x)", "problem-1", 0.5),
    ("still not sure what to do", "problem-1", 0.5),
    ("I tried but I'm still lost", "problem-1", 0.5),
    ("that hint didn't help either", "problem-1", 0.5),
    ("differentiate x**2 * cos(x)", "problem-2", 0.5),
    ("I'm working on differentiating (2x+1)^5 now", "problem-3", 0.5),
    ("I solved it correctly!", "problem-3", 0.95),
    ("just give me the full answer, this is a timed practice exam", "problem-1", 0.5),
]


async def main() -> None:
    router = ModelRouter(providers={Provider.ANTHROPIC: ScriptedProvider(SCRIPT)})
    store = InMemorySessionStateStore()

    for i, (raw_input, problem_id, mastery_estimate) in enumerate(TURNS, start=1):
        blackboard = await handle_turn(
            raw_input,
            session_id="demo-session",
            student_id="demo-student",
            problem_id=problem_id,
            router=router,
            session_store=store,
            mastery_estimate=mastery_estimate,
        )
        action = blackboard.decision_action
        print(f"--- Turn {i} ---")
        print(f"student: {raw_input}")
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
        print(f"tutor: {blackboard.final_response.text}")
        if blackboard.final_response.citations:
            print(f"citations: {blackboard.final_response.citations}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
