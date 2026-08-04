# AI Tutor — Phase 1: Reasoning Core

Phase 1 of the intelligence layer described in *AI-Native Academic
Assistant for the IB DP — Engineering Blueprint v1.0* (Mathematics: Analysis
& Approaches HL/SL). This phase builds the orchestrator: a real
decision-making core made of independently testable, typed components — a
deterministic policy the LLM cannot override, and an execution path where
the model is one interchangeable component, not the whole program.

No real curriculum content, retrieval, CAS/SymPy verification, memory
persistence, or multimodal ingestion is implemented here — those are later
phases of the blueprint (see "Non-goals" below).

## File tree

```
.
├── app/
│   ├── main.py                     # Phase 0 FastAPI gateway (minimal): /health, /turn
│   ├── config.py                   # Settings (env-driven)
│   ├── db/
│   │   └── session.py              # Minimal async Postgres engine skeleton (Phase 0)
│   ├── models/
│   │   └── contracts.py            # IntentResult, DecisionSignals, Action, TutorResponse, SafetyResult, Blackboard
│   ├── policy/
│   │   └── decision.py             # decide_pedagogical_action — pure function, zero I/O, §1.5 pseudocode
│   ├── agents/
│   │   ├── router_agent.py         # Router/Intent agent (§2.2): structured output + confidence fallback
│   │   ├── templates.py            # Tutor Agent System Prompt Skeleton (§7.7), one shared template
│   │   ├── tutor_agent.py          # Tutor agent: generation + structural leak-check gate + streaming
│   │   └── fallback.py             # Templated, non-LLM fallback responses
│   ├── llm/
│   │   ├── router_config.py        # THE single place model names/providers are configured
│   │   └── client.py               # ModelRouter.call()/.stream() — providers behind one interface
│   ├── session/
│   │   └── state.py                # Hint-ladder session state (§7.2): stores + escalation math
│   └── orchestrator/
│       ├── signals.py              # Heuristic Safety/Integrity + frustration estimators (stand-ins for §2.2 agents)
│       └── handle_turn.py          # Wires everything together; hard-gates REFUSE before the Tutor agent
├── scripts/
│   └── run_scripted_conversation.py  # Runnable, no-API-key demo of a multi-turn conversation
├── tests/
│   ├── test_contracts.py             # Pydantic boundary validation
│   ├── test_decision_policy.py       # Full branch + branch-ordering coverage of the §1.5 pseudocode
│   ├── test_session_state.py         # Hint ladder escalation/reset rules
│   ├── test_router_agent.py          # Confidence fallback, parse-failure fallback, outage fallback
│   ├── test_tutor_agent.py           # Structural leak-check gate, provider-outage fallback
│   └── test_integration_handle_turn.py  # Scripted 5-turn conversation, hard gate, leak-check, end to end
├── requirements.txt
├── pytest.ini
├── .env.example
└── .gitignore
```

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Unit + integration tests** (55 tests, no API key or network needed — all
model calls are mocked):

```bash
pytest -q
```

**Scripted multi-turn demo** (prints a 5-turn conversation showing the hint
ladder escalate per §1.5's exact math, and the live-exam hard gate refuse,
using a mocked model provider):

```bash
python scripts/run_scripted_conversation.py
```

**Run the Phase 0 gateway** (optional — not required to see Phase 1 work;
needs `ANTHROPIC_API_KEY` set to actually generate, Postgres/Redis are
best-effort and fall back gracefully if unavailable):

```bash
uvicorn app.main:app --reload
```

## What "done" looks like, mapped to code

1. **Intent classification per turn** → `app/agents/router_agent.py::classify_intent` (spec §2.2)
2. **Pure decision policy, not an LLM call** → `app/policy/decision.py::decide_pedagogical_action` — a direct, branch-for-branch implementation of the §1.5 pseudocode, zero I/O, fully unit-tested in `tests/test_decision_policy.py`
3. **Action as a binding, structurally-enforced contract** → `app/agents/tutor_agent.py`: the §7.7 system prompt skeleton (`templates.py`) asks nicely; `_violates_action_contract` + regex leak patterns are the real enforcement — a HINT/QUESTION draft that looks like it states a final answer is discarded and replaced with a templated fallback, unconditionally.
4. **Hint ladder escalation/reset** → `app/session/state.py::advance_session_state`, exercised turn-by-turn in `tests/test_integration_handle_turn.py::test_scripted_conversation_hint_ladder_escalates_and_resets`.
5. **Templated fallback on failure/timeout** → `app/agents/fallback.py`, invoked from both `router_agent.py` (intent classify failure) and `tutor_agent.py` (generation failure/timeout/leak).
6. **Streaming** → `tutor_agent.stream_response`: buffer-then-check, matching spec §2.6's "full post-hoc critique...for high-risk paths" strategy — the full draft is generated and passed through the structural gate first, then the *approved* text is chunked out. Nothing ungated is ever emitted.
7. **Zero hardcoded model names outside one router config** → `app/llm/router_config.py::CAPABILITY_MODEL_MAP` is the only file with a model string in it; everything else calls `router.call(capability=...)`.

## Deviations from the spec, and why

The spec (Engineering Blueprint v1.0) was provided after the initial build,
so the contracts and policy below were reconciled to match it exactly —
field names, enum values, and the §1.5 pseudocode's branch order/level-cap
math are implemented verbatim. The deviations below are Phase 1 scoping
decisions, not disagreements with the spec:

- **`route_to_ia_supervisor` and `schedule_next_review_item` are stubs.**
  The §1.5 pseudocode calls these two functions directly. The real IA
  Supervisor Agent (§11 — state machine, guard architecture, disclosure
  logging) and the real Adaptive Learning Engine (§12 — FSRS scheduling,
  mastery-threshold bands) are far outside a Phase 1 reasoning core.
  `_route_to_ia_supervisor` (in `decision.py`) preserves the one property
  that matters at this layer — `ia_ee_help` is never routed to a normal
  solve/explain path — by refusing with an `ia_methodology_coaching` offer.
  `_schedule_next_review_item` falls back to the decision table's own
  described behavior for that cell ("question (retrieval-practice style)"),
  deterministically. Both are marked `TODO(Section 11)` / `TODO(Section 12)`.

- **Safety/Integrity and frustration signals are keyword heuristics, not
  the real agents.** Spec §2.2 specifies a dedicated fast-classifier
  Safety/Integrity Agent and a "Tutor agent sentiment/behavior model" for
  frustration. Phase 1 implements both as small, deterministic keyword/rule
  functions in `app/orchestrator/signals.py` — testable and spec-shaped
  (same output enums: `IntegrityRisk` low/medium/high, `FrustrationLevel`
  none/mild/high) but not the trained classifiers the full system calls for.

- **Hint ladder de-escalation is not implemented.** Spec §7.2's ladder
  table has explicit "drop to level N-1 if next attempt is correct" rules.
  Determining correctness requires the Math Solver + CAS agent (§2.2),
  which doesn't exist until Phase 2. `advance_session_state` only
  escalates (matching what §1.5's pseudocode itself computes:
  `level = min(hint_ladder_level + 1, cap)`, which is escalation-only by
  construction) and resets to 0 on a new problem. De-escalation is a
  documented `TODO(Phase 2)` in `app/session/state.py`.

- **Mastery estimate is a passed-in constant, not a real BKT/IRT model.**
  Spec §4.3 defines a full Bayesian Knowledge Tracing + Item Response
  Theory hybrid mastery model (Student Memory System, §4) — a large
  standalone subsystem. `handle_turn` accepts `mastery_estimate` as an
  explicit parameter defaulting to a neutral `0.5`
  (`app/orchestrator/signals.py::DEFAULT_MASTERY_ESTIMATE`), so the
  mastery-based CHALLENGE branch is real, tested code, just fed a stub
  value until §4's memory system exists.

- **`handle_turn` returns the full `Blackboard`, not a bare
  `TutorResponse`.** The response text is at `blackboard.final_response.text`
  (matching the spec's own field name); returning the whole Blackboard lets
  callers/tests also inspect `intent_result`, `safety_result`, and
  `decision_action` — which is exactly what the Blackboard concept in §2.5
  is for.

- **One shared Tutor system-prompt skeleton, not five separate templates.**
  Spec §7.7 gives a single "Tutor Agent System Prompt Skeleton" with a
  `BOUND ACTION` line and hard constraints that vary by action type — that
  is what's implemented in `templates.py`. This does not weaken enforcement:
  the skeleton's constraints are structured/declarative, not "if user says
  X do Y" prose branching, and the real enforcement layer is the structural
  leak-check in `tutor_agent.py`, independent of what the prompt says.

- **Orchestrator runs a fixed linear sequence, not the Planner's parallel
  stage graph.** Spec §6.3's `handle_turn` pseudocode fans out Router/
  Intent, Safety/Integrity, and Memory state-load concurrently via
  `asyncio.gather`, and only Retrieval/CAS/Diagnosis run in parallel after
  that. Phase 1 has none of Retrieval, CAS, Memory, or Diagnosis built yet
  (all explicit non-goals below), so there is nothing to parallelize —
  `handle_turn` runs Router/Intent → Safety heuristic → policy → Tutor
  agent sequentially. The `TODO`s on `Blackboard`'s stubbed fields mark
  exactly where those stages plug back in.

## Non-goals (explicitly out of scope for Phase 1)

Stubbed with `TODO` comments at the relevant point in the code, referencing
the blueprint section that owns the real implementation:

- Retrieval / knowledge base — `Blackboard.retrieved_chunks` (§5)
- CAS/SymPy verification — `Blackboard.cas_result`, Math Solver+CAS Agent (§2.2, §1.4)
- Real curriculum content, Planner agent, parallel stage graph (§2.2, §6)
- Question generation (§9)
- Grader/Examiner agent (§10)
- Misconception Diagnostician — `Blackboard.diagnosis_result` (§8)
- IA Supervisor Agent — stubbed in `decision.py::_route_to_ia_supervisor` (§11)
- Persisted mastery model (BKT/IRT), Student Memory System — `app/orchestrator/signals.py::DEFAULT_MASTERY_ESTIMATE` (§4)
- Adaptive Learning Engine / spaced repetition — stubbed in `decision.py::_schedule_next_review_item` (§12)
- Full AI Quality Control layer (self-consistency, retrieval-grounding, critic model) — Phase 1 approximates only the leak-check slice of this in `tutor_agent.py` (§13)
- OCR/multimodal ingestion — `IntentResult.requires_multimodal_parse` is classified but unused, `Blackboard.normalized_input.ocr_confidence` stubbed (§3.2)
