# AI Tutor — Phase 1: Reasoning Core

Phase 1 of the 9-phase roadmap for an AI-native academic tutoring assistant
(IB DP Math AA, extensible later). This phase builds the orchestrator: a
real decision-making core made of independently testable, typed components
— a deterministic policy the LLM cannot override, and an execution path
where the model is one interchangeable component, not the whole program.

No real curriculum content, retrieval, or CAS/SymPy verification is
implemented here — those are later phases (see "Non-goals" below).

## File tree

```
.
├── app/
│   ├── main.py                     # Phase 0 FastAPI gateway (minimal): /health, /turn
│   ├── config.py                   # Settings (env-driven)
│   ├── db/
│   │   └── session.py              # Minimal async Postgres engine skeleton (Phase 0)
│   ├── models/
│   │   └── contracts.py            # IntentResult, DecisionSignals, Action, TutorResponse, Blackboard
│   ├── policy/
│   │   └── decision.py             # decide_pedagogical_action — pure function, zero I/O
│   ├── agents/
│   │   ├── router_agent.py         # Router/Intent agent (structured output + confidence fallback)
│   │   ├── templates.py            # Per-action-type system prompt templates
│   │   ├── tutor_agent.py          # Tutor agent: generation + structural leak-check gate + streaming
│   │   └── fallback.py             # Templated, non-LLM fallback responses
│   ├── llm/
│   │   ├── router_config.py        # THE single place model names/providers are configured
│   │   └── client.py               # ModelRouter.call()/.stream() — providers behind one interface
│   ├── session/
│   │   └── state.py                # Hint-ladder session state: stores + escalate/de-escalate policy
│   └── orchestrator/
│       ├── signals.py              # Heuristic integrity-risk / frustration estimators
│       └── handle_turn.py          # Wires everything together; hard-gates REFUSE before the Tutor agent
├── scripts/
│   └── run_scripted_conversation.py  # Runnable, no-API-key demo of a multi-turn conversation
├── tests/
│   ├── test_contracts.py             # Pydantic boundary validation
│   ├── test_decision_policy.py       # Full branch + branch-ordering coverage of the policy
│   ├── test_session_state.py         # Hint ladder escalate/de-escalate/reset rules
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
ladder escalate and the exam-mode hard gate refuse, using a mocked model
provider):

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

1. **Intent classification per turn** → `app/agents/router_agent.py::classify_intent`
2. **Pure decision policy, not an LLM call** → `app/policy/decision.py::decide_pedagogical_action` (zero I/O, fully unit-tested in `tests/test_decision_policy.py`)
3. **Action as a binding, structurally-enforced contract** → `app/agents/tutor_agent.py`: the system prompt template (`templates.py`) asks nicely; `_violates_action_contract` + regex leak patterns are the real enforcement — a HINT/QUESTION draft that looks like it states a final answer is discarded and replaced with a templated fallback, unconditionally.
4. **Hint ladder escalation/reset** → `app/session/state.py::apply_turn_outcome`, exercised turn-by-turn in `tests/test_integration_handle_turn.py::test_scripted_conversation_hint_ladder_escalates_and_resets`.
5. **Templated fallback on failure/timeout** → `app/agents/fallback.py`, invoked from both `router_agent.py` (intent classify failure) and `tutor_agent.py` (generation failure/timeout/leak).
6. **Streaming** → `tutor_agent.stream_response`: buffer-then-check — the full draft is generated and passed through the structural gate first, then the *approved* text is chunked out. Nothing ungated is ever emitted.
7. **Zero hardcoded model names outside one router config** → `app/llm/router_config.py::CAPABILITY_MODEL_MAP` is the only file with a model string in it; everything else calls `router.call(capability=...)`.

## Deviations from the spec pseudocode, and why

I did not have the actual engineering spec document (the `§` sections
referenced in the brief) in hand — only the field names, enums, and branch
descriptions given inline in the task itself. I treated that inline
description as the source of truth and made the following concrete,
documented choices where the pseudocode wasn't fully specified:

- **Hint ladder level vs. attempt count.** I treat `hint_ladder_level` (not
  `attempt_count` directly) as what determines the *action* on the
  Socratic ladder: `attempt_count == 0` always restarts Socratic
  questioning (a fresh attempt shouldn't inherit a stale hint level);
  otherwise `hint_ladder_level == 0` keeps asking Socratic questions, and
  `hint_ladder_level` 1–4 map directly to HINT levels 1–4, with level 4
  additionally setting `offer="offer_full_solution_after_attempt"`.
  `attempt_count` is session state that drives *when* the ladder escalates
  (see below), not the level number itself.

- **§7.2 escalation/de-escalation rules (not available to me).** I
  implemented: two consecutive incorrect/unresolved attempts on the same
  problem escalate the ladder by one level (not on the very first miss, so
  one wrong try doesn't immediately jump to a strong hint); a correct
  attempt de-escalates the ladder by one level and resets `attempt_count`,
  rewarding demonstrated understanding; an explicit hint request escalates
  immediately regardless of `attempt_count`; any problem_id change resets
  both counters to zero. This is a reasonable, defensible policy given the
  actual rules weren't available to me — it should be reconciled against
  the real §7.2 text if/when it's available.

- **No CAS/answer verification exists yet (correctly, per non-goals), so
  the orchestrator cannot know if an attempt was actually correct.**
  `handle_turn._infer_turn_outcome` treats a same-problem `practice`-intent
  turn as another unresolved attempt (i.e. "incorrect" in ladder terms),
  since Phase 1 has no way to confirm correctness. `handle_turn` accepts
  an optional `turn_outcome` override so a future caller that *does* know
  the real outcome (e.g. once Phase 2's CAS check exists) can pass it in
  directly instead of relying on this inference.

- **Mastery estimate.** Phase 5 owns persisted mastery; Phase 1 has
  nothing to estimate it from. `handle_turn` takes `mastery_estimate` as
  an explicit parameter defaulting to a neutral `0.5` constant
  (`app/orchestrator/signals.py::DEFAULT_MASTERY_ESTIMATE`), so the
  mastery-based CHALLENGE shortcut is real, tested code — just fed a stub
  value until Phase 5 exists.

- **Integrity risk and frustration signals.** Implemented as small
  keyword/phrase heuristics in `app/orchestrator/signals.py`, clearly
  marked as stubs (`TODO(Phase 4)` for a real integrity signal from the
  grading/examiner agent). The spec references these as inputs to the
  policy without specifying how they're computed in Phase 1, so I built
  the simplest defensible version rather than leaving them unimplemented.

- **`handle_turn` returns the full `Blackboard`, not a bare
  `TutorResponse`.** The brief's step 6 says "returns the response," but
  tests need to assert on the intermediate `IntentResult`/
  `DecisionSignals`/`Action` as well (e.g. "hint ladder escalates
  correctly" requires inspecting `decision_signals.hint_ladder_level`).
  The actual reply text is at `blackboard.tutor_response.text`; returning
  the Blackboard costs nothing and makes the system's decisions
  inspectable, which the spec's own Blackboard concept seems designed for.

- **Streaming implementation is buffer-then-check, not token-by-token
  release.** The brief explicitly allows either approach as long as the
  check gates release. I chose buffer-then-check because it's the only
  approach where the leak-check can see the complete text before anything
  is shown — a token-by-token release could leak a partial "the answer
  is..." before the check has enough text to catch it.

## Non-goals (explicitly out of scope for Phase 1)

Stubbed with `TODO(Phase N)` comments at the relevant point in the code:

- Retrieval / knowledge base — `Blackboard.retrieved_chunks` (Phase 2+)
- CAS/SymPy verification — `Blackboard.cas_result` (Phase 2)
- Real curriculum content (Phase 2)
- Question generation (Phase 3)
- Grading/examiner, real integrity detection — `app/orchestrator/signals.py` (Phase 4)
- Persisted mastery model — `app/orchestrator/signals.py::DEFAULT_MASTERY_ESTIMATE` (Phase 5)
- OCR/multimodal — `IntentResult.requires_multimodal_parse` is classified but unused (Phase 7)
