# AI Tutor - Phase 1: Reasoning Core

Phase 1 of the intelligence layer for an AI-native IB DP Math AA tutoring assistant, built to the Engineering Blueprint v1.0 spec.

## What this does

- Classifies each student message into an intent: solve_request, check_work, concept_explain, exam_prep, ia_ee_help, or general_chat.
- Runs a pure, unit-tested decision function that picks a pedagogical action (EXPLAIN, HINT, QUESTION, REFUSE, CHALLENGE, SUPPORTIVE_SCAFFOLD), following the spec's decision policy exactly, including hard gates for academic integrity and exam conditions.
- Passes that action to a Tutor agent as a binding contract. The agent cannot produce a full solution when the action is HINT or QUESTION. This is enforced by a structural check on the generated text, not just prompt wording.
- Tracks a per-problem hint ladder in session state. It escalates across repeated attempts and resets when the student moves to a new problem.
- Falls back to a templated, non-LLM response if generation fails, times out, or violates the action's contract.
- Streams the approved response to the client.
- Routes every model call through one config file, so no model name is hardcoded anywhere else in the codebase.

Not built in this phase: retrieval/knowledge base, CAS/SymPy verification, real curriculum content, question generation, grading, persisted mastery model, IA supervisor, adaptive learning engine, OCR/multimodal input. These are stubbed with TODOs pointing at the relevant spec section.

## Structure

```
app/
  models/contracts.py      typed contracts: IntentResult, DecisionSignals, Action, TutorResponse, Blackboard
  policy/decision.py        the pure decision policy function
  agents/                   router/intent agent, tutor agent, prompt template, fallback templates
  llm/                      model router: one config file for all model names, one call() entrypoint
  session/state.py          hint ladder session state
  orchestrator/             wires everything together into handle_turn()
  main.py                   minimal FastAPI gateway
scripts/run_scripted_conversation.py   runnable demo, no API key needed
tests/                      unit + integration tests
```

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the tests (55 tests, all mocked, no API key or network needed):

```bash
pytest -q
```

Run the scripted demo conversation:

```bash
python scripts/run_scripted_conversation.py
```

Run the gateway (optional, needs ANTHROPIC_API_KEY to actually generate):

```bash
uvicorn app.main:app --reload
```
