# AI Tutor - Phases 1-2: Reasoning Core + Verification/Retrieval

The intelligence layer for an AI-native IB DP Math AA tutoring assistant, built to the Engineering Blueprint v1.0 spec. Covers Phase 1 (reasoning core) and Phase 2 (CAS verification + retrieval).

## What this does

Phase 1, reasoning core:

- Classifies each student message into an intent: solve_request, check_work, concept_explain, exam_prep, ia_ee_help, or general_chat.
- Runs a pure, unit-tested decision function that picks a pedagogical action (EXPLAIN, HINT, QUESTION, REFUSE, CHALLENGE, SUPPORTIVE_SCAFFOLD), following the spec's decision policy exactly, including hard gates for academic integrity and exam conditions.
- Passes that action to a Tutor agent as a binding contract. The agent cannot produce a full solution when the action is HINT or QUESTION. This is enforced by a structural check on the generated text, not just prompt wording.
- Tracks a per-problem hint ladder in session state. It escalates across repeated attempts and resets when the student moves to a new problem.
- Falls back to a templated, non-LLM response if generation fails, times out, or violates the action's contract.
- Streams the approved response to the client.
- Routes every model call through one config file, so no model name is hardcoded anywhere else in the codebase.

Phase 2, verification and retrieval, only runs for EXPLAIN/CHALLENGE (the two action types allowed to state a final answer):

- Extracts a checkable math task from the student's message (differentiate, integrate, solve, simplify, evaluate) and computes the real answer with SymPy.
- If the Tutor's draft states a final value that disagrees with the SymPy result, the draft is discarded and replaced with a response built directly from the verified result. If SymPy can't verify anything (unsolvable, malformed, timed out), the draft is discarded and the response degrades to asking the student to work through it together instead.
- Retrieves relevant chunks from a small seed knowledge base (a handful of real IB AA formulas, learning objectives, and one worked example) using TF-IDF keyword matching plus a topic-hint boost, and attaches citations to the response when a chunk clears the confidence threshold.

Not built yet: real curriculum content beyond the seed set, question generation, grading, persisted mastery model, IA supervisor, adaptive learning engine, OCR/multimodal input, dense/vector retrieval, a reranker. These are stubbed with TODOs pointing at the relevant spec section.

## Structure

```
app/
  models/contracts.py      typed contracts: IntentResult, DecisionSignals, Action, TutorResponse, Blackboard
  policy/decision.py        the pure decision policy function
  agents/                   router/intent agent, tutor agent, prompt template, fallback templates
  cas/                      Math Solver + CAS agent: SymPy solver, verify_claim, task extraction from text
  knowledge/                seed curriculum content + TF-IDF retriever
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

Run the tests (99 tests, all mocked at the model boundary, no API key or network needed; SymPy and retrieval run for real):

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
