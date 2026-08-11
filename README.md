# AI Tutor - Phases 1-4: Reasoning Core, Verification/Retrieval, Question Generation, Grading

The intelligence layer for an AI-native IB DP Math AA tutoring assistant, built to the Engineering Blueprint v1.0 spec. Covers Phase 1 (reasoning core), Phase 2 (CAS verification + retrieval), Phase 3 (question generation), and Phase 4 (grading / AI examiner).

## What this does

Phase 1, reasoning core:

- Classifies each student message into an intent: solve_request, check_work, concept_explain, exam_prep, ia_ee_help, or general_chat.
- Runs a pure, unit-tested decision function that picks a pedagogical action (EXPLAIN, HINT, QUESTION, REFUSE, CHALLENGE, SUPPORTIVE_SCAFFOLD), following the spec's decision policy exactly, including hard gates for academic integrity and exam conditions.
- Passes that action to a Tutor agent as a binding contract. The agent cannot produce a full solution when the action is HINT, QUESTION, or CHALLENGE. This is enforced by a structural check on the generated text, not just prompt wording.
- Tracks a per-problem hint ladder in session state. It escalates across repeated attempts and resets when the student moves to a new problem.
- Falls back to a templated, non-LLM response if generation fails, times out, or violates the action's contract.
- Streams the approved response to the client.
- Routes every model call through one config file, so no model name is hardcoded anywhere else in the codebase.

Phase 2, verification and retrieval, only runs for EXPLAIN (the one action type allowed to state a final answer):

- Extracts a checkable math task from the student's message (differentiate, integrate, solve, simplify, evaluate) and computes the real answer with SymPy.
- If the Tutor's draft states a final value that disagrees with the SymPy result, the draft is discarded and replaced with a response built directly from the verified result. If SymPy can't verify anything (unsolvable, malformed, timed out), the draft is discarded and the response degrades to asking the student to work through it together instead.
- Retrieves relevant chunks from a small seed knowledge base (a handful of real IB AA formulas, learning objectives, and one worked example) using TF-IDF keyword matching plus a topic-hint boost, and attaches citations to the response when a chunk clears the confidence threshold.

Phase 3, question generation, runs for CHALLENGE (a high-mastery student gets a harder problem instead of a full solve):

- Samples parameters for a real item template (product rule, chain rule, power rule, or quadratic formula), computes the answer with SymPy, and checks it against quality gates: solvability, uniqueness, number-friendliness for non-calculator items, style compliance, and duplicate/leakage prevention. Failed gates trigger resampling, bounded by a retry limit.
- Generates a distractor tied to a real, named misconception for each template (e.g. forgetting the chain rule's inner-derivative factor).
- Builds a mark scheme directly from the CAS solution steps.
- The Tutor agent presents the generated item as the new problem to attempt. The item's own answer is never revealed, same structural enforcement as HINT/QUESTION.

Phase 4, grading, runs for check_work when the student's typed working is provided alongside the problem:

- Segments the submission into discrete steps (algebraic manipulation, final answer, justification, restatement of the given) and checks each against the mark scheme using real symbolic equivalence, not string matching.
- Awards accuracy marks by matching a step's value to the CAS-computed answer (including multi-root answers like a quadratic's two solutions, matched as a set across the submission). Awards method marks by a documented heuristic (real intermediate working shown, not just the given restated).
- Flags a correct final answer with too little supporting work as unsupported, with a real coverage threshold.
- Builds a grounded examiner comment directly from the mark breakdown, no LLM call, no unsupported claims. This response bypasses the Tutor LLM entirely: once the marks are computed there's nothing left for a model to add.
- Scores a confidence tier (high/medium/low) so low-confidence gradings can be flagged for review later.

Not built yet: real curriculum/template content beyond the seed set, persisted mastery model, IA supervisor, adaptive learning engine, OCR/multimodal input, dense/vector retrieval, a reranker, LLM-authored item variants, IRT recalibration from response history, grade-boundary calibration from historical exam data, human-in-the-loop review/appeals. These are stubbed with TODOs pointing at the relevant spec section.

## Structure

```
app/
  models/contracts.py      typed contracts: IntentResult, DecisionSignals, Action, TutorResponse, Blackboard
  policy/decision.py        the pure decision policy function
  agents/                   router/intent agent, tutor agent, prompt template, fallback templates
  cas/                      Math Solver + CAS agent: SymPy solver, verify_claim, task extraction from text
  knowledge/                seed curriculum content + TF-IDF retriever
  questions/                item templates, parametric generator, distractors, mark scheme, quality gates
  examiner/                 step segmentation, alignment, mark awarding, grounded comment generation
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

Run the tests (151 tests, all mocked at the model boundary, no API key or network needed; SymPy, retrieval, item generation, and grading all run for real):

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
