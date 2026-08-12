# AI Tutor - Phases 1-11: Reasoning Core, Verification/Retrieval, Question Generation, Grading, Memory, Quality Control, Multimodal Ingestion, Misconception Diagnosis, Adaptive Learning, IA/EE Supervisor, Planner

The intelligence layer for an AI-native IB DP Math AA tutoring assistant, built to the Engineering Blueprint v1.0 spec. Covers Phase 1 (reasoning core), Phase 2 (CAS verification + retrieval), Phase 3 (question generation), Phase 4 (grading / AI examiner), Phase 5 (student memory system), Phase 6 (AI quality control), Phase 7 (multimodal ingestion), Phase 8 (misconception diagnosis), Phase 9 (adaptive learning / spaced repetition), Phase 10 (IA/EE Supervisor), and Phase 11 (Planner).

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

Phase 5, memory, persists what each student actually knows across turns and sessions:

- Real Bayesian Knowledge Tracing (fast per-subtopic mastery) and Item Response Theory (slow-updating ability estimate) updates, exact formulas from the spec, applied after every confidently-graded check_work submission. Low-confidence gradings never write a mastery update, so a bad or incomplete submission can't corrupt the model.
- Real exponential decay toward a floor for stale mastery, and separate decay for a per-student misconception registry.
- A deterministic node-state classifier (unseen, introduced, practicing, consolidating, mastered, decayed) and a budgeted context-assembly function that injects the real mastery summary into the Tutor prompt's STUDENT MASTERY CONTEXT slot, replacing the placeholder every earlier phase left there.
- The CHALLENGE decision (Phase 3) now reads real persisted mastery instead of a flat default: a student who has actually demonstrated mastery through graded practice gets challenged automatically, with no explicit override needed. A caller can still pass an explicit mastery estimate to override this (useful for tests or a caller with its own signal).

Phase 6, quality control, runs on every Tutor draft that survives the structural leak-check and CAS gate (i.e. anything not already replaced by a templated/CAS-grounded fallback):

- A real, independent second model call (the Verifier/Critic) reviews the draft against a checklist: no answer leaked on a non-EXPLAIN action, no contradiction of the CAS-verified result, no invented content. On a timeout or unparseable response it degrades to a conservative static check (the same leak regex plus a LaTeX-balance check) rather than failing the turn, and says so in the response metadata (`critic_degraded`).
- A real lexical grounding check: for a draft with citations, flags claims that don't actually overlap with the cited content, so a critic that says "pass" doesn't override an ungrounded draft.
- A "block" verdict (or a failed grounding check) discards the draft for the templated fallback. A "revise" verdict triggers one bounded regeneration attempt with the critique's violations fed back as stricter constraints, re-checked against every prior gate; if that also fails, it falls back too.

Phase 7, multimodal ingestion, runs for check_work when the student attaches a photo of their work instead of typing it:

- Validates the image for real before touching a model: format (PNG/JPEG only), byte size, decodability, and pixel dimensions. A rejected image never reaches the vision model.
- Runs real PIL preprocessing: grayscale conversion, contrast normalization, and binarization with a real computed Otsu threshold (not a fixed cutoff), plus resizing into an OCR-friendly resolution band.
- Sends the processed image to a vision-capable model (the `math_ocr` capability) with a transcription-only prompt: read what's on the page, don't solve or correct it.
- Normalizes the raw transcription (strips markdown fences and math delimiters, resolves common LaTeX command aliases) and checks whether it contains a real parseable expression, reusing the same SymPy parser Phase 2's CAS layer already has.
- Scores a composite confidence (transcription length, expression parseability, LaTeX well-formedness) and gates on it: a high-confidence transcription is graded immediately through Phase 4's real grader, exactly as if the student had typed it. Anything less (a rejected image, a failed vision call, or a medium/low-confidence transcription) gets a templated response asking the student to confirm what was read, retype, or retake the photo. The Tutor LLM is never asked to grade an unconfirmed transcription.
- If the caller already has typed student work for the turn, the image is ignored entirely rather than re-transcribing over good text.

Phase 8, misconception diagnosis, runs whenever a check_work submission (typed or photographed) grades as incorrect:

- Tries a real, deterministic pattern match first: for the actual problem the student is working (not a fixed template), computes what a specific named error would produce with SymPy, and checks whether the student's stated answer matches that exactly. A match is high-trust by construction, confidence fixed at 1.0.
- Falls back to a model call (the `misconception_diagnose` capability) only when no pattern matches, constrained to choosing from the same fixed misconception catalog or reporting none rather than free-form diagnosis. Its own reported confidence is gated before anything gets persisted.
- Writes a confirmed diagnosis into the same per-student misconception registry Phase 5 already built and decays over time; a repeat diagnosis increments the existing entry instead of duplicating it.
- Closes the loop Phase 5 left open: the very next turn's memory context assembly (already real since Phase 5) automatically surfaces the misconception in the Tutor prompt's ACTIVE MISCONCEPTIONS slot, with no additional wiring needed.

Phase 9, adaptive learning, maintains a real spaced-repetition schedule per (student, subtopic) and drives exam_prep:

- A real FSRS-lite forgetting-curve model: a stability/difficulty state per subtopic, a retrievability curve that decays over time, and review updates that grow stability on a correct review (more so for a harder item or one recalled after a longer gap) and shrink it on an incorrect one, raising difficulty. Uses fixed, documented parameters rather than the published algorithm's per-user ML-fitted weights, which need training data this system doesn't have, the same honest-simplification posture as Phase 5's BKT/IRT defaults.
- Every graded check_work submission, typed or photographed, updates this schedule for its subtopic, independent of and in addition to Phase 5's mastery update.
- On exam_prep, a real due-review query decides the action for real instead of an unconditional stub: something genuinely overdue gets a retrieval-practice question bound to a real, CAS-verified generated item for that exact subtopic (reusing Phase 3's generator, the same way CHALLENGE does); nothing due gets a plain explanation instead of manufacturing a quiz out of nothing.
- The bound review item's answer is protected by the same structural leak check as a CHALLENGE item, extended to cover QUESTION drafts too.

Phase 10, IA/EE Supervisor, runs for ia_ee_help and replaces Phase 1's unconditional-refusal stub with real coaching, guarded against ever ghostwriting:

- A real regex/heuristic guard checks the request itself for a ghostwriting pattern ("write my introduction," "give me a research question," "do my IA for me," and similar) before any coaching is attempted. A detected request is refused, same as before, but now for a real, checked reason rather than unconditionally.
- A real project-stage state machine (topic selection, research question, methodology, analysis, drafting, revision) classifies what the student's message is actually about from real keywords and advances the project's persisted stage accordingly. Movement is deliberately permissive both forward and backward, since real IA/EE work is iterative, but a project marked COMPLETE (only reachable via genuine submission language, not just "I finished my draft") is a true terminal state: coaching closes for good.
- Coaching that clears the guard is real: routed through the normal Tutor path with a dedicated hard constraint block and a structural word-count cap (180 words), enforced the same way HINT/QUESTION/CHALLENGE's answer-leak checks are, not just asked for in the prompt.
- Every ia_ee_help turn writes exactly one entry to a real, append-only, per-project disclosure log, whether coaching was given or a request was refused. A real formatter renders the log into the actual AI-use disclosure statement text a student can include with their submission.

Phase 11, Planner, replaces one specific sequential bottleneck with a real concurrent stage graph rather than rewriting the whole orchestrator:

- A real dependency-graph executor: given named async stages with declared dependencies, it runs every stage whose dependencies are already satisfied concurrently via `asyncio.gather`, repeating until the whole graph has executed. A genuine configuration bug (a cycle, or a dependency name that doesn't exist) raises immediately instead of hanging forever; one stage failing is isolated and recorded, not allowed to cancel or block its independent siblings.
- Applied to the one place in the orchestrator with real, independent side effects: after a check_work submission grades, Phase 5's mastery write, Phase 8's misconception diagnosis, and Phase 9's review record all key only off the completed grading, never off each other, so they now run as one concurrent stage batch instead of three sequential awaits.
- The real plan that executed (stage names, dependencies, per-stage timing, any isolated error) is stored on `Blackboard.execution_plan`, a field that had been an unpopulated stub since Phase 1.
- Deliberately not a rewrite of the rest of the turn's sequencing: intent classification, the decision policy, grounding, and generation each genuinely depend on the previous stage's output, so representing them as a "parallel" graph would mean inventing concurrency that doesn't actually exist.

Phases 12 onward extend specific real capabilities identified in earlier "not built yet" notes, verified by their own test suites rather than growing the single scripted demo indefinitely:

Phase 12, expression parsing, extends the CAS layer to real IB AA HL syllabus content Phase 2 didn't originally cover:

- Definite integrals with limits: "integrate x**2 from 0 to 2" computes a real bounded value via SymPy's own (variable, lower, upper) integration form, not the indefinite "+ C" result. Whole-number bounds are kept as exact integers rather than coerced to floats, so the result stays an exact fraction (`8/3`) instead of a floating-point approximation.
- Two real matrix operations (determinant, matrix multiplication), parsing a real `[[1,2],[3,4]]` literal syntax entry-by-entry through the same restricted expression parser every other CAS input uses (never a raw `eval` of the whole string).
- Piecewise function evaluation: given real segments like `"x**2 if x < 0; 2*x if x >= 0"`, builds an actual `sympy.Piecewise` and evaluates it at a point, correctly handling `==`/`!=` conditions (a real, easy-to-miss SymPy gotcha: SymPy overloads Python's `==`/`!=` for structural equality checks, not for building a symbolic condition, so these are special-cased rather than silently mishandled). This is exposed as a structured API rather than a free-text chat trigger, since a natural-sounding request for a specific piecewise definition doesn't reduce to a clean regex the way "integrate X from A to B" does.

Phase 13, LLM-authored item variants, adds a second real generation mode to the Question Generation Engine, never trusted on its own word:

- The LLM proposes a new problem stem plus a claimed answer; the same CAS oracle every other phase relies on independently computes the real answer and checks the claim before anything is ever served. A mismatched claim triggers one bounded regeneration attempt with the specific mismatch fed back as a correction, then gives up rather than looping or serving an unverified item.
- The served answer is always the CAS-computed value, not the LLM's own string, even when the two happen to agree (checked by a test using an equivalent-but-differently-formatted claim).
- Only tried when there's genuinely no good parametric template for the topic (`topic_has_known_template` returns False), not as a redundant, costlier alternative to a template that already covers the topic perfectly for free. Falls back to the generic default template if the LLM can't produce a verifiable item either.
- Distractor generation for an arbitrary LLM-authored stem is out of scope (Phase 3's distractor generators are template-specific); a variant item is served with no distractors rather than a fabricated one.

Phase 14, IRT recalibration, replaces a template's hand-set difficulty prior with a real value computed from actual response history once enough of it exists:

- A real, append-only per-template response log (correct/incorrect, keyed to the template that was attempted). `handle_turn`'s optional `responding_to_template_id` parameter tags a check_work grading as a response to a specific previously-served generated item, so its outcome gets logged.
- Real Rasch-model (1PL) item calibration: difficulty `b = ln((1-p)/p)` from the empirical pass rate `p`, under the standard population-mean-ability-zero simplification (the full joint ability/difficulty estimation a 2PL/3PL model would use needs response volume and infrastructure this system doesn't have). A template attempted mostly incorrectly recalibrates harder; mostly correctly, easier; exactly 50% empirically stays at the neutral prior.
- Gated on a real minimum sample size (10 responses): recalibrating off a handful of data points isn't trusted, same "don't act on too little data" posture as the Misconception Diagnostician's confidence gate.
- Wired for real: a freshly generated item for a template with enough response history reports `difficulty_estimate.source="recalibrated"` and the real computed value, not just an informational report nobody consumes.

Phase 15, mastery-threshold review bands, replaces the Adaptive Learning Engine's flat due/not-due boolean with real urgency ranking:

- A real `ReviewUrgency` classification (mild/significant/critical) computed from an item's actual current retrievability (the same FSRS forgetting-curve formula Phase 9 already built), not a flat days-overdue count. Two items overdue by the same number of days can be at genuinely different forgetting risk if their stabilities differ, and this reflects that directly.
- `get_due_subtopics` now ranks by real ascending retrievability (most likely already forgotten first) rather than raw due date, a strict improvement that happens to preserve every existing test's expected ordering (verified, not assumed) since a lower-stability item overdue by the same margin is both earlier-due and lower-retrievability in every case this system's formulas produce.
- `get_due_subtopics_with_urgency` exposes the full (state, band, retrievability) breakdown for any caller that wants it, not just the single top pick `most_overdue_subtopic` already returned.

Not built yet: real curriculum/template content beyond the seed set, the full spec §8.2 misconception catalog (this build covers four named errors with real detection behind them, not the whole syllabus), per-node solution-graph diagnosis for errors that aren't a clean pattern match, the published FSRS algorithm's per-user ML-fitted weights (fixed illustrative parameters are used instead, see Phase 9 above), per-IB-assessment-criterion rubric feedback for IA/EE coaching (the coaching is real and bounded, but doesn't score against the actual IB criteria A-E), an output-side ghostwriting guard beyond the word cap (the input-side request guard and the word cap are both real; a model that ignored its constraints and wrote a short-but-complete paragraph anyway isn't separately detected), a full turn-wide Planner-produced stage graph (Phase 11's executor is real and general-purpose, but it's applied to the one place in the current turn with genuine independent side effects, not to every stage - most of a turn really is sequential, and forcing a parallel graph onto genuinely dependent stages would be simulated parallelism, not real), multi-page PDF or HEIC image intake (PNG/JPEG only), a specialized math-OCR service (a general vision-capable model is used instead, the documented fallback per spec), dense/vector retrieval, a reranker, grade-boundary calibration from historical exam data, human-in-the-loop review/appeals, memory consolidation batch jobs, GDPR export/erasure workflows, self-consistency multi-sampling (the CAS/mark-scheme verification this system already has is a stronger deterministic guarantee for the claims that matter, so this was a considered tradeoff, not an oversight), offline eval harness, online guardrail metrics, regression gating, shadow evaluation. These are stubbed with TODOs pointing at the relevant spec section.

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
  memory/                   BKT/IRT mastery, decay, node states, misconception registry, context assembly
  verifier/                 Verifier/Critic checklist agent, grounding entailment check
  multimodal/               image intake validation, PIL preprocessing, math_ocr call, LaTeX normalization, confidence gating
  diagnostician/            misconception catalog, SymPy pattern detectors, model-inference fallback, write policy
  adaptive/                 FSRS-lite scheduling math, review state store, due-review queue
  ia_supervisor/            ghostwriting guard, stage classifier + state machine, disclosure log + statement
  planner/                  real dependency-graph stage executor (asyncio.gather over what's actually ready)
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

Run the tests (450 tests and growing as phases 12+ land, all mocked at the model boundary, no API key or network needed; SymPy, retrieval, item generation, grading, memory math, the grounding check, the multimodal preprocessing/normalization/confidence math, the misconception pattern detectors, the FSRS scheduling math, the IA/EE guard/stage classifier/disclosure formatter, the Planner's dependency-graph executor (including real wall-clock timing assertions proving concurrent, not sequential, execution), and the definite-integral/matrix/piecewise CAS operations all run for real):

```bash
pytest -q
```

Run the scripted demo conversation (shows real mastery climb from three gradings driving a later CHALLENGE decision with no override, a `critique:` line per turn confirming the Verifier/Critic and grounding check ran, an `ingestion:` line for a photographed submission showing the real intake/preprocessing/confidence pipeline running before grading, a `diagnosis:` line for a wrong submission showing the misconception it was pattern-matched to, which then appears in the very next turn's `memory:` line, an exam_prep turn whose `reason=exam_prep_review_due` and generated item come from a real, pre-seeded-overdue FSRS schedule, two ia_ee_help turns (real coaching allowed, then a ghostwriting request refused) with a real disclosure statement rendered from both at the end, and a `plan:` line per graded turn showing the real concurrent post-grading stage execution and its actual measured timing):

```bash
python scripts/run_scripted_conversation.py
```

Run the gateway (optional, needs ANTHROPIC_API_KEY to actually generate):

```bash
uvicorn app.main:app --reload
```
