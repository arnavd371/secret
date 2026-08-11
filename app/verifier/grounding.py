"""
Retrieval-grounding checks (spec §13.6): "claim to citation entailment" —
checks whether the sentences in a draft response are actually supported
by the retrieved chunks cited alongside it.

Real, but the same honest simplification as the Phase 2 retriever: a
lexical TF/cosine overlap score per sentence against the cited chunks,
not a trained NLI entailment model. Self-contained rather than importing
app.knowledge.retriever's helpers — this check exists specifically to be
an independent second opinion on retrieval quality, not built from the
same machinery it's meant to be checking.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from app.knowledge.schemas import RetrievedChunk

# Spec §13.6/§13.7 don't give a concrete number for this simplified
# lexical version — thresholds chosen and instrumented here, in the same
# spirit as the spec's instruction to implement and tune every given
# threshold rather than leave it unset.
SENTENCE_GROUNDING_THRESHOLD = 0.12
MAX_UNGROUNDED_SENTENCE_RATIO = 0.5

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class GroundingCheckResult(BaseModel):
    grounded: bool
    score: float
    ungrounded_sentences: list[str] = Field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _term_frequency_vector(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    shared = set(vec_a) & set(vec_b)
    dot_product = sum(vec_a[t] * vec_b[t] for t in shared)
    norm_a = math.sqrt(sum(c * c for c in vec_a.values()))
    norm_b = math.sqrt(sum(c * c for c in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def check_grounding(draft: str, retrieved_chunks: Optional[list[RetrievedChunk]]) -> GroundingCheckResult:
    """No citations to check against isn't itself a grounding failure —
    that's what the citation-attachment logic in tutor_agent.py already
    gates on. This check only evaluates whether a draft that *does* have
    retrieved context actually reflects it."""
    if not retrieved_chunks:
        return GroundingCheckResult(grounded=True, score=1.0)

    sentences = _split_sentences(draft)
    if not sentences:
        return GroundingCheckResult(grounded=True, score=1.0)

    chunk_vectors = [_term_frequency_vector(_tokenize(chunk.text)) for chunk in retrieved_chunks]

    scores: list[float] = []
    ungrounded: list[str] = []
    for sentence in sentences:
        sentence_vector = _term_frequency_vector(_tokenize(sentence))
        best_score = max((_cosine_similarity(sentence_vector, cv) for cv in chunk_vectors), default=0.0)
        scores.append(best_score)
        if best_score < SENTENCE_GROUNDING_THRESHOLD:
            ungrounded.append(sentence)

    average_score = sum(scores) / len(scores)
    grounded = (len(ungrounded) / len(sentences)) <= MAX_UNGROUNDED_SENTENCE_RATIO

    return GroundingCheckResult(grounded=grounded, score=average_score, ungrounded_sentences=ungrounded)
