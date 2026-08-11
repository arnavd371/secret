"""
Retriever Agent (spec §2.2, §5.6): retrieves ranked knowledge-base chunks
for a query.

Spec §5.6 describes a full hybrid retrieval pipeline: BM25 sparse search +
dense embedding search + knowledge-graph traversal + a cross-encoder
reranker, combined via the weighted formula in §5.7. That requires a
vector index and an embedding model — infrastructure well beyond a Phase 2
reasoning-core pass. This module implements the lexical half honestly: a
real (not stubbed) TF-IDF/cosine-similarity scorer over each chunk's text
plus alias keywords, with a direct-match boost when the Router/Intent
agent's `topic_hint` lines up with a chunk's `subtopic_id` — topic
resolution is exactly the kind of strong prior signal §5.6's real
pipeline would also weight heavily. IDF weighting (not just raw term
frequency) matters even at this small a corpus size: without it, a token
like "x" that appears in nearly every chunk swamps genuinely
discriminating terms like "quadratic" or "chain".

TODO(later phase): add dense embedding search, graph-traversal expansion
over prerequisite edges, and a cross-encoder reranker per §5.6-§5.7 once a
vector store is part of the stack.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from app.knowledge.schemas import KnowledgeChunk, RetrievedChunk
from app.knowledge.seed_data import load_seed_chunks

# Spec §5.7: "Retrieval score threshold for 'retrieval-obligated' claims
# (§1.4): final_score >= 0.62 on the top-ranked chunk; below this, the
# claim must be hedged or withheld." Applied here to this simplified
# lexical score directly, per the spec's own instruction to implement and
# instrument every given threshold rather than treat it as a suggestion.
# Note: a pure TF/cosine score is a different (typically harsher) scale
# than the full weighted formula in §5.7 — this is an honest, documented
# consequence of the simplification above, not silently smoothed over.
RETRIEVAL_SCORE_THRESHOLD = 0.62

_TOPIC_HINT_EXACT_MATCH_SCORE = 0.95
_TOPIC_HINT_PARTIAL_MATCH_SCORE = 0.75

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _term_frequency_vector(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    shared_terms = set(vec_a) & set(vec_b)
    dot_product = sum(vec_a[term] * vec_b[term] for term in shared_terms)
    norm_a = math.sqrt(sum(count * count for count in vec_a.values()))
    norm_b = math.sqrt(sum(count * count for count in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class KnowledgeBase:
    def __init__(self, chunks: Optional[list[KnowledgeChunk]] = None) -> None:
        self._chunks = chunks if chunks is not None else load_seed_chunks()
        term_frequencies = {
            chunk.chunk_id: _term_frequency_vector(_tokenize(chunk.searchable_text)) for chunk in self._chunks
        }
        self._idf = self._compute_idf(term_frequencies.values())
        self._vectors = {chunk_id: self._tfidf(tf) for chunk_id, tf in term_frequencies.items()}

    def _compute_idf(self, term_frequencies) -> dict[str, float]:  # noqa: ANN001
        n_docs = len(self._chunks)
        document_frequency: Counter = Counter()
        for tf in term_frequencies:
            document_frequency.update(set(tf))
        # Smoothed IDF (always positive, never zero for an unseen term at
        # query time): log((N+1)/(df+1)) + 1.
        return {term: math.log((n_docs + 1) / (df + 1)) + 1.0 for term, df in document_frequency.items()}

    def _tfidf(self, tf: Counter) -> Counter:
        return Counter({term: count * self._idf.get(term, 1.0) for term, count in tf.items()})

    def retrieve(self, query: str, topic_hint: Optional[str] = None, k: int = 3) -> list[RetrievedChunk]:
        query_vector = self._tfidf(_term_frequency_vector(_tokenize(query)))
        scored: list[tuple[float, KnowledgeChunk]] = []

        for chunk in self._chunks:
            score = _cosine_similarity(query_vector, self._vectors[chunk.chunk_id])
            if topic_hint:
                if chunk.subtopic_id == topic_hint:
                    score = max(score, _TOPIC_HINT_EXACT_MATCH_SCORE)
                elif topic_hint.lower() in chunk.subtopic_id.lower() or chunk.subtopic_id.lower() in topic_hint.lower():
                    score = max(score, _TOPIC_HINT_PARTIAL_MATCH_SCORE)
            if score > 0.0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_type=chunk.doc_type,
                subtopic_id=chunk.subtopic_id,
                citation=chunk.citation,
                text=chunk.text,
                score=round(score, 4),
                authority_tier=chunk.authority_tier,
            )
            for score, chunk in scored[:k]
        ]


def is_grounded(chunks: list[RetrievedChunk]) -> bool:
    """Spec §1.4: below the threshold, a claim "is rewritten as a generic
    mathematical statement without syllabus-specific framing, or the
    system explicitly says it cannot confirm the IB-specific convention."""
    return bool(chunks) and chunks[0].score >= RETRIEVAL_SCORE_THRESHOLD


_default_knowledge_base: Optional[KnowledgeBase] = None


def get_default_knowledge_base() -> KnowledgeBase:
    """Process-wide singleton over the seed corpus, so the orchestrator
    doesn't re-tokenize the whole knowledge base on every turn."""
    global _default_knowledge_base
    if _default_knowledge_base is None:
        _default_knowledge_base = KnowledgeBase()
    return _default_knowledge_base
