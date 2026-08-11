from app.knowledge.retriever import RETRIEVAL_SCORE_THRESHOLD, KnowledgeBase, is_grounded


def test_topic_hint_exact_match_is_grounded():
    kb = KnowledgeBase()
    chunks = kb.retrieve("help me with this problem", topic_hint="calculus.differentiation.chain_rule")
    assert chunks
    assert chunks[0].subtopic_id == "calculus.differentiation.chain_rule"
    assert chunks[0].score >= RETRIEVAL_SCORE_THRESHOLD
    assert is_grounded(chunks)


def test_strong_keyword_overlap_ranks_relevant_chunk_first():
    kb = KnowledgeBase()
    chunks = kb.retrieve("what is the quadratic formula and how do I use the discriminant")
    assert chunks
    assert chunks[0].subtopic_id == "algebra.quadratics.solving"


def test_unrelated_query_is_not_grounded():
    kb = KnowledgeBase()
    chunks = kb.retrieve("what is the capital of France?")
    assert not is_grounded(chunks)


def test_retrieve_respects_k_limit():
    kb = KnowledgeBase()
    chunks = kb.retrieve("differentiate a function", k=2)
    assert len(chunks) <= 2


def test_results_are_sorted_by_score_descending():
    kb = KnowledgeBase()
    chunks = kb.retrieve("trig identity compound angle sin", k=5)
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)


def test_zero_score_chunks_are_excluded():
    kb = KnowledgeBase()
    chunks = kb.retrieve("zzz_no_overlap_at_all_qqqxyz")
    assert chunks == []


def test_idf_weighting_does_not_let_a_common_token_dominate_ranking():
    """Regression: 'x' appears in nearly every chunk. A query that's
    actually about differentiating cos(x) should not rank the unrelated
    quadratic-formula chunk above (or within grounding range of) the
    directly relevant trig-derivative chunk just because both mention x."""
    kb = KnowledgeBase()
    chunks = kb.retrieve("differentiate x cos x", k=5)
    assert chunks
    assert chunks[0].subtopic_id == "calculus.differentiation.trig_functions"
    quadratic_chunk = next((c for c in chunks if c.subtopic_id == "algebra.quadratics.solving"), None)
    if quadratic_chunk is not None:
        assert quadratic_chunk.score < RETRIEVAL_SCORE_THRESHOLD
