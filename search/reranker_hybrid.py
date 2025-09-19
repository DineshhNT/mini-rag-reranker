import sqlite3, numpy as np
from rank_bm25 import BM25Okapi
from .baseline_search import model, index, chunks

def bm25_candidates(question, top_n=20):
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    ranked = np.argsort(scores)[::-1][:top_n]
    return [(i, scores[i]) for i in ranked]

def search_hybrid(question: str, k: int, alpha=0.6):
    # vector scores
    import faiss
    q_emb = model.encode([question])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    vec_scores = {i: float(s) for s, i in zip(D[0], I[0])}

    # keyword scores
    kw_scores = dict(bm25_candidates(question, top_n=max(k*2, 20)))

    # normalize
    if vec_scores: vmax = max(vec_scores.values())
    if kw_scores: kmax = max(kw_scores.values())
    final = []
    for i in set(vec_scores) | set(kw_scores):
        v = vec_scores.get(i, 0)/ (vmax or 1)
        kscore = kw_scores.get(i, 0)/ (kmax or 1)
        score = alpha * v + (1-alpha) * kscore
        final.append((i, score))
    final = sorted(final, key=lambda x: x[1], reverse=True)[:k]
    results = [{
        "text": chunks[i],
        "source": "chunks.db",
        "score": float(s)
    } for i, s in final]
    answer = results[0]["text"][:300] if results else None
    return {
        "answer": answer,
        "contexts": results,
        "reranker_used": "hybrid"
    }
