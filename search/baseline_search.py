import faiss, pickle, sqlite3, numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "data/chunks.db"
INDEX_PATH = "data/faiss.index"
EMB_PATH = "data/embeddings.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
chunks = pickle.load(open(EMB_PATH, "rb"))

def search_baseline(question: str, k: int):
    q_emb = model.encode([question])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "text": chunks[idx],
            "source": "chunks.db",
            "score": float(score)
        })
    answer = results[0]["text"][:300] if results else None
    return {
        "answer": answer,
        "contexts": results,
        "reranker_used": "baseline"
    }
