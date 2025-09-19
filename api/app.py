from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import pickle
import faiss
import numpy as np
import os
from search.baseline_search import vector_search
from search.reranker_hybrid import hybrid_rerank
from sentence_transformers import SentenceTransformer

# ---------- Load Index & Model ----------
DB_PATH = "data/chunks.db"
EMB_PATH = "data/embeddings.pkl"
FAISS_PATH = "data/faiss.index"

# Load embeddings
with open(EMB_PATH, "rb") as f:
    embeddings_data = pickle.load(f)
    embeddings = np.array(embeddings_data["embeddings"], dtype="float32")
    ids = embeddings_data["ids"]

# Load FAISS index
index = faiss.read_index(FAISS_PATH)

# Model for encoding queries
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Mini RAG + Reranker API")

class AskRequest(BaseModel):
    q: str
    k: int = 5
    mode: Optional[str] = "baseline"   # 'baseline' or 'hybrid'

class Context(BaseModel):
    text: str
    score: float
    source: str

class AskResponse(BaseModel):
    answer: Optional[str]
    contexts: List[Context]
    reranker_used: str

def fetch_chunks(id_list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(id_list))
    cursor.execute(f"SELECT id, text, source FROM chunks WHERE id IN ({placeholders})", id_list)
    rows = cursor.fetchall()
    conn.close()
    # Map id -> (text, source)
    return {r[0]: (r[1], r[2]) for r in rows}

def build_answer(chunks):
    # naive extractive answer = top chunk text, truncated
    if not chunks:
        return None
    best = chunks[0]
    text = best["text"].strip().replace("\n", " ")
    return text[:500]  # short summary

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    query = req.q
    k = req.k

    q_emb = model.encode([query])[0]
    top_ids, vec_scores = vector_search(index, q_emb, k=30)  # fetch more for reranker

    chunk_map = fetch_chunks(top_ids)
    results = [{"id": i, "text": chunk_map[i][0], "source": chunk_map[i][1], "vec_score": s}
               for i, s in zip(top_ids, vec_scores) if i in chunk_map]

    if req.mode == "hybrid":
        results = hybrid_rerank(query, results)[:k]
        used = "hybrid"
    else:
        results = sorted(results, key=lambda x: x["vec_score"], reverse=True)[:k]
        used = "baseline"

    answer = build_answer(results)
    contexts = [Context(text=r["text"], score=float(r["vec_score"])_]()]()
