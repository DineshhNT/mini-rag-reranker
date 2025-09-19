import sqlite3, faiss, numpy as np, pickle
from sentence_transformers import SentenceTransformer

DB_PATH = "data/chunks.db"
INDEX_PATH = "data/faiss.index"
EMB_PATH = "data/embeddings.pkl"

def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    conn = sqlite3.connect(DB_PATH)
    chunks = [row[0] for row in conn.execute("SELECT chunk FROM chunks")]
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    pickle.dump(chunks, open(EMB_PATH, "wb"))
    conn.close()

if __name__ == "__main__":
    build_index()
    print("FAISS index built.")
