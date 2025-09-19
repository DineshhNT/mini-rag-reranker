import os, sqlite3, textwrap
from PyPDF2 import PdfReader

DB_PATH = "data/chunks.db"

def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY,
        source TEXT,
        chunk TEXT
    )""")
    conn.commit()
    conn.close()

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, size=300):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

def ingest():
    create_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pdf_dir = "data/industrial-safety-pdfs"
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            full = os.path.join(pdf_dir, fname)
            txt = pdf_to_text(full)
            for chunk in chunk_text(txt):
                c.execute("INSERT INTO chunks(source,chunk) VALUES (?,?)",
                          (fname, chunk))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    ingest()
    print("Ingestion complete.")
