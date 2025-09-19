# Mini RAG Reranker

## Overview
**Mini RAG Reranker** is a Python-based application designed to demonstrate real-world use of AI-powered search and ranking. The project focuses on **retrieving, embedding, and ranking textual data** using modern NLP techniques, including transformer models and vector embeddings. This project was completed as part of a technical assessment for a Python developer role.

The system can:
- Ingest and process search data.
- Build an embedding-based index for efficient retrieval.
- Rerank search results using semantic similarity models.

---

## Features
- **Data Ingestion**: Supports adding structured text data for search and indexing.
- **Embedding Generation**: Uses `sentence-transformers` to convert text into vector representations.
- **Reranking**: Improves search relevance using semantic similarity.
- **Python Ecosystem**: Built with `pandas`, `numpy`, `scikit-learn`, `scipy`, and `sentence-transformers`.
- **Extensible**: Can be integrated with larger AI/ML search applications.

---

## Tech Stack
- **Python 3.10**
- **Libraries**: 
  - `pandas`, `numpy`, `scikit-learn`, `scipy`
  - `sentence-transformers` (for embeddings)
  - `transformers` (for NLP models)
  - `matplotlib` (for visualization)
- **Tools**: Git, GitHub

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<USERNAME>/mini-rag-reranker.git
cd mini-rag-reranker
````

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/Scripts/activate   # Windows
# or
source venv/bin/activate       # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python ingest/build_index.py
```

---

## Project Structure

```
mini-rag-reranker/
│
├── ingest/                 # Scripts for data ingestion and indexing
│   └── build_index.py
│
├── search/                 # Scripts for performing search and reranking
│   └── search_app.py
│
├── data/                   # Sample data files for indexing
│
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
```

---

## Usage

1. Place your dataset in the `data/` folder.
2. Run the indexing script:

```bash
python ingest/build_index.py
```

3. Use the search script to query the data:

```bash
python search/search_app.py
```

4. Results are reranked based on semantic similarity scores.

---

## Key Learnings / Highlights

* Implemented **embedding-based search** and semantic reranking using Python.
* Integrated multiple **Python libraries** for data processing, ML modeling, and evaluation.
* Solved **compatibility issues** with `numpy`, `scipy`, `tensorflow`, and `sentence-transformers`.
* Developed a **clean, modular, and production-ready code structure**.

---

## References

* [Sentence Transformers](https://www.sbert.net/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* Python official documentation: [https://docs.python.org/3/](https://docs.python.org/3/)


---

## License

This project is licensed under the MIT License.

```

---

If you want, I can also **create a `requirements.txt`** that matches all the exact working versions from your setup (`numpy`, `tensorflow`, `keras`, `sentence-transformers`, etc.), so anyone can run it without dependency errors.  

Do you want me to do that next?
```
