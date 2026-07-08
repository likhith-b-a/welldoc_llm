# QueryDocs AI — Hybrid RAG System (Local LLM)

A  Retrieval-Augmented Generation (RAG) AI assistant built using local LLMs, semantic search, hybrid retrieval, and explainable answer generation.

This system allows users to ask natural language questions and receive accurate, source-grounded answers strictly based on provided Welldoc PDF documents.

---

## 🚀 Features

- Fully Local AI 
- Semantic search using vector embeddings
- Hybrid search (Semantic + Keyword ranking)
- Semantic chunking with overlap for better retrieval
- Re-ranking layer for improved answer precision
- Source attribution (document + page number)
- Streamlit GUI for interactive querying
- Persistent vector database using ChromaDB

---

## System Architecture

```
PDF Documents
      ↓
Extraction (Page-Level)
      ↓
Semantic Chunking + Overlap
      ↓
Ollama Embeddings
      ↓
ChromaDB (Persistent Vector Store)
      ↓
Hybrid Retrieval Layer
      ↓
Re-ranking Mechanism
      ↓
LLM Answer Generation (llama2)
      ↓
Streamlit GUI
```

---

## 📂 Project Structure

```
WELLDOC_RAG/
│
├── FAQ/                       # Input PDF documents
├── chroma_db/                 # Persistent vector database
│
├── extraction.py              # PDF extraction logic
├── chunking.py                # Semantic chunking with overlap
├── embedding.py               # Embedding + storage logic
├── build_vector_store.py      # Vector database builder
│
├── rag_query.py               # RAG pipeline (retrieval + LLM)
├── app.py                     # Streamlit GUI
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/likhith-b-a/welldoc_llm.git
cd WELLDOC_RAG
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install chromadb
pip install ollama
pip install pypdf
pip install sentence-transformers
pip install numpy
pip install streamlit
```

---

## Setup Ollama (Local Models)

Install Ollama:

https://ollama.com

Pull required models:

```bash
ollama pull llama2:7b
ollama pull nomic-embed-text
```

---

## 📚 Build Vector Database

Place your PDF documents inside:

```
FAQ/
```

Then run:

```bash
python build_vector_store.py
```

This step will:

- Extract PDF text
- Perform semantic chunking
- Generate embeddings
- Store vectors in persistent ChromaDB

---

## 🖥 Run GUI

```bash
streamlit run app.py
```

Open browser and ask questions.

---

##  Retrieval Pipeline

The system uses hybrid search:

###  Semantic Search

Vector similarity using embeddings generated via Ollama.

###  Keyword Boosting

Exact keyword matching improves retrieval precision.

###  Re-ranking Layer

Refines candidate chunks before sending to LLM.

---

## 🤖 Answer Generation

The LLM:

- Uses retrieved context
- Returns concise grounded answers
- Displays source references



```
Answers strictly based on provided documents,
If answer not found, response is:
    I could not find this in the provided documents.
```

---

## 🧰 Tech Stack

- Python
- Ollama (LLM + embeddings)
- ChromaDB (Vector Database)
- SentenceTransformers
- Streamlit

## 🖼 Demo Screenshots

### 💬 Example Query and Answer
![Query Example](assets/query_result1.png)
![Query Example](assets/query_result2.png)


