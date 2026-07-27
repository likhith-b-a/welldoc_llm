# QueryDocs AI — Hybrid RAG System

A Retrieval-Augmented Generation (RAG) AI assistant built using semantic search, hybrid retrieval, and explainable answer generation.

This system allows users to ask natural language questions and receive accurate, source-grounded answers strictly based on the provided PDF documents — either a bundled demo FAQ or documents you upload yourself (resumes excluded — see [Scope](#scope) below).

---

## 🚀 Features

- Cloud-deployable (no local LLM server required)
- Two modes: a bundled **Demo FAQ** and **My documents**, where you upload your own PDFs and get answers grounded strictly in what you uploaded
- Uploaded documents are session-scoped, in-memory only, and never written to disk or shared between users
- Conversational chat interface with per-mode history and collapsible source citations
- Semantic search using vector embeddings
- Hybrid search (semantic + keyword ranking, with stopwords filtered out of the keyword signal)
- Structure-aware chunking: handles both prose (FAQs, legal text) and bullet-heavy sections (job descriptions, requirement lists) instead of collapsing a whole page into one chunk
- Answers can synthesize information across multiple retrieved chunks, not just literal single-sentence matches
- Source attribution (document + page number)
- Native Streamlit theming (no custom CSS/HTML) with a dark, violet-accented theme
- Persistent vector database using ChromaDB (demo mode) / ephemeral in-memory store (upload mode)

---

## Scope

This app is tuned for **prose and semi-structured documents** — FAQs, policies, legal text, and job descriptions. It is **not optimized for resumes**: resumes tend to be dense, purely bulleted documents with little connecting prose, which the current chunking strategy doesn't handle well. Job descriptions work because they mix prose (role/company summary) with bullets (responsibilities, requirements), and the chunker's structure-aware fallback splits those bullets into individual chunks.

---

## System Architecture

```
PDF Documents
      ↓
Extraction (Page-Level)
      ↓
Structure-Aware Chunking (prose sentences + bulleted/line-based fallback)
      ↓
Local Embeddings (Hugging Face sentence-transformers)
      ↓
ChromaDB (Persistent Vector Store / Ephemeral Session Store)
      ↓
Hybrid Retrieval Layer (semantic + stopword-filtered keyword scoring)
      ↓
LLM Answer Generation (Google Gemini API, context-synthesizing prompt)
      ↓
Streamlit Chat UI
```

Embeddings run locally in-process via `sentence-transformers` (`all-MiniLM-L6-v2`) — no API key or external service needed for that step. Answer generation uses the **Google Gemini API** (`gemini-2.5-flash`), which is free-tier friendly and requires a `GEMINI_API_KEY`.

### Chunking

`chunking.py` primarily splits text into sentences on periods, which works well for prose. If a period-delimited segment comes out unusually long (a sign the source has few or no periods — bullet lists of responsibilities/requirements, numbered clauses, headers), it falls back to splitting that segment by line instead. This keeps prose documents chunked by sentence while still giving bullet-heavy sections (like a job description's requirements list) one chunk per bullet rather than one giant blob per page.

### Retrieval

For each question, up to 30 candidate chunks are retrieved by embedding similarity, then re-scored with a hybrid score (semantic similarity + a keyword-overlap boost that ignores common stopwords). The top 6 chunks are passed to Gemini as context. The prompt explicitly allows the model to combine and summarize information across those chunks — so a question like "is this candidate a good fit given 2 years of experience" can be answered by reasoning over the requirements section, not just quoting a single matching sentence — while still requiring the answer to be grounded in the provided context and refusing to answer if it isn't there.

---

## 📂 Project Structure

```
QueryDocs-AI/
│
├── faq/                        # Bundled demo PDF document(s)
├── chroma_db/                  # Persistent vector database for the demo (committed for Streamlit Cloud)
├── .streamlit/
│   └── config.toml             # Native theme configuration
│
├── extract_pdf.py              # PDF extraction — folder-based (build script) and file-object based (uploads)
├── chunking.py                 # Structure-aware chunking (prose sentences + bullet fallback)
├── embedding.py                # Embedding + storage logic for the demo vector store
├── build_vector_store.py       # Demo vector database builder
│
├── rag_query.py                # RAG pipeline test script (CLI)
├── app.py                      # Streamlit chat app (both Demo FAQ and My documents modes)
│
└── readme.md
```

---

## ⚙️ Local Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/likhith-b-a/QueryDocs-AI.git
cd QueryDocs-AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set your Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey), then create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

---

## 📚 Build the Demo Vector Database

Place your PDF document(s) inside `faq/`, then run:

```bash
python build_vector_store.py
```

This extracts PDF text, performs structure-aware chunking, generates embeddings, and stores vectors in `chroma_db/`. The repo already ships with a prebuilt `chroma_db/` for the included demo FAQ, so this step is only needed if you change the demo source document(s) — it does **not** affect the "My documents" upload mode, which builds its own in-memory store per session at runtime.

---

## 🖥 Run Locally

```bash
streamlit run app.py
```

Pick a source at the top of the app, then ask questions in the chat box:
- **Demo FAQ** — queries the bundled, prebuilt `chroma_db/`.
- **My documents** — upload one or more PDFs, click **Process documents**, and questions are answered strictly from what you uploaded (kept in memory for that browser session only; click **Clear my documents** to reset). Each mode keeps its own separate chat history.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (including the `chroma_db/` folder — the demo vector store must ship with the app since Streamlit Cloud has no separate build step).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo, branch `main`, main file `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your_key_here"
   ```
4. Deploy. If you later change the demo source PDF(s), rebuild `chroma_db/` locally with `python build_vector_store.py` and commit the updated folder — Streamlit Cloud won't run the build script for you. (The upload mode needs no rebuild since it's per-session and in-memory.)

---

## 🤖 Answer Generation

The LLM (Gemini):

- Retrieves and synthesizes across the top matching chunks
- Returns concise, grounded answers
- Displays source references (document + page) in a collapsible section under each answer

```
Answers strictly based on provided documents.
If answer not found, response is:
    I could not find this in the provided documents.
```

---

## 🧰 Tech Stack

- Python
- Google Gemini API (answer generation)
- Hugging Face `sentence-transformers` (local embeddings)
- ChromaDB (vector database — persistent for the demo, ephemeral in-memory for uploads)
- Streamlit (chat UI, native theming via `config.toml`)

## 🖼 Demo Screenshots

> The screenshots below are from an earlier version of the UI (pre-rebrand, pre-chat-interface) and don't reflect the current app.

![Query Example](assets/query_result1.png)
![Query Example](assets/query_result2.png)
