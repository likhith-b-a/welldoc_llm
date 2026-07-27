import streamlit as st
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from extract_pdf import extract_pages_from_file
from chunking import semantic_chunk

load_dotenv()
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
else:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

# Load HuggingFace model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# Connect to the prebuilt demo ChromaDB
# ------------------------------
demo_client = chromadb.PersistentClient(path="./chroma_db")
demo_collection = demo_client.get_collection(name="demo_docs")

# ------------------------------
# Hybrid keyword score
# ------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "this", "that", "these", "those", "it", "its", "as", "by", "from",
    "has", "have", "had", "not", "can", "will", "would", "should", "could",
    "do", "does", "did", "if", "than", "then", "so", "such", "i", "you",
    "he", "she", "they", "we", "my", "your", "his", "her", "their", "our",
    "what", "which", "who", "whom", "when", "where", "why", "how", "about",
}

def keyword_score(query, document):
    query_words = set(query.lower().split()) - STOPWORDS
    doc_words = set(document.lower().split()) - STOPWORDS
    return len(query_words.intersection(doc_words))

# ------------------------------
# Ask Question Function
# ------------------------------
def ask_question(query, collection):

    n_results = min(30, collection.count())
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    hybrid_results = []

    for i in range(len(documents)):

        semantic_score = 1 / (1 + distances[i])
        keyword_match = keyword_score(query, documents[i])
        hybrid_score = semantic_score + (0.1 * keyword_match)

        hybrid_results.append({
            "score": hybrid_score,
            "document": documents[i],
            "metadata": metadatas[i]
        })

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = hybrid_results[:6]

    context = ""
    sources = []

    for item in top_chunks:
        context += "\n" + item["document"] + "\n"
        sources.append(
            f"{item['metadata']['file']} (Page {item['metadata']['page']})"
        )

    prompt = f"""
You are a document assistant.

Rules:
- Base your answer only on the provided context — do not use outside knowledge.
- You may combine and summarize information from multiple parts of the context to answer the question, even if no single sentence states the answer directly.
- If the context genuinely does not contain enough information to answer, say:
  "I could not find this in the provided documents."
- Provide a clear and concise answer.

Context:
{context}

Question:
{query}

Answer:
"""

    model = genai.GenerativeModel('gemini-flash-latest')
    response = model.generate_content(prompt)

    return response.text, list(set(sources))


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="QueryDocs AI", page_icon="📄", layout="wide")

st.title(":material/description: QueryDocs AI")
st.caption("Ask questions grounded strictly in your documents.")

MODES = ["Demo FAQ", "My documents"]
MODE_ICONS = {"Demo FAQ": ":material/quiz:", "My documents": ":material/upload_file:"}
HISTORY_KEYS = {"Demo FAQ": "demo_chat", "My documents": "user_chat"}

mode = st.segmented_control(
    "Source",
    MODES,
    format_func=lambda m: f"{MODE_ICONS[m]} {m}",
    default="Demo FAQ",
    required=True,
    label_visibility="collapsed",
)

history_key = HISTORY_KEYS[mode]
active_collection = None

if mode == "Demo FAQ":
    active_collection = demo_collection
    st.caption("Answers are grounded in the bundled sample FAQ document.")

else:
    if "user_client" not in st.session_state:
        st.session_state.user_client = chromadb.EphemeralClient()
        st.session_state.user_collection = st.session_state.user_client.get_or_create_collection(name="user_docs")
        st.session_state.user_doc_names = set()
        st.session_state.user_chunk_id = 0

    with st.container(border=True):
        st.caption("Uploaded documents stay in memory for this browser session only, are never written to disk, and are not visible to other users.")

        uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

        with st.container(horizontal=True):
            process_clicked = st.button("Process documents", icon=":material/bolt:", type="primary")
            clear_clicked = st.button("Clear my documents", icon=":material/delete:")

        if clear_clicked:
            st.session_state.user_client = chromadb.EphemeralClient()
            st.session_state.user_collection = st.session_state.user_client.get_or_create_collection(name="user_docs")
            st.session_state.user_doc_names = set()
            st.session_state.user_chunk_id = 0
            st.session_state[history_key] = []
            st.success("Cleared. Upload new documents to start again.", icon=":material/check_circle:")

        if process_clicked:
            new_files = [f for f in (uploaded_files or []) if f.name not in st.session_state.user_doc_names]

            if not new_files:
                st.warning("No new PDFs to process. Upload a file first — already-processed files are skipped.", icon=":material/warning:")
            else:
                with st.spinner(f"Processing {len(new_files)} document(s)..."):

                    pages = []
                    for f in new_files:
                        pages.extend(extract_pages_from_file(f, f.name))
                        st.session_state.user_doc_names.add(f.name)

                    chunks = semantic_chunk(pages)

                    for chunk in chunks:
                        embedding = embed_model.encode(chunk["text"]).tolist()
                        st.session_state.user_collection.add(
                            ids=[str(st.session_state.user_chunk_id)],
                            documents=[chunk["text"]],
                            embeddings=[embedding],
                            metadatas=[{"file": chunk["file"], "page": chunk["page"]}]
                        )
                        st.session_state.user_chunk_id += 1

                st.success(f"Processed {len(new_files)} document(s) — {len(chunks)} chunks indexed.", icon=":material/check_circle:")

        if st.session_state.user_doc_names:
            with st.container(horizontal=True):
                for name in sorted(st.session_state.user_doc_names):
                    st.badge(name, icon=":material/description:")

    if st.session_state.user_collection.count() > 0:
        active_collection = st.session_state.user_collection
    else:
        st.info("Upload and process at least one PDF to start asking questions.", icon=":material/info:")

# ------------------------------
# Chat
# ------------------------------

if history_key not in st.session_state:
    st.session_state[history_key] = []

for msg in st.session_state[history_key]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", icon=":material/description:"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

prompt = st.chat_input(
    "Ask a question about your documents",
    disabled=active_collection is None,
    key=f"chat_input_{history_key}",
)

if prompt:
    st.session_state[history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_question(prompt, active_collection)
        st.write(answer)
        if sources:
            with st.expander("Sources", icon=":material/description:"):
                for src in sources:
                    st.write(f"- {src}")

    st.session_state[history_key].append({"role": "assistant", "content": answer, "sources": sources})
