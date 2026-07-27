import streamlit as st
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
else:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

# ------------------------------
# Connect to ChromaDB
# ------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="welldoc_docs")

# ------------------------------
# Hybrid keyword score
# ------------------------------
def keyword_score(query, document):
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())
    return len(query_words.intersection(doc_words))

# ------------------------------
# Ask Question Function
# ------------------------------
def ask_question(query):

    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = result['embedding']

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
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
    top_chunks = hybrid_results[:3]

    context = ""
    sources = []

    for item in top_chunks:
        context += "\n" + item["document"] + "\n"
        sources.append(
            f"{item['metadata']['file']} (Page {item['metadata']['page']})"
        )

    prompt = f"""
You are a Welldoc document assistant.

Rules:
- Answer ONLY from provided context.
- If answer not found, say:
  "I could not find this in the provided documents."
- Provide clear and concise answer.

Context:
{context}

Question:
{query}

Answer:
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text, list(set(sources))



# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Welldoc AI Assistant", page_icon="🤖", layout="wide")

def inject_css():
    try:
        with open("./styles/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
inject_css()

st.markdown("<h1 class='main-header'>🤖 Welldoc FAQ AI Assistant</h1>", unsafe_allow_html=True)
st.write("Ask any question based on the uploaded Welldoc documents.")

user_query = st.text_input("Enter your question:")

if st.button("Ask"):

    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):

            answer, sources = ask_question(user_query)

        st.markdown("<h2 class='sub-header'>📌 Answer</h2>", unsafe_allow_html=True)
        st.info(answer)

        st.markdown("<h2 class='sub-header'>📚 Sources</h2>", unsafe_allow_html=True)
        for src in sources:
            st.markdown(f"<div class='card'><span class='source-text'>- {src}</span></div>", unsafe_allow_html=True)
