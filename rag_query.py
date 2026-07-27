import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="welldoc_docs")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def keyword_score(query, document):

    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())

    overlap = query_words.intersection(doc_words)

    return len(overlap)


def rerank_chunks(query, documents, metadatas):

    scored_chunks = []

    model = genai.GenerativeModel('gemini-1.5-flash')

    for i in range(len(documents)):

        scoring_prompt = f"""
You are a relevance scoring system.

Query:
{query}

Document:
{documents[i]}

On a scale of 0 to 10, how relevant is this document to the query?
Respond with ONLY a number.
"""

        response = model.generate_content(scoring_prompt)

        score_text = response.text.strip()

        try:
            score = float(score_text)
        except:
            score = 0

        scored_chunks.append({
            "score": score,
            "document": documents[i],
            "metadata": metadatas[i]
        })

    # sort by score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks


def ask_question(query):

    print(f"\nUser Query: {query}")

    # Step 1 — Embed query
    query_embedding = embed_model.encode(query).tolist()

    # Step 2 — Retrieve semantic candidates
    results = collection.query(
          query_embeddings=[query_embedding],
          n_results=10
     )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    hybrid_results = []

    for i in range(len(documents)):

          semantic_score = 1 / (1 + distances[i])  # normalize

          keyword_match = keyword_score(query, documents[i])

          hybrid_score = semantic_score + (0.1 * keyword_match)

          hybrid_results.append({
               "score": hybrid_score,
               "document": documents[i],
               "metadata": metadatas[i]
          })

     # sort by combined score
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)

    top_chunks = hybrid_results[:3]


    # Step 3 — Re-rank
    reranked = rerank_chunks(query, documents, metadatas)

    # take top 3 after reranking
    top_chunks = reranked[:3]

    context = ""
    sources = []

    for item in top_chunks:
        context += "\n" + item["document"] + "\n"
        sources.append(
            f"{item['metadata']['file']} (Page {item['metadata']['page']})"
        )

    # Step 4 — Final Answer Generation
    prompt = f"""
You are a Welldoc document assistant.

Answer the question DIRECTLY and specifically.
Do not include unrelated procedural steps unless explicitly asked.

If question asks about eligibility, provide only eligibility criteria.

Use ONLY provided context.
If answer not found, say:
"I could not find this in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    print("\n===== ANSWER =====\n")
    print(response.text)

    print("\n===== SOURCES =====\n")
    for src in set(sources):
        print(src)

# Test
ask_question("How do I reset my password?")
ask_question("What is the refund policy?")
ask_question("Insulin Adjustment Program eligibility")
ask_question("who is mohan")
