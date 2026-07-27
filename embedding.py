import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
else:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass


client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="welldoc_docs"
)

def store_chunks(chunks):

    for i, chunk in enumerate(chunks):

        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk["text"],
            task_type="retrieval_document"
        )
        embedding = result['embedding']

        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "file": chunk["file"],
                "page": chunk["page"]
            }]
        )
