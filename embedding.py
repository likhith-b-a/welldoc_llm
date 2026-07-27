import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="welldoc_docs"
)

# Load HuggingFace model
model = SentenceTransformer('all-MiniLM-L6-v2')

def store_chunks(chunks):

    for i, chunk in enumerate(chunks):

        embedding = model.encode(chunk["text"]).tolist()

        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "file": chunk["file"],
                "page": chunk["page"]
            }]
        )
