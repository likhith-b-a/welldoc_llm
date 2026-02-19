import chromadb
import ollama


client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="welldoc_docs"
)

def store_chunks(chunks):

    for i, chunk in enumerate(chunks):

        embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk["text"]
        )["embedding"]

        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "file": chunk["file"],
                "page": chunk["page"]
            }]
        )
