import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.65
WINDOW_SIZE = 4
OVERLAP_SIZE = 1

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_chunk(pages):

    chunks = []

    for page in pages:

        sentences = [s.strip() for s in page["text"].split(".") if len(s.strip()) > 20]

        if len(sentences) == 0:
            continue

        embeddings = model.encode(sentences)

        groups = []
        current = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[i-1])

            if sim > SIMILARITY_THRESHOLD:
                current.append(sentences[i])
            else:
                groups.append(current)
                current = [sentences[i]]

        groups.append(current)

        for group in groups:

            start = 0

            while start < len(group):

                chunk_text = ". ".join(group[start:start+WINDOW_SIZE])

                chunks.append({
                    "file": page["file"],
                    "page": page["page"],
                    "text": chunk_text
                })

                start += (WINDOW_SIZE - OVERLAP_SIZE)

    return chunks
