import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.65
WINDOW_SIZE = 4
OVERLAP_SIZE = 1
MIN_UNIT_LENGTH = 20
MAX_UNIT_LENGTH = 250

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def split_into_units(text):
    # Period-based splitting handles prose well, but structured sections
    # (bullet lists of responsibilities/requirements, numbered clauses,
    # headers) often have few or no periods, so the whole page collapses
    # into one oversized "sentence". Any unit that comes out too long is a
    # sign of that, so fall back to splitting it by line instead.
    units = []

    for raw in text.split("."):
        candidate = raw.strip()

        if len(candidate) <= MIN_UNIT_LENGTH:
            continue

        if len(candidate) <= MAX_UNIT_LENGTH:
            units.append(candidate)
            continue

        for line in candidate.split("\n"):
            line = line.strip()
            if len(line) > MIN_UNIT_LENGTH:
                units.append(line)

    return units

def semantic_chunk(pages):

    chunks = []

    for page in pages:

        sentences = split_into_units(page["text"])

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
