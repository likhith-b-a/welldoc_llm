from extract_pdf import extract_pages
from chunking import semantic_chunk
from embedding import store_chunks

FAQ_FOLDER = "FAQ"

print("Extracting...")
pages = extract_pages(FAQ_FOLDER)

print("Chunking...")
chunks = semantic_chunk(pages)

print("Embedding and storing...")
store_chunks(chunks)

print("Vector DB built successfully!")
