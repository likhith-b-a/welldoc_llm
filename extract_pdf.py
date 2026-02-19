import os
from pypdf import PdfReader

def extract_pages(folder):

    pages = []

    for filename in os.listdir(folder):

        if not filename.endswith(".pdf"):
            continue

        reader = PdfReader(os.path.join(folder, filename))

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if text and len(text.strip()) > 50:
                pages.append({
                    "file": filename,
                    "page": page_num + 1,
                    "text": text
                })

    return pages
