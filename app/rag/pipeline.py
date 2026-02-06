from app.rag.loader import load_pdf
from app.rag.chunking import semantic_chunking
from app.rag.vector_store import store_chunks
import json
import os
def save_chunks_debug(chunks):

    os.makedirs("data/processed_chunks", exist_ok=True)

    with open("data/processed_chunks/chunks_debug.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

def build_pipeline(pdf_path):

    docs = load_pdf(pdf_path)

    chunks = semantic_chunking(docs)
    save_chunks_debug(chunks)
    store_chunks(chunks)

    return {
        "documents": len(docs),
        "nb_chunks": len(chunks),
        "chunks": chunks[:10]
    }
    