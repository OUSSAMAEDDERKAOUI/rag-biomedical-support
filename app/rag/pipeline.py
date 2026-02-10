from app.rag.loader import load_pdf,group_docs_by_page
from app.rag.chunking import hybrid_chunking
from app.rag.vector_store import store_chunks,reset_chroma
import json
import os
from app.monitoring.mlflow_logger import start_rag_run, log_rag_config





def save_chunks_debug(chunks):

    os.makedirs("data/processed_chunks", exist_ok=True)

    with open("data/processed_chunks/chunks_debug.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

def build_pipeline(pdf_path):
    
    reset_chroma()
    docs = load_pdf(pdf_path)
    grouped_docs = group_docs_by_page(docs)

    # chunks = semantic_chunking(grouped_docs)
    chunks = hybrid_chunking(grouped_docs)

    save_chunks_debug(chunks)
    store_chunks(chunks)

    return {
        "documents": len(grouped_docs),
        "nb_chunks": len(chunks),
        "chunks": chunks[:10]
    }
    
    