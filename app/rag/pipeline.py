from app.rag.loader import load_pdf
from app.rag.chunking import semantic_chunking
from app.rag.vector_store import store_chunks

def build_pipeline(pdf_path):

    docs = load_pdf(pdf_path)

    chunks = semantic_chunking(docs)

    store_chunks(chunks)

    return {
        "documents": len(docs),
        "chunks": len(chunks)
    }
