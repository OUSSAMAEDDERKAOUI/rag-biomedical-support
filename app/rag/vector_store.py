import chromadb
from langchain.vectorstores import Chroma
from app.rag.embeddings import get_embeddings
import os

def store_chunks(chunks):

    client = chromadb.HttpClient(host="chroma", port=8000)

    vectorstore = Chroma(
        client=client,
        collection_name="biomedical",
        embedding_function=get_embeddings()
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas
    )

    return vectorstore
