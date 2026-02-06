import chromadb
from langchain_community.vectorstores import Chroma
from app.rag.embeddings import get_embeddings
import os

def store_chunks(chunks):


    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST")
    )


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
    data = vectorstore._collection.get(limit=10)
    print(data)

    return vectorstore
