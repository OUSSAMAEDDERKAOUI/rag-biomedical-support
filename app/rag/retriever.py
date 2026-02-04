from langchain.vectorstores import Chroma
from app.rag.embeddings import get_embeddings
import chromadb

def get_retriever():

    client = chromadb.HttpClient(host="chroma", port=8000)

    vectorstore = Chroma(
        client=client,
        collection_name="biomedical",
        embedding_function=get_embeddings()
    )

    return vectorstore.as_retriever()
