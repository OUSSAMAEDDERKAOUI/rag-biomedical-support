from langchain_community.vectorstores import Chroma
from app.rag.embeddings import get_embeddings
import chromadb

def get_retriever():

    client = chromadb.HttpClient(host="chroma", port=8000)

    vectorstore = Chroma(
        client=client,
        collection_name="biomedical",
        embedding_function=get_embeddings()
    )

    return vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )
def get_vectorstore():
    embeddings = get_embeddings()

    client = chromadb.HttpClient(host="chroma", port=8000)

    vectorstore = Chroma(
        client=client,
        collection_name="biomedical",
        embedding_function=get_embeddings()
    )

    return vectorstore