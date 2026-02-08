from langchain_community.vectorstores import Chroma
from app.rag.embeddings import get_embeddings
import chromadb
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# def get_retriever():

#     client = chromadb.HttpClient(host="chroma", port=8000)

#     vectorstore = Chroma(
#         client=client,
#         collection_name="biomedical",
#         embedding_function=get_embeddings()
#     )

#     return vectorstore.as_retriever(
#             search_kwargs={"k": 6}
#         )

def get_vectorstore():
    embeddings = get_embeddings()

    client = chromadb.HttpClient(host="chroma", port=8000)

    vectorstore = Chroma(
        client=client,
        collection_name="biomedical",
        embedding_function=get_embeddings()
    )

    return vectorstore




def load_chunks_from_chroma():


    vectorstore = get_vectorstore()

    all_data = vectorstore.get()

    docs = []

    for text, metadata in zip(all_data["documents"], all_data["metadatas"]):
        docs.append({
            "text": text,
            "metadata": metadata
        })

    return docs






def get_retriever():


    vectorstore = get_vectorstore()

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    docs = load_chunks_from_chroma()

    texts = [d["text"] for d in docs]

    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 6

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return hybrid_retriever
