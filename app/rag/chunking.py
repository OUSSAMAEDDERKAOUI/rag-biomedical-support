from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

def semantic_chunking(docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    splitter = SemanticChunker(embeddings)

    chunks = []

    for d in docs:
        split = splitter.split_text(d["text"])

        for s in split:
            chunks.append({
                "text": s,
                "metadata": d["metadata"]
            })

    return chunks
