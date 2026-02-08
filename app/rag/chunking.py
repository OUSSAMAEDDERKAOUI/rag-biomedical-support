# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.embeddings import HuggingFaceEmbeddings

# def semantic_chunking(docs):

#     embeddings = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )

#     splitter = SemanticChunker(embeddings)

#     chunks = []

#     for d in docs:
#         split = splitter.split_text(d["text"])

#         for s in split:
#             chunks.append({
#                 "text": s,
#                 "metadata": d["metadata"]
#             })

#     return chunks


# def semantic_chunking(docs):
#     embeddings = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )

#     splitter = SemanticChunker(embeddings)

#     chunks = []

#     for d in docs:
#         split_texts = splitter.split_text(d["text"])
#         for s in split_texts:
#             chunks.append({
#                 "text": s,
#                 "metadata": d["metadata"]
#             })

#     return chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def hybrid_chunking(docs):

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    semantic = SemanticChunker(embeddings)

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for d in docs:
        text = d["text"]

        # 1) Découpage sémantique
        semantic_splits = semantic.split_text(text)

        for part in semantic_splits:

            # 2) Redécouper si trop grand
            if len(part) > 700:
                sub = recursive.split_text(part)
            else:
                sub = [part]

            for s in sub:
                chunks.append({
                    "text": s,
                    "metadata": d["metadata"]
                })

    return chunks
