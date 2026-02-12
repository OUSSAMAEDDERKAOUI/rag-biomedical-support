from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    # return HuggingFaceEmbeddings(
    #     model_name="all-MiniLM-L6-v2"
    # )
    return  HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
    )

