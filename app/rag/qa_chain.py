# from langchain.chains import RetrievalQA
# from langchain_community.llms import Ollama
# from app.rag.retriever import get_retriever

# def get_qa_chain():

#     llm = Ollama(model="mistral")   

#     retriever = get_retriever()

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff"
#     )

#     return qa_chain
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from app.rag.retriever import get_retriever

def get_qa_chain():

    llm = Ollama(
      model="mistral",
      temperature=0.3,
      top_k=10,
      top_p=0.9
     )


    retriever = get_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain
