from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from app.rag.retriever import get_retriever

def get_qa_chain():

    llm = Ollama(model="mistral")   

    retriever = get_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain
