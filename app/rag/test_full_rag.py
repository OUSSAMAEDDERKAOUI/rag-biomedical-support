# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

# from app.rag.retriever import get_retriever
# import os
# from langchain_community.llms import Ollama

# ollama_url = os.getenv("OLLAMA_URL")

# llm = Ollama(
#     model="mistral:latest",
#     base_url=ollama_url 
# )




# retriever = get_retriever()

# # llm = Ollama(model="mistral")

# template = """
# Tu es un assistant biomédical expert.

# Utilise UNIQUEMENT le contexte suivant pour répondre.

# Contexte :
# {context}

# Question : {question}

# Réponse en français :
# """

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type_kwargs={"prompt": prompt}
# )

# question = "Explique le fonctionnement d'un spectromètre."

# print("\n QUESTION :", question)

# response = qa_chain.invoke({"query": question})

# print("\n RÉPONSE :")
# print(response)
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.rag.retriever import get_retriever
import os

ollama_url = os.getenv("OLLAMA_URL")  

llm = Ollama(
    model="mistral:latest",
    base_url="http://ollama:11434"
)

retriever = get_retriever()

template = """
Tu es un assistant biomédical expert.

Utilise UNIQUEMENT le contexte suivant pour répondre.

Contexte :
{context}

Question : {question}

Réponse en français :
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

question = "c'est quoi la féministe ."
print("\nQUESTION :", question)

response = qa_chain.invoke({"query": question})

print("\nRÉPONSE :")
print(response)
