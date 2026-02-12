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
from app.monitoring.mlflow_logger import start_rag_run
import mlflow
ollama_url = os.getenv("OLLAMA_URL")  

llm = Ollama(
    model="mistral:latest",
    base_url="http://ollama:11434"
)


with start_rag_run("retrieval","HybridRetriever_dense_bm25"):

    retriever = get_retriever()




template =     """
            Tu es un assistant technique spécialisé dans la maintenance d'équipements biomédicaux. 
            Ton unique mission est d'extraire des informations précises à partir des MANUELS fournis.

            ### CONTEXTE DES MANUELS :
            {context}

            ### RÈGLES STRICTES :
            
Zéro Hallucination : Si l'information n'est pas explicitement écrite dans le texte ci-dessus, réponds exactement : "Je ne trouve pas l'information dans les manuels disponibles".
Fidélité Technique : Ne reformule pas les codes d'erreur ou les valeurs de pression/température. Utilise le langage exact du manuel.
Isolation de Connaissance : Ne réponds pas en utilisant tes propres connaissances générales sur le sujet. Si le manuel est silencieux, tu es silencieux.
Priorité au Contexte : Si l'utilisateur demande une procédure non décrite ici, refuse d'y répondre.

            ### QUESTION DE L'UTILISATEUR :
            {question}

            RÉPONSE (en français) :
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

question = " A QUOI SERT UNE BALANCE Au laboratoire ?"
print("\nQUESTION :", question)

response = qa_chain.invoke({"query": question})

print("\nRÉPONSE :")
print(response)




JUDGE_PROMPT = """
Tu es un évaluateur objectif d’un système RAG.

Question :
{question}

Contexte fourni :
{context}

Réponse générée :
{answer}

Évalue la réponse selon ces critères :
- Fidélité au contexte (0-5)
- Pertinence (0-5)
- Clarté (0-5)

Donne :
- Un score total sur 15
- Une justification courte (2 lignes max)

Réponds en JSON.
"""





def llm_judge(llm, question, context, answer):
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )

    result = llm(prompt)  # mistral local ou API
    return result




def evaluate_rag(query,context, response, llm):
    


    evaluation = llm_judge(
        llm,
        query,
        context,
        response
    )

    return {
        "answer": response,
        "evaluation": evaluation
    }

