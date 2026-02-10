from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from app.rag.retriever import get_retriever
from langchain.schema import Document
import os

ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

llm = Ollama(
    model="mistral:latest",
    base_url="http://ollama:11434",
    temperature=0.3,
    top_k=10,
    top_p=0.9
)

retriever = get_retriever()


QA_PROMPT = """
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


def generate_answer(question, retriever, llm):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs)

    answer = llm(QA_PROMPT.format(context=context, question=question))

    return answer, context

def llm_judge(question, context, answer, llm):
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )
    return llm(prompt)

def evaluate_rag(question, retriever, llm):
    answer, context = generate_answer(question, retriever, llm)
    evaluation = llm_judge(question, context, answer, llm)

    return {
        "question": question,
        "answer": answer,
        "evaluation": evaluation
    }


if __name__ == "__main__":
    question = " comment on peut Vérifie la  fonctionnement d'une BALANCE ?"
    result = evaluate_rag(question, retriever, llm)

    print("\nQUESTION :", result["question"])
    print("\nRÉPONSE :", result["answer"])
    print("\nÉVALUATION :", result["evaluation"])
