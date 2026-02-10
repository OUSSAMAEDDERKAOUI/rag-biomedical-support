from app.rag.qa_chain import get_qa_chain
from app.monitoring.mlflow_logger import log_answer

def ask_question(question: str):

    qa = get_qa_chain()

    result = qa.invoke({"query": question})
    log_answer(question, result)

    return {
        "question": question,
        "answer": result
    }

    