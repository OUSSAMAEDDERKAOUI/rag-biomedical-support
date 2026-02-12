from app.rag.qa_chain import get_qa_chain
from app.monitoring.mlflow_logger import log_answer
from app.monitoring.evaluator import evaluate_rag
from app.rag.retriever import get_retriever

def ask_question(question: str):

    qa = get_qa_chain()

    result = qa.invoke({"query": question})
    log_answer(question, result)
    retriever = get_retriever()
    # contexts = retriever.get_relevant_documents(question)

    # results = evaluate_rag(question, result, contexts)

    return {
        "question": question,
        "answer": result,
        # "answer evaluation":results
    }

    