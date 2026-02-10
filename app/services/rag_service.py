from app.rag.qa_chain import get_qa_chain

def ask_question(question: str):

    qa = get_qa_chain()

    result = qa.invoke({"query": question})

    return {
        "question": question,
        "answer": result
    }

    