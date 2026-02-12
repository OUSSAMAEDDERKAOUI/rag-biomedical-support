import mlflow
from app.rag.qa_chain import get_qa_chain
from app.monitoring.mlflow_logger import start_rag_run

def load_qa():
    return get_qa_chain()

with start_rag_run("model_registration_mistral", "Hybrid_RAG"):

    qa = get_qa_chain()

    mlflow.langchain.log_model(
        qa,
        artifact_path="rag_chain",
        loader_fn=load_qa
    )
