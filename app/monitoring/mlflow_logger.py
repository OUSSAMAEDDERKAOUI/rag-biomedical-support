import mlflow
import os
from datetime import datetime

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "rag_biomedical")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


def start_rag_run():
    run_name = f"rag_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return mlflow.start_run(run_name=run_name)


def log_rag_config(config: dict):
    for k, v in config.items():
        mlflow.log_param(k, v)


def log_retrieval_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_llm_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_answer(question, answer, context):
    mlflow.log_text(question, "question.txt")
    mlflow.log_text(answer, "answer.txt")
    mlflow.log_text("\n\n".join(context), "context.txt")


def end_run():
    mlflow.end_run()
