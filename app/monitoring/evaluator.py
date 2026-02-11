from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import mlflow


def evaluate_rag(question, answer, context):

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context
    )

    metrics = [
        AnswerRelevancyMetric(),
        FaithfulnessMetric()
    ]

    results = {}

    for metric in metrics:
        metric.measure(test_case)
        results[metric.__class__.__name__] = metric.score

    for k, v in results.items():
        mlflow.log_metric(k, v)

    return results
