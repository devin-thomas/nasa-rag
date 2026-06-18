import pytest

from batch_evaluate import aggregate_scores, load_evaluation_dataset
from ragas_evaluator import (
    evaluate_response_quality,
    lexical_context_precision,
    reference_token_f1,
)


def test_evaluation_dataset_covers_required_categories():
    dataset = load_evaluation_dataset("evaluation_dataset.txt")

    assert len(dataset) >= 5
    assert {item["category"] for item in dataset} >= {
        "overview",
        "emergency",
        "disaster_analysis",
        "crew",
        "technical",
        "timeline",
    }
    assert {item["mission"] for item in dataset} == {"apollo_11", "apollo_13", "challenger"}


def test_lexical_metrics_are_bounded_and_interpretable():
    precision = lexical_context_precision(
        "Apollo landed safely",
        ["Apollo safely landed on the Moon"],
    )
    token_f1 = reference_token_f1("Apollo landed", "Apollo landed on the Moon")

    assert precision == 1.0
    assert token_f1 == pytest.approx(0.8)


def test_evaluator_returns_clear_input_errors_without_api_calls():
    result = evaluate_response_quality("", "answer", ["context"])
    assert result["error"] == "Question cannot be empty"
    assert evaluate_response_quality("question", "answer", [""])["error"].startswith(
        "Retrieved contexts"
    )


def test_aggregate_scores_ignores_errors_and_means_numeric_metrics():
    results = [
        {"scores": {"faithfulness": 0.5, "response_relevancy": 0.8}},
        {"scores": {"faithfulness": 1.0, "response_relevancy": 0.6}},
        {"error": "failed"},
    ]

    assert aggregate_scores(results) == {"faithfulness": 0.75, "response_relevancy": 0.7}
