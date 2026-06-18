"""Real-time RAGAS evaluation for NASA RAG responses."""

from __future__ import annotations

import asyncio
import math
import os
import re
from typing import Any

from openai import AsyncOpenAI

try:
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR = ""
except ImportError as exc:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(exc)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _content_tokens(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOP_WORDS}


def lexical_context_precision(answer: str, contexts: list[str]) -> float:
    """Estimate how much answer vocabulary is supported by retrieved context."""
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = _content_tokens("\n".join(contexts))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def reference_token_f1(answer: str, reference_answer: str) -> float:
    """Compute a transparent token-overlap F1 against an optional reference answer."""
    answer_tokens = _content_tokens(answer)
    reference_tokens = _content_tokens(reference_answer)
    if not answer_tokens or not reference_tokens:
        return 0.0
    overlap = len(answer_tokens & reference_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _normalized_score(value: float) -> float:
    score = float(value)
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


async def _evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    api_key: str,
    evaluator_model: str,
) -> tuple[float, float]:
    client = AsyncOpenAI(api_key=api_key)
    evaluator_llm = llm_factory(
        evaluator_model,
        provider="openai",
        client=client,
        temperature=0,
    )
    evaluator_embeddings = RagasOpenAIEmbeddings(
        client=client,
        model="text-embedding-3-small",
    )
    try:
        relevancy_metric = AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=3,
        )
        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        relevancy, faithfulness = await asyncio.gather(
            relevancy_metric.ascore(user_input=question, response=answer),
            faithfulness_metric.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            ),
        )
        return float(relevancy.value), float(faithfulness.value)
    finally:
        await client.close()


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: list[str],
    *,
    reference_answer: str | None = None,
    api_key: str | None = None,
    evaluator_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Evaluate a question/context/answer triple with RAGAS and lexical metrics.

    The two primary scores are RAGAS answer relevancy (reported as response
    relevancy to match the project rubric) and faithfulness. Lexical context
    precision is always included; reference token F1 is included when a reference
    answer is supplied.
    """
    if not isinstance(question, str) or not question.strip():
        return {"error": "Question cannot be empty"}
    if not isinstance(answer, str) or not answer.strip():
        return {"error": "Answer cannot be empty"}
    if not isinstance(contexts, list) or not contexts:
        return {"error": "At least one retrieved context is required"}
    clean_contexts = [item.strip() for item in contexts if isinstance(item, str) and item.strip()]
    if not clean_contexts:
        return {"error": "Retrieved contexts must contain non-empty strings"}
    if not RAGAS_AVAILABLE:
        return {"error": f"RAGAS is not available: {RAGAS_IMPORT_ERROR}"}

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        return {"error": "Set OPENAI_API_KEY to run RAGAS evaluation"}
    if not evaluator_model.strip():
        return {"error": "Evaluator model cannot be empty"}

    try:
        relevancy, faithfulness = asyncio.run(
            _evaluate_with_ragas(
                question.strip(),
                answer.strip(),
                clean_contexts,
                resolved_key,
                evaluator_model.strip(),
            )
        )
    except Exception as exc:
        return {"error": f"RAGAS evaluation failed: {exc}"}

    results: dict[str, Any] = {
        "response_relevancy": _normalized_score(relevancy),
        "faithfulness": _normalized_score(faithfulness),
        "lexical_context_precision": _normalized_score(
            lexical_context_precision(answer, clean_contexts)
        ),
    }
    if reference_answer and reference_answer.strip():
        results["reference_token_f1"] = _normalized_score(
            reference_token_f1(answer, reference_answer)
        )
    return results
