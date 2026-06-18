#!/usr/bin/env python3
"""Run end-to-end retrieval, generation, and evaluation over the test dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import llm_client
import rag_client
import ragas_evaluator

LOGGER = logging.getLogger(__name__)
REQUIRED_FIELDS = {"id", "category", "mission", "question", "reference_answer"}


def load_evaluation_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the JSON-formatted evaluation text file."""
    dataset_path = Path(path)
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Evaluation dataset not found: {dataset_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation dataset is not valid JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Evaluation dataset must be a non-empty JSON list")

    validated: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object")
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(f"Dataset item {index} is missing: {', '.join(sorted(missing))}")
        normalized = {field: str(item[field]).strip() for field in REQUIRED_FIELDS}
        if any(not value for value in normalized.values()):
            raise ValueError(f"Dataset item {index} contains an empty required value")
        if normalized["id"] in seen_ids:
            raise ValueError(f"Duplicate dataset id: {normalized['id']}")
        seen_ids.add(normalized["id"])
        validated.append(normalized)
    return validated


def aggregate_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute a mean for each numeric metric present in successful results."""
    metric_names = {
        name
        for result in results
        for name, value in result.get("scores", {}).items()
        if isinstance(value, (int, float))
    }
    return {
        name: fmean(
            float(result["scores"][name])
            for result in results
            if isinstance(result.get("scores", {}).get(name), (int, float))
        )
        for name in sorted(metric_names)
    }


def run_batch(
    dataset: list[dict[str, str]],
    collection,
    api_key: str,
    model: str,
    top_k: int,
) -> dict[str, Any]:
    """Execute retrieval, answer generation, and RAGAS evaluation per question."""
    results: list[dict[str, Any]] = []
    for index, item in enumerate(dataset, start=1):
        LOGGER.info("[%d/%d] %s", index, len(dataset), item["question"])
        record: dict[str, Any] = {
            key: item[key] for key in ("id", "category", "mission", "question")
        }
        try:
            retrieved = rag_client.retrieve_documents(
                collection,
                item["question"],
                n_results=top_k,
                mission_filter=item["mission"],
            )
            documents = (retrieved.get("documents") or [[]])[0]
            metadatas = (retrieved.get("metadatas") or [[]])[0]
            distances = (retrieved.get("distances") or [[]])[0]
            context = rag_client.format_context(documents, metadatas, distances)
            answer = llm_client.generate_response(api_key, item["question"], context, [], model)
            scores = ragas_evaluator.evaluate_response_quality(
                item["question"],
                answer,
                documents,
                reference_answer=item["reference_answer"],
                api_key=api_key,
            )
            record.update(
                {
                    "answer": answer,
                    "retrieved_sources": [
                        metadata.get("file_path") or metadata.get("source", "unknown")
                        for metadata in metadatas
                    ],
                    "scores": scores,
                }
            )
            if "error" in scores:
                record["error"] = scores["error"]
        except Exception as exc:
            record["error"] = str(exc)
        results.append(record)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions": len(results),
        "successful": sum("error" not in item for item in results),
        "aggregate_scores": aggregate_scores(results),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation_dataset.txt")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.openai_key:
        parser.error("Set OPENAI_API_KEY or pass --openai-key")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    try:
        dataset = load_evaluation_dataset(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit:
        dataset = dataset[: args.limit]

    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.openai_key
    collection, success, error = rag_client.initialize_rag_system(
        args.chroma_dir,
        args.collection_name,
    )
    if not success:
        LOGGER.error("Unable to initialize retrieval: %s", error)
        return 1

    report = run_batch(dataset, collection, args.openai_key, args.model, args.top_k)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Wrote report to %s", args.output)
    return 0 if report["successful"] == report["questions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
