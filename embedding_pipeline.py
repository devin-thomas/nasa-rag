#!/usr/bin/env python3
"""Build and inspect a persistent ChromaDB index for NASA mission documents."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

LOGGER = logging.getLogger(__name__)
UpdateMode = Literal["skip", "update", "replace"]


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging without creating repository-local artifacts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class ChromaEmbeddingPipelineTextOnly:
    """Chunk NASA text files, embed them with OpenAI, and persist them in Chroma."""

    def __init__(
        self,
        openai_api_key: str | None,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1_000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        self.openai_api_key = openai_api_key or None
        self.chroma_persist_directory = str(Path(chroma_persist_directory))
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_client = (
            OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        )

        settings = Settings(anonymized_telemetry=False)
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_directory,
            settings=settings,
        )

        if self.openai_api_key:
            self.embedding_function = OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name=self.embedding_model,
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "NASA Apollo 11, Apollo 13, and Challenger archive",
                    "embedding_model": self.embedding_model,
                    "hnsw:space": "cosine",
                },
            )
        else:
            available = {item.name for item in self.chroma_client.list_collections()}
            if self.collection_name not in available:
                raise ValueError(
                    "An OpenAI API key is required to create a new collection. "
                    "Set OPENAI_API_KEY or pass --openai-key."
                )
            self.embedding_function = None
            self.collection = self.chroma_client.get_collection(self.collection_name)

    def chunk_text(
        self, text: str, metadata: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Split text into bounded, consistently overlapping character chunks."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return []

        positions: list[tuple[int, int]] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            maximum_end = min(start + self.chunk_size, text_length)
            end = maximum_end

            if maximum_end < text_length:
                minimum_break = start + max(
                    self.chunk_overlap + 1,
                    int(self.chunk_size * 0.6),
                )
                candidate = text[minimum_break:maximum_end]
                boundaries = [
                    candidate.rfind("\n\n"),
                    candidate.rfind(". "),
                    candidate.rfind("? "),
                    candidate.rfind("! "),
                    candidate.rfind("\n"),
                ]
                best_boundary = max(boundaries)
                if best_boundary >= 0:
                    separator_length = 2 if candidate[best_boundary : best_boundary + 2] in {
                        "\n\n",
                        ". ",
                        "? ",
                        "! ",
                    } else 1
                    end = minimum_break + best_boundary + separator_length

            if end <= start:
                end = maximum_end
            positions.append((start, end))
            if end == text_length:
                break
            start = end - self.chunk_overlap

        total_chunks = len(positions)
        chunks: list[tuple[str, dict[str, Any]]] = []
        for index, (chunk_start, chunk_end) in enumerate(positions):
            chunk_metadata = dict(metadata)
            chunk_metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "char_start": chunk_start,
                    "char_end": chunk_end,
                }
            )
            chunks.append((text[chunk_start:chunk_end], chunk_metadata))
        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """Return whether ``doc_id`` is present in the collection."""
        if not doc_id:
            return False
        result = self.collection.get(ids=[doc_id], include=[])
        return bool(result.get("ids"))

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a batch, retrying transient OpenAI failures."""
        if not texts:
            return []
        if self.openai_client is None:
            raise ValueError("An OpenAI API key is required to create embeddings")

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts,
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                return [item.embedding for item in ordered]
            except Exception as exc:  # OpenAI exposes several transport subclasses
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI embedding request failed: {last_error}") from last_error

    def get_embedding(self, text: str) -> list[float]:
        """Create one OpenAI embedding."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        return self.get_embeddings([text])[0]

    def generate_document_id(
        self, file_path: Path, metadata: dict[str, Any]
    ) -> str:
        """Generate a stable, readable ID from mission, source, path, and chunk."""
        mission = self._slug(str(metadata.get("mission", "unknown")))
        source = self._slug(str(metadata.get("source", file_path.stem)))[:48]
        chunk_index = int(metadata.get("chunk_index", 0))
        normalized_path = file_path.as_posix().lower()
        path_digest = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:8]
        return f"{mission}-{source}-{path_digest}-chunk-{chunk_index:05d}"

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"

    def process_text_file(
        self, file_path: Path
    ) -> list[tuple[str, dict[str, Any]]]:
        """Read a UTF-8 text file and return chunks with source metadata."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise OSError(f"Unable to read {file_path}: {exc}") from exc
        if not content.strip():
            return []

        metadata: dict[str, Any] = {
            "source": file_path.stem,
            "file_path": file_path.as_posix(),
            "file_type": "text",
            "content_type": "full_text",
            "mission": self.extract_mission_from_path(file_path),
            "data_type": self.extract_data_type_from_path(file_path),
            "document_category": self.extract_document_category_from_filename(
                file_path.name
            ),
            "file_size": len(content),
            "processed_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return self.chunk_text(content, metadata)

    @staticmethod
    def extract_mission_from_path(file_path: Path) -> str:
        path = file_path.as_posix().lower()
        if "apollo11" in path or "apollo_11" in path:
            return "apollo_11"
        if "apollo13" in path or "apollo_13" in path:
            return "apollo_13"
        if "challenger" in path:
            return "challenger"
        return "unknown"

    @staticmethod
    def extract_data_type_from_path(file_path: Path) -> str:
        path = file_path.as_posix().lower()
        name = file_path.name.lower()
        if "audio" in path or "mission_audio" in name:
            return "audio_transcript"
        if "transcript" in path or "transcript" in name or "transscript" in name:
            return "transcript"
        if "textract" in path or "textract" in name:
            return "textract_extracted"
        if "flight_plan" in path or "flight_plan" in name:
            return "flight_plan"
        return "document"

    @staticmethod
    def extract_document_category_from_filename(filename: str) -> str:
        name = filename.lower()
        categories = (
            ("pao", "public_affairs_officer"),
            ("flight_plan", "flight_plan"),
            ("mission_audio", "mission_audio"),
            ("ntrs", "nasa_archive"),
            ("19900066485", "technical_report"),
            ("19710015566", "mission_report"),
            ("tec", "technical"),
            ("cm", "command_module"),
            ("full_text", "complete_document"),
        )
        return next(
            (category for marker, category in categories if marker in name),
            "general_document",
        )

    def scan_text_files_only(self, base_path: str) -> list[Path]:
        """Return supported mission text files in deterministic order."""
        base = Path(base_path)
        if not base.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {base}")

        files: list[Path] = []
        for mission in ("apollo11", "apollo13", "challenger"):
            mission_dir = base / mission
            if not mission_dir.is_dir():
                LOGGER.warning("Mission directory not found: %s", mission_dir)
                continue
            files.extend(
                path
                for path in mission_dir.rglob("*.txt")
                if not path.name.startswith(".") and "summary" not in path.name.lower()
            )
        return sorted(files, key=lambda path: path.as_posix().lower())

    def get_file_documents(self, file_path: Path) -> list[str]:
        """Return all indexed chunk IDs associated with ``file_path``."""
        source = file_path.stem
        mission = self.extract_mission_from_path(file_path)
        result = self.collection.get(
            where={"$and": [{"source": source}, {"mission": mission}]},
            include=[],
        )
        return list(result.get("ids", []))

    def add_documents_to_collection(
        self,
        documents: list[tuple[str, dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: UpdateMode = "skip",
    ) -> dict[str, int]:
        """Embed and write one file's chunks according to ``update_mode``."""
        if update_mode not in {"skip", "update", "replace"}:
            raise ValueError("update_mode must be skip, update, or replace")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not documents:
            return {"added": 0, "updated": 0, "skipped": 0}

        ids = [self.generate_document_id(file_path, metadata) for _, metadata in documents]
        current_ids = set(self.get_file_documents(file_path))
        expected_ids = set(ids)
        stats = {"added": 0, "updated": 0, "skipped": 0}

        if update_mode == "skip":
            pending = [
                item
                for item in zip(ids, documents, strict=True)
                if item[0] not in current_ids
            ]
            stats["skipped"] = len(documents) - len(pending)
        else:
            pending = list(zip(ids, documents, strict=True))

        prepared: list[tuple[list[str], list[str], list[dict[str, Any]], list[list[float]]]] = []
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            batch_ids = [item[0] for item in batch]
            texts = [item[1][0] for item in batch]
            metadatas = [item[1][1] for item in batch]
            prepared.append((batch_ids, texts, metadatas, self.get_embeddings(texts)))

        if update_mode == "replace" and current_ids:
            self.collection.delete(ids=sorted(current_ids))
            current_ids.clear()
        elif update_mode == "update":
            stale_ids = current_ids - expected_ids
            if stale_ids:
                self.collection.delete(ids=sorted(stale_ids))

        for batch_ids, texts, metadatas, embeddings in prepared:
            if update_mode == "update":
                self.collection.upsert(
                    ids=batch_ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                stats["updated"] += sum(item in current_ids for item in batch_ids)
                stats["added"] += sum(item not in current_ids for item in batch_ids)
            else:
                self.collection.add(
                    ids=batch_ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                stats["added"] += len(batch_ids)
        return stats

    def process_all_text_data(
        self,
        base_path: str,
        update_mode: UpdateMode = "skip",
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Process every supported NASA source file and return aggregate statistics."""
        stats: dict[str, Any] = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }

        for file_path in self.scan_text_files_only(base_path):
            mission = self.extract_mission_from_path(file_path)
            mission_stats = stats["missions"].setdefault(
                mission,
                {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0},
            )
            try:
                chunks = self.process_text_file(file_path)
                result = self.add_documents_to_collection(
                    chunks,
                    file_path,
                    batch_size=batch_size,
                    update_mode=update_mode,
                )
                stats["files_processed"] += 1
                stats["total_chunks"] += len(chunks)
                stats["documents_added"] += result["added"]
                stats["documents_updated"] += result["updated"]
                stats["documents_skipped"] += result["skipped"]
                mission_stats["files"] += 1
                mission_stats["chunks"] += len(chunks)
                for key in ("added", "updated", "skipped"):
                    mission_stats[key] += result[key]
                LOGGER.info("Indexed %s (%d chunks)", file_path.name, len(chunks))
            except Exception as exc:
                stats["errors"] += 1
                LOGGER.error("Failed to process %s: %s", file_path, exc)
        return stats

    def get_collection_info(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata or {},
            "persist_directory": self.chroma_persist_directory,
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> dict[str, Any]:
        if not query_text.strip():
            raise ValueError("query_text cannot be empty")
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")
        if self.collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    def get_collection_stats(self) -> dict[str, Any]:
        """Return collection size and metadata aggregates."""
        result = self.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        stats: dict[str, Any] = {
            "total_chunks": self.collection.count(),
            "unique_sources": len(
                {item.get("file_path", item.get("source")) for item in metadatas}
            ),
            "missions": {},
            "data_types": {},
            "document_categories": {},
        }
        for metadata in metadatas:
            for field, bucket in (
                ("mission", "missions"),
                ("data_type", "data_types"),
                ("document_category", "document_categories"),
            ):
                value = str(metadata.get(field, "unknown"))
                stats[bucket][value] = stats[bucket].get(value, 0) + 1
        return stats


def _log_mapping(title: str, values: dict[str, Any]) -> None:
    LOGGER.info("%s", title)
    for key, value in values.items():
        LOGGER.info("  %s: %s", key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="./data_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--update-mode", choices=["skip", "update", "replace"], default="skip")
    parser.add_argument("--test-query")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if not args.openai_key and not args.stats_only:
        parser.error("Set OPENAI_API_KEY or pass --openai-key")

    try:
        pipeline = ChromaEmbeddingPipelineTextOnly(
            openai_api_key=args.openai_key,
            chroma_persist_directory=args.chroma_dir,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if args.stats_only:
            _log_mapping("Collection statistics", pipeline.get_collection_stats())
            return 0

        started = time.monotonic()
        stats = pipeline.process_all_text_data(
            args.data_path,
            update_mode=args.update_mode,
            batch_size=args.batch_size,
        )
        _log_mapping("Processing summary", stats)
        LOGGER.info("Elapsed seconds: %.2f", time.monotonic() - started)
        _log_mapping("Collection", pipeline.get_collection_info())

        if args.test_query:
            results = pipeline.query_collection(args.test_query)
            for index, document in enumerate(results.get("documents", [[]])[0], start=1):
                LOGGER.info("Result %d: %s", index, document[:240].replace("\n", " "))
        return 1 if stats["errors"] else 0
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
