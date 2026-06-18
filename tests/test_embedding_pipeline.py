from pathlib import Path

import pytest

from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


def pipeline_for_chunks(size: int = 100, overlap: int = 20):
    pipeline = ChromaEmbeddingPipelineTextOnly.__new__(ChromaEmbeddingPipelineTextOnly)
    pipeline.chunk_size = size
    pipeline.chunk_overlap = overlap
    return pipeline


def test_chunk_text_respects_size_and_exact_overlap():
    pipeline = pipeline_for_chunks()
    text = "Sentence one. Sentence two! Sentence three? " * 20

    chunks = pipeline.chunk_text(text, {"mission": "apollo_11"})

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk, _ in chunks)
    assert all(
        left[-20:] == right[:20]
        for (left, _), (right, _) in zip(chunks, chunks[1:], strict=False)
    )
    assert [metadata["chunk_index"] for _, metadata in chunks] == list(range(len(chunks)))
    assert all(metadata["total_chunks"] == len(chunks) for _, metadata in chunks)


def test_chunk_text_handles_empty_and_short_text():
    pipeline = pipeline_for_chunks()

    assert pipeline.chunk_text("   ", {}) == []
    short_chunk = pipeline.chunk_text("short mission note", {"source": "note"})
    assert short_chunk[0][0] == "short mission note"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data_text/apollo11/report.txt", "apollo_11"),
        ("data_text/apollo13/report.txt", "apollo_13"),
        ("data_text/challenger/audio.txt", "challenger"),
    ],
)
def test_extract_mission_from_path(path, expected):
    assert ChromaEmbeddingPipelineTextOnly.extract_mission_from_path(Path(path)) == expected


def test_document_id_is_stable_and_chunk_specific():
    pipeline = pipeline_for_chunks()
    path = Path("data_text/apollo11/report.txt")
    first = {"mission": "apollo_11", "source": "report", "chunk_index": 0}
    second = {**first, "chunk_index": 1}

    assert pipeline.generate_document_id(path, first) == pipeline.generate_document_id(path, first)
    assert pipeline.generate_document_id(path, first) != pipeline.generate_document_id(path, second)
