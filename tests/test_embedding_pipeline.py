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


class FakeCollection:
    def __init__(self):
        self.added = []
        self.upserted = []
        self.deleted = []

    def add(self, **kwargs):
        self.added.append(kwargs)

    def upsert(self, **kwargs):
        self.upserted.append(kwargs)

    def delete(self, *, ids):
        self.deleted.extend(ids)


def pipeline_for_updates(existing_ids):
    pipeline = pipeline_for_chunks()
    pipeline.collection = FakeCollection()
    pipeline.get_file_documents = lambda _path: list(existing_ids)
    pipeline.get_embeddings = lambda texts: [[float(len(text))] for text in texts]
    return pipeline


def update_documents(pipeline):
    path = Path("data_text/apollo11/report.txt")
    documents = [
        ("first", {"mission": "apollo_11", "source": "report", "chunk_index": 0}),
        ("second", {"mission": "apollo_11", "source": "report", "chunk_index": 1}),
    ]
    ids = [pipeline.generate_document_id(path, metadata) for _, metadata in documents]
    return path, documents, ids


def test_skip_mode_only_adds_missing_chunks():
    seed = pipeline_for_updates([])
    path, documents, ids = update_documents(seed)
    pipeline = pipeline_for_updates([ids[0]])

    stats = pipeline.add_documents_to_collection(documents, path, update_mode="skip")

    assert stats == {"added": 1, "updated": 0, "skipped": 1}
    assert pipeline.collection.added[0]["ids"] == [ids[1]]


def test_update_mode_upserts_current_chunks_and_removes_stale_ids():
    seed = pipeline_for_updates([])
    path, documents, ids = update_documents(seed)
    pipeline = pipeline_for_updates([ids[0], "stale-id"])

    stats = pipeline.add_documents_to_collection(documents, path, update_mode="update")

    assert stats == {"added": 1, "updated": 1, "skipped": 0}
    assert pipeline.collection.deleted == ["stale-id"]
    assert pipeline.collection.upserted[0]["ids"] == ids


def test_replace_mode_embeds_then_replaces_all_file_chunks():
    seed = pipeline_for_updates([])
    path, documents, ids = update_documents(seed)
    pipeline = pipeline_for_updates([ids[0], "old-id"])

    stats = pipeline.add_documents_to_collection(documents, path, update_mode="replace")

    assert stats == {"added": 2, "updated": 0, "skipped": 0}
    assert set(pipeline.collection.deleted) == {ids[0], "old-id"}
    assert pipeline.collection.added[0]["ids"] == ids
