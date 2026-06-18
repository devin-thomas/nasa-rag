
import pytest

import rag_client


class FakeCollection:
    def __init__(self, count=5):
        self._count = count
        self.query_kwargs = None

    def count(self):
        return self._count

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {"documents": [["result"]], "metadatas": [[{}]], "distances": [[0.1]]}


def test_retrieve_documents_applies_normalized_mission_and_caps_top_k():
    collection = FakeCollection(count=2)

    rag_client.retrieve_documents(collection, "oxygen tank", 10, "Apollo 13")

    assert collection.query_kwargs["n_results"] == 2
    assert collection.query_kwargs["where"] == {"mission": "apollo_13"}
    assert collection.query_kwargs["query_texts"] == ["oxygen tank"]


def test_retrieve_documents_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        rag_client.retrieve_documents(FakeCollection(), "  ")


def test_format_context_sorts_deduplicates_and_attributes_sources():
    documents = ["Later excerpt", "Best excerpt", "Best excerpt"]
    metadata = [
        {"mission": "apollo_13", "source": "later", "document_category": "technical"},
        {"mission": "apollo_11", "file_path": "data/best.txt", "document_category": "report"},
        {"mission": "apollo_11", "source": "duplicate"},
    ]

    context = rag_client.format_context(documents, metadata, [0.4, 0.1, 0.2])

    assert context.index("Best excerpt") < context.index("Later excerpt")
    assert context.count("Best excerpt") == 1
    assert "[Source 1 | Mission: Apollo 11 | File: data/best.txt" in context
    assert "Relevance: 0.900" in context


def test_discover_backends_returns_empty_for_missing_root():
    assert rag_client.discover_chroma_backends("definitely-not-a-real-directory") == {}
