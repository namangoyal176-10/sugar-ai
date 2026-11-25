# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Unit and integration tests for Milvus vector store backend."""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch
import pytest

from langchain_core.documents import Document

from app.vectorstore.base import (
    CollectionNotFoundError,
    DimensionMismatchError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
)
from app.vectorstore.milvus import MilvusVectorStore


@pytest.fixture
def mock_embeddings():
    emb = Mock()
    emb.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
    emb.embed_query.return_value = [0.1] * 384
    return emb


# --- Unit Tests with Mocks ------------------------------------------------


def test_milvus_load_raises_when_collection_not_found(mock_embeddings):
    mock_client = Mock()
    mock_client.has_collection.return_value = False

    store = MilvusVectorStore(
        embeddings=mock_embeddings,
        uri="http://localhost:19530",
        collection_name="nonexistent_collection",
    )
    store._client = mock_client

    with pytest.raises(CollectionNotFoundError) as exc_info:
        store.load()

    assert "does not exist" in str(exc_info.value)
    assert "nonexistent_collection" in str(exc_info.value)


def test_milvus_load_raises_when_collection_is_empty(mock_embeddings):
    mock_client = Mock()
    mock_client.has_collection.return_value = True
    mock_client.describe_collection.return_value = {
        "fields": [{"name": "vector", "params": {"dim": 384}}]
    }
    mock_client.get_collection_stats.return_value = {"row_count": 0}

    store = MilvusVectorStore(
        embeddings=mock_embeddings,
        uri="http://localhost:19530",
        collection_name="empty_collection",
        embedding_dimension=384,
    )
    store._client = mock_client

    with pytest.raises(CollectionNotFoundError) as exc_info:
        store.load()

    assert "contains 0 documents" in str(exc_info.value)


def test_milvus_load_raises_on_dimension_mismatch(mock_embeddings):
    mock_client = Mock()
    mock_client.has_collection.return_value = True
    mock_client.describe_collection.return_value = {
        "fields": [{"name": "vector", "params": {"dim": 768}}]
    }
    mock_client.get_collection_stats.return_value = {"row_count": 10}

    store = MilvusVectorStore(
        embeddings=mock_embeddings,
        uri="http://localhost:19530",
        collection_name="mismatched_collection",
        embedding_dimension=384,  # expecting 384, but collection has 768
    )
    store._client = mock_client

    with pytest.raises(DimensionMismatchError) as exc_info:
        store.load()

    assert "vector dimension (768) does not match expected embedding dimension (384)" in str(exc_info.value)


def test_milvus_load_succeeds_when_collection_is_valid(mock_embeddings):
    mock_client = Mock()
    mock_client.has_collection.return_value = True
    mock_client.describe_collection.return_value = {
        "fields": [{"name": "vector", "params": {"dim": 384}}]
    }
    mock_client.get_collection_stats.return_value = {"row_count": 42}

    store = MilvusVectorStore(
        embeddings=mock_embeddings,
        uri="http://localhost:19530",
        collection_name="valid_collection",
        embedding_dimension=384,
    )
    store._client = mock_client

    assert store.load() is True
    mock_client.load_collection.assert_called_once_with("valid_collection")


def test_milvus_connection_error_is_wrapped(mock_embeddings):
    with patch("pymilvus.MilvusClient", side_effect=Exception("Connection refused")):
        store = MilvusVectorStore(
            embeddings=mock_embeddings,
            uri="http://unavailable-milvus.example:19530",
        )
        with pytest.raises(VectorStoreConnectionError) as exc_info:
            store.load()

        assert "Failed to connect to Milvus" in str(exc_info.value)


def test_milvus_upsert_creates_collection_if_missing(mock_embeddings):
    mock_client = Mock()
    mock_client.has_collection.return_value = False

    store = MilvusVectorStore(
        embeddings=mock_embeddings,
        uri="http://localhost:19530",
        collection_name="new_collection",
        embedding_dimension=384,
        metric_type="COSINE",
    )
    store._client = mock_client

    docs = [
        Document(
            page_content="Sugar Labs activity tutorial",
            metadata={
                "source": "docs/sugar.pdf",
                "doc_id": "sugar",
                "chunk_id": "sugar_p1_c0",
                "title": "Sugar Tutorial",
                "page": 1,
                "content_hash": "hash123",
                "doc_hash": "dochash123",
            },
        )
    ]

    ids = store.upsert_documents(docs)

    assert ids == ["sugar_p1_c0"]
    mock_client.create_collection.assert_called_once_with(
        collection_name="new_collection",
        dimension=384,
        metric_type="COSINE",
        id_type="string",
        max_length=128,
        auto_id=False,
        enable_dynamic_field=True,
    )
    mock_client.upsert.assert_called_once()
    mock_client.load_collection.assert_called_once_with("new_collection")


def test_milvus_similarity_search_preserves_metadata(mock_embeddings):
    mock_client = Mock()
    mock_client.search.return_value = [
        [
            {
                "id": "sugar_p1_c0",
                "distance": 0.92,
                "entity": {
                    "text": "Sugar Labs activity tutorial content",
                    "source": "docs/sugar.pdf",
                    "title": "Sugar Tutorial",
                    "page": 1,
                    "doc_id": "sugar",
                    "content_hash": "hash123",
                    "doc_hash": "dochash123",
                },
            }
        ]
    ]

    store = MilvusVectorStore(
        embeddings=mock_embeddings,
        uri="http://localhost:19530",
        collection_name="test_search",
        metric_type="COSINE",
    )
    store._client = mock_client

    results = store.similarity_search_with_relevance_scores("activity tutorial", k=1)

    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "Sugar Labs activity tutorial content"
    assert doc.metadata["source"] == "docs/sugar.pdf"
    assert doc.metadata["title"] == "Sugar Tutorial"
    assert doc.metadata["page"] == 1
    assert doc.metadata["chunk_id"] == "sugar_p1_c0"
    assert score == pytest.approx(0.92)


def test_milvus_close_releases_client(mock_embeddings):
    mock_client = Mock()
    store = MilvusVectorStore(embeddings=mock_embeddings)
    store._client = mock_client

    store.close()

    mock_client.close.assert_called_once()
    assert store._client is None


# --- Integration Test with Local Milvus Lite ---------------------------------


def test_milvus_lite_integration_lifecycle(mock_embeddings):
    """End-to-end integration test using embedded file-backed Milvus Lite."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_lifecycle.db")

    try:
        store = MilvusVectorStore(
            embeddings=mock_embeddings,
            uri=db_path,
            collection_name="test_integration",
            embedding_dimension=384,
            metric_type="COSINE",
        )

        # 1. Uninitialized load raises CollectionNotFoundError
        with pytest.raises(CollectionNotFoundError):
            store.load()

        # 2. Ingest document
        docs = [
            Document(
                page_content="Educational software for children by Sugar Labs",
                metadata={"source": "docs/intro.pdf", "page": 1, "doc_id": "intro", "chunk_id": "intro_p1_c0"},
            )
        ]
        ids = store.upsert_documents(docs)
        assert ids == ["intro_p1_c0"]
        assert store.count() == 1

        # 3. Subsequent load succeeds
        assert store.load() is True

        # 4. Search returns matching document and score
        results = store.similarity_search_with_relevance_scores("Sugar Labs education", k=1)
        assert len(results) == 1
        assert "Sugar Labs" in results[0][0].page_content
        assert results[0][1] > 0.0

        # 5. Delete document
        store.delete_by_ids(["intro_p1_c0"])
        meta = store.get_indexed_metadata()
        assert "intro_p1_c0" not in meta

        # 6. Close connection
        store.close()
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
