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

"""Unit tests for vector store configuration, backend factory, and error handling."""

from unittest.mock import Mock, patch
import pytest

from app.vectorstore.base import (
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbeddingModelMismatchError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
)
from app.vectorstore.factory import get_vector_store
from app.vectorstore.milvus import MilvusVectorStore
from app.vectorstore.faiss import FAISSVectorStore


def test_factory_returns_faiss_store_when_configured():
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    store = get_vector_store(
        backend="faiss",
        embeddings=mock_embeddings,
        index_path="/tmp/test_faiss_config",
    )
    assert isinstance(store, FAISSVectorStore)
    assert store.index_path == "/tmp/test_faiss_config"


def test_factory_returns_milvus_store_when_configured():
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 384

    store = get_vector_store(
        backend="milvus",
        embeddings=mock_embeddings,
        uri="http://localhost:19530",
        collection_name="test_col",
    )
    assert isinstance(store, MilvusVectorStore)
    assert store.uri == "http://localhost:19530"
    assert store.collection_name == "test_col"


def test_factory_case_insensitive_backend_selection():
    mock_embeddings = Mock()
    store_lower = get_vector_store(backend="faiss", embeddings=mock_embeddings)
    store_upper = get_vector_store(backend="FAISS", embeddings=mock_embeddings)
    assert isinstance(store_lower, FAISSVectorStore)
    assert isinstance(store_upper, FAISSVectorStore)


def test_factory_rejects_unknown_backend():
    mock_embeddings = Mock()
    with pytest.raises(VectorStoreConfigurationError) as exc_info:
        get_vector_store(backend="pinecone", embeddings=mock_embeddings)

    assert "Unsupported VECTOR_STORE_BACKEND" in str(exc_info.value)
    assert "pinecone" in str(exc_info.value)


def test_vectorstore_exception_hierarchy():
    assert issubclass(VectorStoreConfigurationError, Exception)
    assert issubclass(CollectionNotFoundError, Exception)
    assert issubclass(VectorStoreConnectionError, Exception)
    assert issubclass(DimensionMismatchError, Exception)
    assert issubclass(EmbeddingModelMismatchError, Exception)
