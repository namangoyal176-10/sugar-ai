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

"""Unit tests for FAISS vector store backend with safe persistence and integrity checks."""

import os
import shutil
import tempfile
from unittest.mock import Mock
import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.vectorstore.base import (
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbeddingModelMismatchError,
    VectorStoreError,
)
from app.vectorstore.faiss import FAISSVectorStore


class FakeEmbeddings(Embeddings):
    def __init__(self, dim=384):
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.dim


@pytest.fixture
def mock_embeddings():
    return FakeEmbeddings(dim=384)


@pytest.fixture
def temp_index_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_faiss_load_raises_when_index_not_found(mock_embeddings, temp_index_dir):
    empty_dir = os.path.join(temp_index_dir, "nonexistent")
    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=empty_dir)

    with pytest.raises(CollectionNotFoundError) as exc_info:
        store.load()

    assert "No persistent FAISS index found" in str(exc_info.value)


def test_faiss_save_and_reload_persistence(mock_embeddings, temp_index_dir):
    store1 = FAISSVectorStore(
        embeddings=mock_embeddings,
        index_path=temp_index_dir,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
    )

    docs = [
        Document(
            page_content="Sugar Activities for Kids",
            metadata={"source": "docs/activities.pdf", "doc_id": "act", "chunk_id": "act_p1_c0"},
        )
    ]
    ids = store1.upsert_documents(docs)
    assert ids == ["act_p1_c0"]
    assert store1.count() == 1

    # Verify manifest and index files were created
    assert os.path.exists(os.path.join(temp_index_dir, "index.faiss"))
    assert os.path.exists(os.path.join(temp_index_dir, "index.pkl"))
    assert os.path.exists(os.path.join(temp_index_dir, "manifest.json"))

    # Reload in a completely fresh store instance without re-embedding
    store2 = FAISSVectorStore(
        embeddings=mock_embeddings,
        index_path=temp_index_dir,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    assert store2.load() is True
    assert store2.count() == 1

    # Search should find the persisted document
    results = store2.similarity_search_with_relevance_scores("Sugar Kids", k=1)
    assert len(results) == 1
    assert "Sugar Activities" in results[0][0].page_content


def test_faiss_rejects_tampered_pickle_file(mock_embeddings, temp_index_dir):
    """Integrity check detects altered index.pkl to protect against deserialization attacks."""
    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_index_dir)
    docs = [Document(page_content="Sample text", metadata={"chunk_id": "c1"})]
    store.upsert_documents(docs)

    # Tamper with index.pkl by appending data
    pkl_file = os.path.join(temp_index_dir, "index.pkl")
    with open(pkl_file, "ab") as f:
        f.write(b"corrupted_tampered_bytes")

    # Reloading should fail integrity check
    fresh_store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_index_dir)
    with pytest.raises(VectorStoreError) as exc_info:
        fresh_store.load()

    assert "integrity check failed" in str(exc_info.value)


def test_faiss_rejects_mismatched_embedding_model(mock_embeddings, temp_index_dir):
    store = FAISSVectorStore(
        embeddings=mock_embeddings,
        index_path=temp_index_dir,
        embedding_model_name="old-model-name",
    )
    docs = [Document(page_content="Sample text", metadata={"chunk_id": "c1"})]
    store.upsert_documents(docs)

    fresh_store = FAISSVectorStore(
        embeddings=mock_embeddings,
        index_path=temp_index_dir,
        embedding_model_name="new-different-model-name",
    )
    with pytest.raises(EmbeddingModelMismatchError) as exc_info:
        fresh_store.load()

    assert "was created with model 'old-model-name'" in str(exc_info.value)


def test_faiss_clear_removes_persisted_files(mock_embeddings, temp_index_dir):
    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_index_dir)
    docs = [Document(page_content="Sample", metadata={"chunk_id": "c1"})]
    store.upsert_documents(docs)
    assert os.path.exists(temp_index_dir)

    store.clear()
    assert not os.path.exists(temp_index_dir)
    assert store.count() == 0
