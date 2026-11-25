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

"""Unit tests for the incremental document ingestion engine."""

import os
import shutil
import tempfile
from unittest.mock import Mock
import pytest

from langchain_core.embeddings import Embeddings

from app.ingest import (
    IngestionEngine,
    compute_content_hash,
    compute_file_hash,
    generate_doc_id,
)
from app.vectorstore.faiss import FAISSVectorStore


class CountingFakeEmbeddings(Embeddings):
    def __init__(self, dim=384):
        self.dim = dim
        self.embed_documents_calls = 0

    def embed_documents(self, texts):
        self.embed_documents_calls += 1
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.dim


@pytest.fixture
def mock_embeddings():
    return CountingFakeEmbeddings(dim=384)


@pytest.fixture
def temp_workspace():
    ws = tempfile.mkdtemp()
    index_dir = os.path.join(ws, "index")
    docs_dir = os.path.join(ws, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    yield {"root": ws, "index": index_dir, "docs": docs_dir}
    shutil.rmtree(ws, ignore_errors=True)


def test_hashing_and_doc_id_helpers(temp_workspace):
    test_file = os.path.join(temp_workspace["docs"], "test_doc.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Sugar Labs creates software for children.")

    doc_id = generate_doc_id(test_file)
    assert doc_id == "test_doc"

    fhash = compute_file_hash(test_file)
    assert len(fhash) == 64  # sha256 hex string

    chash = compute_content_hash("Hello World")
    assert len(chash) == 64


def test_initial_ingestion(mock_embeddings, temp_workspace):
    doc1 = os.path.join(temp_workspace["docs"], "doc1.txt")
    with open(doc1, "w", encoding="utf-8") as f:
        f.write("First document content about Sugar Labs and Python activities.")

    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_workspace["index"])
    engine = IngestionEngine(vector_store=store, embeddings=mock_embeddings)

    stats = engine.sync_documents(file_paths=[doc1])

    assert stats.total_files == 1
    assert stats.new_files == 1
    assert stats.added_chunks > 0
    assert stats.skipped_chunks == 0
    assert store.count() > 0


def test_incremental_ingestion_skips_unchanged_file(mock_embeddings, temp_workspace):
    doc1 = os.path.join(temp_workspace["docs"], "doc1.txt")
    with open(doc1, "w", encoding="utf-8") as f:
        f.write("Persistent document content.")

    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_workspace["index"])
    engine = IngestionEngine(vector_store=store, embeddings=mock_embeddings)

    # First run: indexes file
    stats1 = engine.sync_documents(file_paths=[doc1])
    assert stats1.new_files == 1
    assert stats1.added_chunks > 0

    mock_embeddings.embed_documents_calls = 0

    # Second run: file is unchanged, must skip embedding entirely
    stats2 = engine.sync_documents(file_paths=[doc1])
    assert stats2.unchanged_files == 1
    assert stats2.new_files == 0
    assert stats2.modified_files == 0
    assert stats2.added_chunks == 0
    assert stats2.skipped_chunks == stats1.added_chunks

    # Zero embedding calls on subsequent unchanged run
    assert mock_embeddings.embed_documents_calls == 0


def test_incremental_ingestion_updates_modified_file(mock_embeddings, temp_workspace):
    doc1 = os.path.join(temp_workspace["docs"], "doc1.txt")
    with open(doc1, "w", encoding="utf-8") as f:
        f.write("Initial version of the document.")

    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_workspace["index"])
    engine = IngestionEngine(vector_store=store, embeddings=mock_embeddings)

    engine.sync_documents(file_paths=[doc1])

    # Modify the file content
    with open(doc1, "w", encoding="utf-8") as f:
        f.write("Modified version with updated text and instructions.")

    stats = engine.sync_documents(file_paths=[doc1])
    assert stats.modified_files == 1
    assert stats.updated_chunks > 0 or stats.added_chunks > 0
    assert stats.unchanged_files == 0


def test_incremental_ingestion_removes_deleted_file(mock_embeddings, temp_workspace):
    doc1 = os.path.join(temp_workspace["docs"], "doc1.txt")
    doc2 = os.path.join(temp_workspace["docs"], "doc2.txt")
    with open(doc1, "w", encoding="utf-8") as f:
        f.write("First document.")
    with open(doc2, "w", encoding="utf-8") as f:
        f.write("Second document to be deleted.")

    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_workspace["index"])
    engine = IngestionEngine(vector_store=store, embeddings=mock_embeddings)

    engine.sync_documents(file_paths=[doc1, doc2])

    # Now remove doc2 from disk and sync with only doc1
    os.remove(doc2)
    stats = engine.sync_documents(file_paths=[doc1])

    assert stats.deleted_files == 1
    assert stats.deleted_chunks > 0


def test_dry_run_does_not_modify_storage(mock_embeddings, temp_workspace):
    doc1 = os.path.join(temp_workspace["docs"], "doc1.txt")
    with open(doc1, "w", encoding="utf-8") as f:
        f.write("Sample document for dry run test.")

    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_workspace["index"])
    engine = IngestionEngine(vector_store=store, embeddings=mock_embeddings)

    stats = engine.sync_documents(file_paths=[doc1], dry_run=True)

    assert stats.new_files == 1
    assert stats.added_chunks > 0
    # Store must remain empty because dry_run=True
    assert store.count() == 0


def test_reindex_clears_and_rebuilds(mock_embeddings, temp_workspace):
    doc1 = os.path.join(temp_workspace["docs"], "doc1.txt")
    with open(doc1, "w", encoding="utf-8") as f:
        f.write("Sample document for reindex test.")

    store = FAISSVectorStore(embeddings=mock_embeddings, index_path=temp_workspace["index"])
    engine = IngestionEngine(vector_store=store, embeddings=mock_embeddings)

    engine.sync_documents(file_paths=[doc1])
    assert store.count() > 0

    # Reindex should clear and re-add
    stats = engine.sync_documents(file_paths=[doc1], reindex=True)
    assert stats.new_files == 1
    assert store.count() > 0
