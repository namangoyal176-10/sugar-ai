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

"""Tests for application startup, shutdown, and retriever lifecycle."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.documents import Document

from app.ai import RAGAgent
from app.vectorstore.base import (
    CollectionNotFoundError,
    VectorStoreConnectionError,
)


class DummyProvider:
    def __init__(self, name="dummy-model"):
        self.name = name

    def get_model_name(self):
        return self.name

    async def generate(self, prompt, params=None):
        return "Child-friendly answer: Here is a clear, kid-friendly answer."

    async def close(self):
        pass


@pytest.fixture
def dummy_agent():
    return RAGAgent(provider=DummyProvider())


def test_startup_loads_existing_vectorstore(dummy_agent):
    mock_store = Mock()
    mock_store.load.return_value = True
    mock_retriever = Mock()
    mock_store.as_retriever.return_value = mock_retriever

    with patch("app.ai.get_vector_store", return_value=mock_store):
        retriever = dummy_agent.load_vectorstore()

    assert retriever is mock_retriever
    assert dummy_agent.retriever is mock_retriever
    assert dummy_agent.vector_store is mock_store
    mock_store.load.assert_called_once()


def test_startup_handles_missing_collection_gracefully(dummy_agent):
    mock_store = Mock()
    mock_store.load.side_effect = CollectionNotFoundError("Collection does not exist")

    with patch("app.ai.get_vector_store", return_value=mock_store):
        retriever = dummy_agent.load_vectorstore()

    # Application should start cleanly with retriever as None
    assert retriever is None
    assert dummy_agent.retriever is None


def test_startup_handles_connection_error_gracefully(dummy_agent):
    mock_store = Mock()
    mock_store.load.side_effect = VectorStoreConnectionError("Connection refused")

    with patch("app.ai.get_vector_store", return_value=mock_store):
        retriever = dummy_agent.load_vectorstore()

    assert retriever is None
    assert dummy_agent.retriever is None


@pytest.mark.asyncio
async def test_rag_agent_runs_without_retriever(dummy_agent):
    """When retriever is None, RAGAgent falls back cleanly to 'No relevant documentation found'."""
    dummy_agent.retriever = None

    response = await dummy_agent.run("What is Sugar Activities?")
    assert response is not None
    assert len(response) > 0


@pytest.mark.asyncio
async def test_rag_agent_runs_with_retriever(dummy_agent):
    """When retriever returns matching documents, context is included in the prompt."""
    mock_retriever = Mock()
    mock_vectorstore = Mock()
    mock_doc = Document(
        page_content="Sugar Activities are educational programs for kids.",
        metadata={"source": "docs/sugar.pdf"},
    )
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.95)]
    mock_retriever.vectorstore = mock_vectorstore
    mock_retriever.search_kwargs = {}

    dummy_agent.retriever = mock_retriever

    doc, score = await dummy_agent.get_relevant_document("Sugar Activities")
    assert doc is mock_doc
    assert score == pytest.approx(0.95)

    response = await dummy_agent.run("What is Sugar Activities?")
    assert response is not None


def test_backward_compatible_setup_vectorstore(dummy_agent):
    """setup_vectorstore still works for legacy callers."""
    mock_store = Mock()
    mock_retriever = Mock()
    mock_store.as_retriever.return_value = mock_retriever

    with patch("app.ai.get_vector_store", return_value=mock_store), \
         patch("app.ingest.IngestionEngine.sync_documents"):
        retriever = dummy_agent.setup_vectorstore(["docs/test.pdf"])

    assert retriever is mock_retriever
    assert dummy_agent.retriever is mock_retriever
