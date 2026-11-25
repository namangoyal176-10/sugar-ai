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

"""Base interfaces and exceptions for Sugar-AI vector storage."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class VectorStoreError(Exception):
    """Base exception for all vector store operations."""
    pass


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when configuration for a vector store backend is invalid or missing."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when the requested collection or index does not exist or is empty."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when connection to the vector store service fails."""
    pass


class DimensionMismatchError(VectorStoreError):
    """Raised when the collection dimension does not match the embedding dimension."""
    pass


class EmbeddingModelMismatchError(VectorStoreError):
    """Raised when indexed embeddings were created with a different embedding model."""
    pass


class BaseVectorStore(VectorStore, ABC):
    """Abstract base class for persistent vector stores in Sugar-AI."""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings

    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings instance."""
        return self._embeddings

    @abstractmethod
    def load(self) -> bool:
        """Load an existing collection/index from persistent storage.

        Raises:
            CollectionNotFoundError: If the collection or index does not exist or is empty.
            VectorStoreConnectionError: If connection to backend service fails.
            DimensionMismatchError: If stored vectors dimension does not match current model.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend service is available and ready."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently stored in the collection."""
        pass

    @abstractmethod
    def get_indexed_metadata(self) -> Dict[str, dict]:
        """Return a mapping of chunk_id -> metadata for all indexed chunks.

        Used by incremental ingestion to compare content hashes and identify changes.
        """
        pass

    @abstractmethod
    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert or update documents and their vector embeddings."""
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete specific documents/chunks by their unique IDs."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the collection or drop and recreate it."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any connections, files, or background resources."""
        pass

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a vector store from texts and embeddings."""
        store = cls(embeddings=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add raw texts and metadatas, conforming to VectorStore interface."""
        documents = [
            Document(page_content=t, metadata=metadatas[i] if metadatas and i < len(metadatas) else {})
            for i, t in enumerate(texts)
        ]
        return self.upsert_documents(documents=documents, ids=ids)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete chunks by IDs, conforming to VectorStore interface."""
        if ids:
            self.delete_by_ids(ids)
            return True
        return False
