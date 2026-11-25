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

"""Vector store package for Sugar-AI."""

from app.vectorstore.base import (
    BaseVectorStore,
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbeddingModelMismatchError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from app.vectorstore.factory import get_default_embeddings, get_vector_store
from app.vectorstore.faiss import FAISSVectorStore
from app.vectorstore.milvus import MilvusVectorStore

__all__ = [
    "BaseVectorStore",
    "CollectionNotFoundError",
    "DimensionMismatchError",
    "EmbeddingModelMismatchError",
    "FAISSVectorStore",
    "MilvusVectorStore",
    "VectorStoreConfigurationError",
    "VectorStoreConnectionError",
    "VectorStoreError",
    "get_default_embeddings",
    "get_vector_store",
]
