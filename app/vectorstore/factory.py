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

"""Vector store factory for Sugar-AI."""

import logging
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.vectorstore.base import (
    BaseVectorStore,
    VectorStoreConfigurationError,
)
from app.vectorstore.milvus import MilvusVectorStore
from app.vectorstore.faiss import FAISSVectorStore

logger = logging.getLogger("sugar-ai.vectorstore")


def get_default_embeddings(model_name: Optional[str] = None) -> Embeddings:
    """Instantiate the default HuggingFace embedding model."""
    name = model_name or getattr(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=name)


def get_vector_store(
    backend: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
    **kwargs,
) -> BaseVectorStore:
    """Create a vector store instance according to application configuration.

    Args:
        backend: Backend name ("milvus" or "faiss"). Defaults to settings.VECTOR_STORE_BACKEND.
        embeddings: Embeddings instance. Defaults to HuggingFaceEmbeddings with configured model.
        **kwargs: Additional overrides passed to the vector store constructor.

    Returns:
        An instance of BaseVectorStore.

    Raises:
        VectorStoreConfigurationError: If the configured backend is not recognized.
    """
    selected_backend = (backend or getattr(settings, "VECTOR_STORE_BACKEND", "faiss")).lower().strip()

    if embeddings is None:
        model_name = getattr(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        embeddings = get_default_embeddings(model_name=model_name)

    embedding_dim = kwargs.pop("embedding_dimension", 384)

    if selected_backend == "milvus":
        uri = kwargs.pop("uri", getattr(settings, "MILVUS_URI", "http://localhost:19530"))
        collection_name = kwargs.pop(
            "collection_name",
            getattr(settings, "MILVUS_COLLECTION_NAME", "sugar_ai_docs"),
        )
        token = kwargs.pop("token", getattr(settings, "MILVUS_TOKEN", None))
        user = kwargs.pop("user", getattr(settings, "MILVUS_USER", None))
        password = kwargs.pop("password", getattr(settings, "MILVUS_PASSWORD", None))
        db_name = kwargs.pop("db_name", getattr(settings, "MILVUS_DB_NAME", "default"))
        metric_type = kwargs.pop("metric_type", getattr(settings, "MILVUS_METRIC_TYPE", "COSINE"))
        index_type = kwargs.pop("index_type", getattr(settings, "MILVUS_INDEX_TYPE", "AUTOINDEX"))

        return MilvusVectorStore(
            embeddings=embeddings,
            uri=uri,
            collection_name=collection_name,
            token=token,
            user=user,
            password=password,
            db_name=db_name,
            metric_type=metric_type,
            index_type=index_type,
            embedding_dimension=embedding_dim,
            **kwargs,
        )

    if selected_backend == "faiss":
        index_path = kwargs.pop(
            "index_path",
            getattr(settings, "FAISS_INDEX_PATH", None) or getattr(settings, "VECTOR_STORE_PATH", "./data/vectorstore"),
        )
        model_name = getattr(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        return FAISSVectorStore(
            embeddings=embeddings,
            index_path=index_path,
            embedding_model_name=model_name,
            embedding_dimension=embedding_dim,
            **kwargs,
        )

    raise VectorStoreConfigurationError(
        f"Unsupported VECTOR_STORE_BACKEND: '{selected_backend}'. "
        "Valid choices are 'milvus' and 'faiss'."
    )
