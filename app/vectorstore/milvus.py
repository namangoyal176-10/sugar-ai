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

"""Milvus persistent vector store backend for Sugar-AI."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.vectorstore.base import (
    BaseVectorStore,
    CollectionNotFoundError,
    DimensionMismatchError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
)

logger = logging.getLogger("sugar-ai.vectorstore.milvus")


class MilvusVectorStore(BaseVectorStore):
    """Persistent vector store using Milvus or Milvus Lite."""

    def __init__(
        self,
        embeddings: Embeddings,
        uri: str = "http://localhost:19530",
        collection_name: str = "sugar_ai_docs",
        token: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db_name: str = "default",
        metric_type: str = "COSINE",
        index_type: str = "AUTOINDEX",
        embedding_dimension: int = 384,
    ):
        super().__init__(embeddings)
        self.uri = uri
        self.collection_name = collection_name
        self.token = token
        self.user = user
        self.password = password
        self.db_name = db_name
        self.metric_type = metric_type.upper()
        self.index_type = index_type
        self.embedding_dimension = embedding_dimension
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or initialize the MilvusClient."""
        if self._client is not None:
            return self._client

        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise VectorStoreConfigurationError(
                "pymilvus is required for MilvusVectorStore. "
                "Install it with: pip install pymilvus"
            ) from exc

        client_kwargs: Dict[str, Any] = {"uri": self.uri}
        if self.token:
            client_kwargs["token"] = self.token
        elif self.user and self.password:
            client_kwargs["user"] = self.user
            client_kwargs["password"] = self.password
        if self.db_name:
            client_kwargs["db_name"] = self.db_name

        try:
            self._client = MilvusClient(**client_kwargs)
            return self._client
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to connect to Milvus at {self.uri}: {str(exc)}"
            ) from exc

    def is_available(self) -> bool:
        """Check if Milvus is reachable."""
        try:
            client = self._get_client()
            return client is not None
        except Exception:
            return False

    def _validate_collection_schema(self, client: Any) -> None:
        """Validate collection dimension and schema against current settings."""
        desc = client.describe_collection(self.collection_name)
        fields = desc.get("fields", [])
        for field in fields:
            if field.get("name") == "vector" or "FLOAT_VECTOR" in str(field.get("type")):
                dim = field.get("params", {}).get("dim")
                if dim is not None and dim != self.embedding_dimension:
                    raise DimensionMismatchError(
                        f"Milvus collection '{self.collection_name}' vector dimension "
                        f"({dim}) does not match expected embedding dimension "
                        f"({self.embedding_dimension}). Please reindex with --reindex."
                    )

    def load(self) -> bool:
        """Load an existing collection for retrieval during application startup.

        Raises:
            CollectionNotFoundError: If the collection does not exist or has 0 entities.
            DimensionMismatchError: If the vector dimension does not match.
            VectorStoreConnectionError: If connection fails.
        """
        client = self._get_client()
        try:
            if not client.has_collection(self.collection_name):
                raise CollectionNotFoundError(
                    f"Milvus collection '{self.collection_name}' does not exist on {self.uri}. "
                    "Run 'python -m app.ingest' to ingest documents before starting."
                )

            self._validate_collection_schema(client)

            # Check if collection has entities
            row_count = self.count()
            if row_count == 0:
                raise CollectionNotFoundError(
                    f"Milvus collection '{self.collection_name}' exists on {self.uri} "
                    "but contains 0 documents. Run 'python -m app.ingest' to index documents."
                )

            client.load_collection(self.collection_name)
            logger.info(
                "Successfully loaded Milvus collection '%s' with %d document chunks from %s",
                self.collection_name,
                row_count,
                self.uri,
            )
            return True
        except (CollectionNotFoundError, DimensionMismatchError, VectorStoreConnectionError):
            raise
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Error loading Milvus collection '{self.collection_name}': {str(exc)}"
            ) from exc

    def count(self) -> int:
        """Return the number of entities in the collection."""
        client = self._get_client()
        try:
            if not client.has_collection(self.collection_name):
                return 0
            stats = client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def ensure_collection(self) -> None:
        """Ensure collection exists with appropriate schema, creating it if needed."""
        client = self._get_client()
        if client.has_collection(self.collection_name):
            self._validate_collection_schema(client)
            return

        logger.info(
            "Creating Milvus collection '%s' (dim=%d, metric=%s)",
            self.collection_name,
            self.embedding_dimension,
            self.metric_type,
        )
        client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dimension,
            metric_type=self.metric_type,
            id_type="string",
            max_length=128,
            auto_id=False,
            enable_dynamic_field=True,
        )

    def get_indexed_metadata(self) -> Dict[str, dict]:
        """Fetch all indexed chunk metadata from Milvus for incremental sync."""
        client = self._get_client()
        if not client.has_collection(self.collection_name):
            return {}

        try:
            client.load_collection(self.collection_name)
            results = client.query(
                collection_name=self.collection_name,
                filter='id != ""',
                output_fields=[
                    "id",
                    "doc_id",
                    "doc_hash",
                    "content_hash",
                    "source",
                    "title",
                    "page",
                    "chunk_index",
                    "embedding_model",
                ],
                limit=16384,
            )
            metadata_map = {}
            for item in results:
                cid = item.get("id")
                if cid:
                    metadata_map[cid] = item
            return metadata_map
        except Exception as exc:
            logger.warning("Failed to query indexed metadata from Milvus: %s", exc)
            return {}

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert or update documents and their vector representations in Milvus."""
        if not documents:
            return []

        self.ensure_collection()
        client = self._get_client()

        if embeddings is None:
            texts = [doc.page_content for doc in documents]
            embeddings = self._embeddings.embed_documents(texts)

        if ids is None:
            ids = [
                doc.metadata.get("chunk_id") or f"chunk_{i}"
                for i, doc in enumerate(documents)
            ]

        data_rows: List[Dict[str, Any]] = []
        for i, (doc, emb, chunk_id) in enumerate(zip(documents, embeddings, ids)):
            row: Dict[str, Any] = {
                "id": chunk_id,
                "vector": emb,
                "text": doc.page_content,
                "doc_id": doc.metadata.get("doc_id", ""),
                "source": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
                "page": int(doc.metadata.get("page", 0)) if doc.metadata.get("page") is not None else 0,
                "chunk_index": int(doc.metadata.get("chunk_index", i)),
                "content_hash": doc.metadata.get("content_hash", ""),
                "doc_hash": doc.metadata.get("doc_hash", ""),
                "embedding_model": doc.metadata.get("embedding_model", ""),
            }
            # Include any other custom metadata fields
            for k, v in doc.metadata.items():
                if k not in row:
                    row[k] = v
            data_rows.append(row)

        # Batch upsert to Milvus
        batch_size = 100
        for b in range(0, len(data_rows), batch_size):
            client.upsert(
                collection_name=self.collection_name,
                data=data_rows[b : b + batch_size],
            )

        client.load_collection(self.collection_name)
        return ids

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete document chunks by their IDs."""
        if not ids:
            return
        client = self._get_client()
        if client.has_collection(self.collection_name):
            client.delete(collection_name=self.collection_name, ids=ids)

    def clear(self) -> None:
        """Drop collection to allow a complete reindex."""
        client = self._get_client()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)
            logger.info("Cleared Milvus collection '%s'", self.collection_name)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search Milvus for nearest neighbors to the query vector."""
        client = self._get_client()
        query_vector = self._embeddings.embed_query(query)

        search_kwargs: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": k,
            "output_fields": [
                "text",
                "source",
                "title",
                "page",
                "chunk_index",
                "doc_id",
                "content_hash",
                "doc_hash",
                "embedding_model",
            ],
        }
        if filter:
            search_kwargs["filter"] = filter

        raw_results = client.search(**search_kwargs)
        results: List[Tuple[Document, float]] = []

        if raw_results and len(raw_results) > 0:
            for hit in raw_results[0]:
                distance = hit.get("distance", 0.0)
                entity = hit.get("entity", {})
                text = entity.get("text", "")

                metadata = {k: v for k, v in entity.items() if k != "text"}
                metadata["chunk_id"] = hit.get("id")

                results.append((Document(page_content=text, metadata=metadata), float(distance)))

        return results

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search Milvus with normalized relevance scores in [0.0, 1.0]."""
        results_with_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        relevance_results: List[Tuple[Document, float]] = []

        for doc, score in results_with_scores:
            if self.metric_type == "COSINE":
                # Cosine similarity is in [-1, 1], normalize to [0, 1]
                norm_score = max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score))
            elif self.metric_type == "L2":
                # L2 distance is >= 0, convert to similarity
                norm_score = 1.0 / (1.0 + score)
            else:
                # IP or other metric
                norm_score = max(0.0, min(1.0, score))
            relevance_results.append((doc, norm_score))

        return relevance_results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return nearest documents without scores."""
        scored = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in scored]

    def close(self) -> None:
        """Close connection to Milvus."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.debug("Error while closing Milvus client: %s", exc)
            self._client = None
