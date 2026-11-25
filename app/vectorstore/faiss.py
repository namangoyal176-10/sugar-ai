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

"""Persistent FAISS vector store backend with safe local serialization for Sugar-AI."""

import hashlib
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.vectorstore.base import (
    BaseVectorStore,
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbeddingModelMismatchError,
    VectorStoreError,
)

logger = logging.getLogger("sugar-ai.vectorstore.faiss")

MANIFEST_FILENAME = "manifest.json"
INDEX_FAISS = "index.faiss"
INDEX_PKL = "index.pkl"


def _compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class FAISSVectorStore(BaseVectorStore):
    """Persistent local FAISS vector store with integrity verification."""

    def __init__(
        self,
        embeddings: Embeddings,
        index_path: str = "./data/vectorstore",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
    ):
        super().__init__(embeddings)
        self.index_path = os.path.abspath(index_path)
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension
        self._faiss: Optional[FAISS] = None
        self._metadata_map: Dict[str, dict] = {}

    def is_available(self) -> bool:
        """Check if FAISS is available in the current environment."""
        try:
            import faiss
            return True
        except ImportError:
            return False

    def _manifest_path(self) -> str:
        return os.path.join(self.index_path, MANIFEST_FILENAME)

    def _load_manifest(self) -> Optional[dict]:
        manifest_file = self._manifest_path()
        if not os.path.exists(manifest_file):
            return None
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read FAISS manifest: %s", exc)
            return None

    def _write_manifest(self) -> None:
        os.makedirs(self.index_path, exist_ok=True)
        pkl_path = os.path.join(self.index_path, INDEX_PKL)
        faiss_path = os.path.join(self.index_path, INDEX_FAISS)

        manifest = {
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "count": self.count(),
            "index_pkl_sha256": _compute_sha256(pkl_path) if os.path.exists(pkl_path) else None,
            "index_faiss_sha256": _compute_sha256(faiss_path) if os.path.exists(faiss_path) else None,
            "metadata_map": self._metadata_map,
        }
        with open(self._manifest_path(), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def load(self) -> bool:
        """Load an existing persistent FAISS index from disk.

        Performs integrity verification to avoid unsafe deserialization.
        """
        faiss_file = os.path.join(self.index_path, INDEX_FAISS)
        pkl_file = os.path.join(self.index_path, INDEX_PKL)

        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
            raise CollectionNotFoundError(
                f"No persistent FAISS index found at '{self.index_path}'. "
                "Run 'python -m app.ingest' to index documents."
            )

        # Verify integrity using manifest if present
        manifest = self._load_manifest()
        if manifest:
            saved_model = manifest.get("embedding_model")
            if saved_model and saved_model != self.embedding_model_name:
                raise EmbeddingModelMismatchError(
                    f"Index was created with model '{saved_model}', but active model is "
                    f"'{self.embedding_model_name}'. Please reindex with --reindex."
                )

            expected_pkl_hash = manifest.get("index_pkl_sha256")
            if expected_pkl_hash:
                actual_pkl_hash = _compute_sha256(pkl_file)
                if actual_pkl_hash != expected_pkl_hash:
                    raise VectorStoreError(
                        f"FAISS index integrity check failed for '{pkl_file}'. "
                        "The index file may have been modified or corrupted. "
                        "Rebuild the index with 'python -m app.ingest --reindex'."
                    )

            self._metadata_map = manifest.get("metadata_map", {})

        try:
            self._faiss = FAISS.load_local(
                folder_path=self.index_path,
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )

            # Validate dimensions
            dim = getattr(self._faiss.index, "d", None)
            if dim is not None and dim != self.embedding_dimension:
                raise DimensionMismatchError(
                    f"FAISS index dimension ({dim}) does not match expected "
                    f"dimension ({self.embedding_dimension}). Please reindex."
                )

            logger.info(
                "Successfully loaded persistent FAISS index from '%s' with %d chunks",
                self.index_path,
                self.count(),
            )
            return True
        except (CollectionNotFoundError, DimensionMismatchError, EmbeddingModelMismatchError, VectorStoreError):
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to load FAISS index from {self.index_path}: {exc}") from exc

    def count(self) -> int:
        """Return the number of vectors stored in the FAISS index."""
        if self._faiss is None:
            return 0
        return int(self._faiss.index.ntotal)

    def get_indexed_metadata(self) -> Dict[str, dict]:
        """Return indexed chunk metadata mapping."""
        if self._metadata_map:
            return dict(self._metadata_map)

        manifest = self._load_manifest()
        if manifest and "metadata_map" in manifest:
            self._metadata_map = manifest["metadata_map"]
            return dict(self._metadata_map)

        return {}

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert or replace documents in the FAISS index and persist to disk."""
        if not documents:
            return []

        if ids is None:
            ids = [
                doc.metadata.get("chunk_id") or f"chunk_{i}"
                for i, doc in enumerate(documents)
            ]

        # Update metadata tracking
        for doc, chunk_id in zip(documents, ids):
            self._metadata_map[chunk_id] = {
                "id": chunk_id,
                "doc_id": doc.metadata.get("doc_id", ""),
                "source": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
                "page": doc.metadata.get("page", 0),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "content_hash": doc.metadata.get("content_hash", ""),
                "doc_hash": doc.metadata.get("doc_hash", ""),
                "embedding_model": self.embedding_model_name,
            }

        if self._faiss is None:
            # Try loading existing or build fresh
            try:
                self.load()
            except CollectionNotFoundError:
                pass

        if self._faiss is None:
            self._faiss = FAISS.from_documents(
                documents=documents,
                embedding=self._embeddings,
                ids=ids,
            )
        else:
            # FAISS docstore raises an error if an ID already exists.
            # Delete any existing IDs first so upserting succeeds cleanly.
            docstore_dict = getattr(getattr(self._faiss, "docstore", None), "_dict", {})
            existing_overlap = [chunk_id for chunk_id in ids if chunk_id in docstore_dict]
            if existing_overlap:
                try:
                    self._faiss.delete(existing_overlap)
                except Exception as exc:
                    logger.debug("Failed to delete overlapping FAISS ids before update: %s", exc)
            self._faiss.add_documents(documents=documents, ids=ids)

        self.save()
        return ids

    def save(self) -> None:
        """Persist current FAISS index and manifest to disk."""
        if self._faiss is not None:
            os.makedirs(self.index_path, exist_ok=True)
            self._faiss.save_local(self.index_path)
            self._write_manifest()
            logger.info("Saved persistent FAISS index to '%s'", self.index_path)

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete specific vectors by ID and persist."""
        if not ids or self._faiss is None:
            return

        try:
            self._faiss.delete(ids=ids)
            for cid in ids:
                self._metadata_map.pop(cid, None)
            self.save()
        except Exception as exc:
            logger.warning("Error deleting vectors from FAISS: %s", exc)

    def clear(self) -> None:
        """Clear all indexed documents and delete index files from disk."""
        self._faiss = None
        self._metadata_map.clear()
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
            logger.info("Cleared FAISS index at '%s'", self.index_path)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search nearest neighbors with distance scores."""
        if self._faiss is None:
            return []
        return self._faiss.similarity_search_with_score(query, k=k, **kwargs)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search nearest neighbors with relevance scores."""
        if self._faiss is None:
            return []
        return self._faiss.similarity_search_with_relevance_scores(query, k=k, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search nearest documents without scores."""
        if self._faiss is None:
            return []
        return self._faiss.similarity_search(query, k=k, **kwargs)

    def close(self) -> None:
        """Release in-memory references."""
        self._faiss = None
