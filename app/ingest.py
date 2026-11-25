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

"""Document ingestion and incremental refresh workflow for Sugar-AI.

Supports:
- Initial document indexing
- Incremental indexing of new and modified documents
- Deletion of obsolete document chunks
- Content hashing and stable chunk identifiers
- Safe reindexing when requested
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore.base import (
    BaseVectorStore,
    EmbeddingModelMismatchError,
    VectorStoreError,
)
from app.vectorstore.factory import get_default_embeddings, get_vector_store

logger = logging.getLogger("sugar-ai.ingest")


def compute_file_hash(filepath: str) -> str:
    """Compute the SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_doc_id(filepath: str) -> str:
    """Generate a clean, stable document identifier from its file path."""
    clean_name = os.path.basename(filepath)
    clean_name = os.path.splitext(clean_name)[0]
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]+", "_", clean_name).strip("_").lower()
    return sanitized or "doc"


def extract_document_title(filepath: str) -> str:
    """Derive human-readable document title."""
    base = os.path.basename(filepath)
    title = os.path.splitext(base)[0]
    return title.replace("_", " ").replace("-", " ").strip()


@dataclass
class IngestionStats:
    """Statistics recorded during document ingestion."""
    total_files: int = 0
    new_files: int = 0
    modified_files: int = 0
    unchanged_files: int = 0
    deleted_files: int = 0
    total_chunks: int = 0
    added_chunks: int = 0
    updated_chunks: int = 0
    deleted_chunks: int = 0
    skipped_chunks: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Ingestion completed in {self.duration_seconds:.2f}s: "
            f"{self.total_files} file(s) processed "
            f"({self.new_files} new, {self.modified_files} modified, "
            f"{self.unchanged_files} unchanged, {self.deleted_files} deleted). "
            f"Chunks: {self.added_chunks} added, {self.updated_chunks} updated, "
            f"{self.deleted_chunks} deleted, {self.skipped_chunks} skipped."
        )


class IngestionEngine:
    """Orchestrates document loading, chunking, hashing, and incremental vector synchronization."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: Optional[Embeddings] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        embedding_model_name: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings or vector_store.embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = (
            embedding_model_name
            or getattr(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_file(self, filepath: str) -> List[Document]:
        """Load a single document from disk and attach initial metadata."""
        if not os.path.exists(filepath):
            logger.warning("Document file not found: %s", filepath)
            return []

        doc_id = generate_doc_id(filepath)
        title = extract_document_title(filepath)
        file_hash = compute_file_hash(filepath)

        if filepath.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
            raw_docs = loader.load()
        else:
            loader = TextLoader(filepath, encoding="utf-8")
            raw_docs = loader.load()

        enriched_docs = []
        for i, doc in enumerate(raw_docs):
            page_num = doc.metadata.get("page", i + 1)
            doc.metadata.update({
                "source": filepath,
                "doc_id": doc_id,
                "title": title,
                "page": int(page_num) if page_num is not None else 0,
                "doc_hash": file_hash,
                "embedding_model": self.embedding_model_name,
            })
            enriched_docs.append(doc)

        return enriched_docs

    def chunk_documents(self, documents: List[Document], doc_id: str, file_hash: str) -> List[Document]:
        """Split documents into semantic chunks with stable chunk IDs and content hashes."""
        chunks = self.splitter.split_documents(documents)
        enriched_chunks = []

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.page_content.strip()
            if not chunk_text:
                continue

            content_hash = compute_content_hash(chunk_text)
            page = chunk.metadata.get("page", 0)
            chunk_id = f"{doc_id}_p{page}_c{idx}"

            chunk.metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "content_hash": content_hash,
                "doc_id": doc_id,
                "doc_hash": file_hash,
                "embedding_model": self.embedding_model_name,
            })
            enriched_chunks.append(chunk)

        return enriched_chunks

    def sync_documents(
        self,
        file_paths: List[str],
        reindex: bool = False,
        dry_run: bool = False,
    ) -> IngestionStats:
        """Synchronize documents with the persistent vector store incrementally.

        Args:
            file_paths: List of file paths to index.
            reindex: If True, clears existing index and performs full reindex.
            dry_run: If True, calculates and logs planned changes without modifying storage.

        Returns:
            IngestionStats summarizing the operations performed.
        """
        start_time = time.perf_counter()
        stats = IngestionStats()

        existing_files = [p for p in file_paths if os.path.exists(p)]
        stats.total_files = len(existing_files)

        if reindex:
            logger.info("Full reindex requested. Clearing existing vector store...")
            if not dry_run:
                self.vector_store.clear()
            existing_metadata: Dict[str, dict] = {}
        else:
            existing_metadata = self.vector_store.get_indexed_metadata()

        # Group existing metadata by doc_id
        existing_docs: Dict[str, Dict[str, dict]] = {}
        for chunk_id, meta in existing_metadata.items():
            doc_id = meta.get("doc_id") or ""
            if doc_id not in existing_docs:
                existing_docs[doc_id] = {}
            existing_docs[doc_id][chunk_id] = meta

        active_doc_ids: Set[str] = set()

        for filepath in existing_files:
            doc_id = generate_doc_id(filepath)
            active_doc_ids.add(doc_id)
            current_file_hash = compute_file_hash(filepath)

            existing_chunks_for_doc = existing_docs.get(doc_id, {})

            # Check if file is completely unchanged
            if existing_chunks_for_doc and not reindex:
                # Check stored file hash and embedding model
                sample_meta = next(iter(existing_chunks_for_doc.values()))
                stored_file_hash = sample_meta.get("doc_hash")
                stored_model = sample_meta.get("embedding_model")

                if stored_model and stored_model != self.embedding_model_name:
                    logger.warning(
                        "File '%s' was indexed with model '%s', but current model is '%s'. "
                        "Reindexing file...",
                        filepath,
                        stored_model,
                        self.embedding_model_name,
                    )
                elif stored_file_hash == current_file_hash:
                    # Unchanged file: skip document parsing and embedding entirely
                    stats.unchanged_files += 1
                    stats.skipped_chunks += len(existing_chunks_for_doc)
                    stats.total_chunks += len(existing_chunks_for_doc)
                    logger.debug("File '%s' is unchanged. Skipping re-embedding.", filepath)
                    continue

            # File is either new or modified
            is_new = len(existing_chunks_for_doc) == 0
            if is_new:
                stats.new_files += 1
                logger.info("Indexing new file: %s", filepath)
            else:
                stats.modified_files += 1
                logger.info("Refreshing modified file: %s", filepath)

            # Load and chunk document
            documents = self.load_file(filepath)
            chunks = self.chunk_documents(documents, doc_id, current_file_hash)
            stats.total_chunks += len(chunks)

            # Determine chunks to add, update, or delete
            old_chunk_ids = set(existing_chunks_for_doc.keys())
            new_chunk_ids = {c.metadata["chunk_id"] for c in chunks}

            chunks_to_delete = old_chunk_ids - new_chunk_ids
            if chunks_to_delete:
                stats.deleted_chunks += len(chunks_to_delete)
                if not dry_run:
                    self.vector_store.delete_by_ids(list(chunks_to_delete))

            # Filter chunks that need embedding
            chunks_to_upsert: List[Document] = []
            for chunk in chunks:
                cid = chunk.metadata["chunk_id"]
                content_hash = chunk.metadata["content_hash"]
                existing_chunk_meta = existing_chunks_for_doc.get(cid)

                if (
                    existing_chunk_meta
                    and existing_chunk_meta.get("content_hash") == content_hash
                    and existing_chunk_meta.get("embedding_model") == self.embedding_model_name
                ):
                    stats.skipped_chunks += 1
                else:
                    if existing_chunk_meta:
                        stats.updated_chunks += 1
                    else:
                        stats.added_chunks += 1
                    chunks_to_upsert.append(chunk)

            if chunks_to_upsert and not dry_run:
                chunk_ids = [c.metadata["chunk_id"] for c in chunks_to_upsert]
                self.vector_store.upsert_documents(documents=chunks_to_upsert, ids=chunk_ids)

        # Handle deleted documents (files present in index but no longer in DOC_PATHS)
        if not reindex:
            for indexed_doc_id, chunks_map in existing_docs.items():
                if indexed_doc_id not in active_doc_ids:
                    stats.deleted_files += 1
                    chunk_ids_to_remove = list(chunks_map.keys())
                    stats.deleted_chunks += len(chunk_ids_to_remove)
                    logger.info(
                        "Document '%s' was deleted or removed from configuration. Removing %d chunks.",
                        indexed_doc_id,
                        len(chunk_ids_to_remove),
                    )
                    if not dry_run:
                        self.vector_store.delete_by_ids(chunk_ids_to_remove)

        stats.duration_seconds = time.perf_counter() - start_time
        logger.info(stats.summary())
        return stats


def run_ingest(
    paths: Optional[List[str]] = None,
    backend: Optional[str] = None,
    reindex: bool = False,
    dry_run: bool = False,
) -> IngestionStats:
    """Entry point for document ingestion workflow."""
    doc_paths = paths or settings.DOC_PATHS
    if not doc_paths:
        logger.warning("No DOC_PATHS configured. Nothing to ingest.")
        return IngestionStats()

    selected_backend = backend or settings.VECTOR_STORE_BACKEND
    logger.info("Initializing vector store backend '%s' for ingestion...", selected_backend)

    embeddings = get_default_embeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    vector_store = get_vector_store(backend=selected_backend, embeddings=embeddings)

    engine = IngestionEngine(
        vector_store=vector_store,
        embeddings=embeddings,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        embedding_model_name=settings.EMBEDDING_MODEL_NAME,
    )

    try:
        stats = engine.sync_documents(file_paths=doc_paths, reindex=reindex, dry_run=dry_run)
        return stats
    finally:
        vector_store.close()


def main():
    """CLI interface for Sugar-AI document ingestion."""
    parser = argparse.ArgumentParser(
        description="Sugar-AI Document Ingestion and Refresh Tool"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Perform a full reindex (clears existing vector collection before indexing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check and display changes without writing to the vector store.",
    )
    parser.add_argument(
        "--backend",
        choices=["milvus", "faiss"],
        default=None,
        help="Override configured VECTOR_STORE_BACKEND ('milvus' or 'faiss').",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="Specify document file paths to index instead of DOC_PATHS from configuration.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = run_ingest(
        paths=args.paths,
        backend=args.backend,
        reindex=args.reindex,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("Sugar-AI Document Ingestion Results")
    print("=" * 60)
    print(stats.summary())
    print("=" * 60)


if __name__ == "__main__":
    main()
