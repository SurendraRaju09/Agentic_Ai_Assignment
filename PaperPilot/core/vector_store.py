"""
core/vector_store.py — ChromaDB Interface and Vector Store Operations
=====================================================================
PURPOSE:
    Encapsulates ChromaDB collection creation, embedding generation, document storage,
    and similarity retrieval with metadata.
    Implements Requirement 5 (Agentic RAG) and Requirement 7 (Vector Database).

WHY all-MiniLM-L6-v2:
    - 384-dimensional dense semantic vectors
    - Compact (~90 MB) and CPU-efficient
    - High benchmark quality on information retrieval tasks
"""

import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document
from config import settings


class VectorStoreManager:
    """Manages persistent ChromaDB collections for uploaded research papers."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        # Persistent ChromaDB client saves vector indexes on disk
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        # Sentence Transformer embedding function for text-to-vector conversion
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )

    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        """
        ChromaDB collections require 3-63 characters: [a-zA-Z0-9_-], cannot start with '_'.
        """
        clean = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        clean = clean.strip('_')
        if not clean or len(clean) < 3:
            clean = f"paper_{clean}"
        return clean[:60]

    def get_or_create_collection(self, collection_id: str):
        """Returns or creates a named ChromaDB collection with the embedding function."""
        valid_name = self.sanitize_collection_name(collection_id)
        return self.client.get_or_create_collection(
            name=valid_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def store_documents(self, documents: list[Document], collection_id: str) -> int:
        """
        Stores chunks in ChromaDB. Overwrites any previous collection for the same paper
        to prevent stale or duplicate chunk retrieval.
        
        Args:
            documents: List of chunked Document objects.
            collection_id: Unique identifier for the paper (e.g. filename).
            
        Returns:
            int: Number of chunks stored.
        """
        valid_name = self.sanitize_collection_name(collection_id)
        # Reset previous collection if re-indexing the same file
        try:
            self.client.delete_collection(name=valid_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=valid_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{valid_name}_chunk_{i}" for i in range(len(documents))]

        # ChromaDB allows batch additions
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            collection.add(
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

        return len(texts)

    def retrieve_chunks(
        self, query: str, collection_id: str, k: int = 6
    ) -> list[Document]:
        """
        Performs semantic similarity search and returns Document objects with scores.
        
        Args:
            query: The search query text.
            collection_id: The paper collection identifier.
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            list[Document]: Matched chunks with content, metadata, and distance score.
        """
        valid_name = self.sanitize_collection_name(collection_id)
        try:
            collection = self.client.get_collection(
                name=valid_name,
                embedding_function=self.embedding_fn,
            )
        except Exception:
            # Collection does not exist
            return []

        # Cap k to the number of documents in collection
        total_items = collection.count()
        if total_items == 0:
            return []
        effective_k = min(k, total_items)

        results = collection.query(
            query_texts=[query],
            n_results=effective_k,
        )

        retrieved_docs: list[Document] = []
        if results and results["documents"] and results["documents"][0]:
            doc_texts = results["documents"][0]
            doc_metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(doc_texts)
            doc_distances = results["distances"][0] if results["distances"] else [0.0] * len(doc_texts)

            for text, meta, dist in zip(doc_texts, doc_metas, doc_distances):
                # Cosine distance: 0.0 means identical, 1.0 means orthogonal
                meta_copy = dict(meta)
                meta_copy["distance"] = round(dist, 4)
                retrieved_docs.append(Document(page_content=text, metadata=meta_copy))

        return retrieved_docs

    def get_all_documents(self, collection_id: str) -> list[Document]:
        """
        Retrieves ALL indexed chunks for a paper, ordered by original chunk/page sequence.
        Essential for full-paper batched Map-Reduce summarization so no section is omitted.
        """
        valid_name = self.sanitize_collection_name(collection_id)
        try:
            collection = self.client.get_collection(
                name=valid_name,
                embedding_function=self.embedding_fn,
            )
            raw = collection.get(include=["documents", "metadatas"])
            if not raw or not raw.get("documents"):
                return []

            docs = []
            for text, meta in zip(raw["documents"], raw["metadatas"]):
                docs.append(Document(page_content=text, metadata=dict(meta)))

            # Sort by chunk_index so processing strictly respects the paper's original narrative order
            docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
            return docs
        except Exception:
            return []

    def get_collection_stats(self, collection_id: str) -> dict:
        """Returns statistics about an indexed collection."""
        valid_name = self.sanitize_collection_name(collection_id)
        try:
            col = self.client.get_collection(name=valid_name)
            count = col.count()
            return {"exists": True, "chunk_count": count, "collection_name": valid_name}
        except Exception:
            return {"exists": False, "chunk_count": 0, "collection_name": valid_name}

    def delete_collection(self, collection_id: str) -> bool:
        """Deletes a collection."""
        valid_name = self.sanitize_collection_name(collection_id)
        try:
            self.client.delete_collection(name=valid_name)
            return True
        except Exception:
            return False
