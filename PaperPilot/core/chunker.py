"""
core/chunker.py — Intelligent Text Chunking with Metadata Preservation
=====================================================================
PURPOSE:
    Splits long research documents into manageable, overlapping chunks.
    Implements Requirement 5 (Manageable text chunks preserving page-level source tracking).

WHY OVERLAP MATTERS:
    Academic concepts and formulas often cross arbitrary character boundaries.
    Overlapping chunks ensure semantic context is not fractured between neighboring windows.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings


def chunk_documents(documents: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Document]:
    """
    Splits a list of page-level Document objects into smaller overlapping chunks.
    Preserves all page-level metadata and appends chunk-level index metadata.
    
    Args:
        documents: List of Document objects from pdf_processor.py.
        chunk_size: Maximum characters per chunk (defaults to settings.CHUNK_SIZE).
        chunk_overlap: Overlapping characters between adjacent chunks (defaults to settings.CHUNK_OVERLAP).
        
    Returns:
        list[Document]: Chunked Document objects with updated metadata.
    """
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    # Recursive splitter tries paragraph breaks, then line breaks, then sentence endings, then spaces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunked_docs = text_splitter.split_documents(documents)

    # Enrich metadata with unique chunk identifier and relative chunk index
    for idx, doc in enumerate(chunked_docs):
        doc.metadata["chunk_index"] = idx
        doc.metadata["total_chunks"] = len(chunked_docs)

    return chunked_docs
