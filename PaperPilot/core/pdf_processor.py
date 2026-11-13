"""
core/pdf_processor.py — PDF Text Extraction and Metadata Processing
===================================================================
PURPOSE:
    Handles validation, text extraction, and page-level metadata tracking for research paper PDFs.
    Implements Requirement 5 (PDF processing with page metadata for grounded source tracking).

WHY PyMuPDF:
    PyMuPDF (fitz) is significantly faster and more reliable on multi-column academic layouts
    than pure-python alternatives like PyPDF2 or pdfminer.
"""

from pathlib import Path
import re
from typing import BinaryIO
import pymupdf  # Modern PyMuPDF import
from langchain_core.documents import Document


class PDFProcessingError(Exception):
    """Custom exception raised when a PDF cannot be processed."""
    pass


def clean_text(text: str) -> str:
    """
    Cleans extracted raw PDF text:
    - Fixes hyphenated line-breaks common in two-column papers ('algo-\nrithm' -> 'algorithm')
    - Replaces excessive newlines and tabs with clean spacing
    - Strips non-printable control characters
    """
    if not text:
        return ""
    # Remove hyphenated word breaks at line ends
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Normalize multiple whitespace characters within lines
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize excessive newlines (max 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(file_source: str | Path | BinaryIO, filename: str = "document.pdf") -> list[Document]:
    """
    Extracts text page-by-page from a PDF file path or file-like buffer (e.g., Streamlit UploadedFile).
    
    Args:
        file_source: Path to PDF or BytesIO stream from file upload.
        filename: Original name of the uploaded document.
        
    Returns:
        list[Document]: LangChain Document objects with page content and page metadata.
        
    Raises:
        PDFProcessingError: If the PDF is empty, password-protected, corrupted, or has no readable text.
    """
    try:
        if isinstance(file_source, (str, Path)):
            doc = pymupdf.open(str(file_source))
        else:
            # Handle BytesIO stream from Streamlit
            file_bytes = file_source.read()
            if not file_bytes:
                raise PDFProcessingError("Uploaded PDF file is completely empty (0 bytes).")
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")

        if doc.is_encrypted:
            raise PDFProcessingError("The PDF is password protected and cannot be read.")

        total_pages = doc.page_count
        if total_pages == 0:
            raise PDFProcessingError("The PDF contains 0 pages.")

        documents: list[Document] = []
        total_extracted_chars = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)

            if cleaned:
                total_extracted_chars += len(cleaned)
                documents.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": page_idx + 1,  # 1-indexed for human readability
                            "total_pages": total_pages,
                        }
                    )
                )

        doc.close()

        if total_extracted_chars < 50:
            raise PDFProcessingError(
                "Could not extract sufficient text from the PDF. It may be a scanned image-only PDF without OCR."
            )

        return documents

    except Exception as e:
        if isinstance(e, PDFProcessingError):
            raise
        raise PDFProcessingError(f"Failed to process PDF: {str(e)}") from e
