"""
tests/test_agent_live.py — Live End-to-End Test for SummarizerAgent + Ollama
=============================================================================
PURPOSE:
    Tests the real LLM inference with qwen2.5-coder:7b over the ChromaDB collection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.pdf_processor import extract_text_from_pdf
from core.chunker import chunk_documents
from core.vector_store import VectorStoreManager
from agents.summarizer_agent import SummarizerAgent
from tests.test_backend import create_sample_academic_pdf

def test_live_summarizer():
    print("Creating sample paper...")
    pdf_path = PROJECT_ROOT / "data" / "sample_biofast.pdf"
    create_sample_academic_pdf(str(pdf_path))

    print("Extracting & Indexing into ChromaDB...")
    docs = extract_text_from_pdf(pdf_path, filename="sample_biofast.pdf")
    chunks = chunk_documents(docs)
    
    vstore = VectorStoreManager()
    vstore.store_documents(chunks, collection_id="sample_biofast")

    print("Running SummarizerAgent with qwen2.5-coder:7b via LCEL...")
    agent = SummarizerAgent(vector_store=vstore)
    result = agent.run(collection_id="sample_biofast")

    print("\n--- SUMMARIZER RESULT ---")
    print(result["summary"])
    print("------------------------")
    print(f"Source Pages: {result['source_pages']}")
    print(f"Context Quality: {result['quality']}")
    print(f"Method: {result['method']}")

    assert len(result["summary"]) > 100
    print("\n[LIVE AGENT TEST PASSED]")

if __name__ == "__main__":
    test_live_summarizer()
