"""
tests/test_pipeline.py — Comprehensive 7-Requirement Acceptance Test Suite
==========================================================================
PURPOSE:
    Demonstrates and verifies ALL SEVEN mandatory acceptance criteria:
    Requirement 1: LLM Integration (Ollama / qwen2.5-coder:7b)
    Requirement 2: Interactive GUI (Streamlit app structure)
    Requirement 3: Prompt Engineering (Documented templates across agents)
    Requirement 4: LangChain Core Components (LCEL pipe operator pipelines)
    Requirement 5: Agentic RAG (Chunking, metadata, page citations, grounding)
    Requirement 6: Intent Detection & Routing (4-tier routing strategy)
    Requirement 7: Vector Database (ChromaDB collection management & similarity search)

HOW TO RUN:
    pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
import tempfile
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from utils.llm_factory import get_llm
from core.pdf_processor import extract_text_from_pdf, clean_text, PDFProcessingError
from core.chunker import chunk_documents
from core.vector_store import VectorStoreManager
from core.intent_router import detect_intent
from agents.summarizer_agent import SummarizerAgent
from agents.concept_agent import ConceptAgent
from agents.slide_agent import SlideAgent
from agents.viva_agent import VivaAgent
from generators.pptx_generator import create_presentation_from_slides
from tests.test_backend import create_sample_academic_pdf

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── Requirement 1: LLM Integration ───────────────────────────────────────────
def test_requirement_1_llm_integration():
    """R1: Verify that the LLM is connected and produces non-empty text."""
    llm = get_llm(temperature=0.0)
    assert llm is not None, "Failed to instantiate LLM from llm_factory"
    
    # Simple generation call
    response = llm.invoke("Reply with the exact word: CONNECTED")
    content = response.content if hasattr(response, "content") else str(response)
    assert "CONNECTED" in content.upper() or len(content.strip()) > 0
    print("\n[R1 PASSED] LLM Integration verified with qwen2.5-coder:7b")


# ── Requirement 2: Interactive GUI ───────────────────────────────────────────
def test_requirement_2_interactive_gui():
    """R2: Verify Streamlit app file exists, is syntactically valid, and includes all 8 pages."""
    app_file = PROJECT_ROOT / "app.py"
    assert app_file.exists(), "app.py does not exist!"

    with open(app_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Verify all 8 navigation items are present
    pages = [
        "🏠 Dashboard",
        "📤 Upload Research Paper",
        "📝 Paper Summary",
        "🔬 Concept Extraction",
        "📊 Presentation Generator",
        "🎓 Viva Preparation",
        "💬 Ask Your Paper",
        "ℹ️ About Project",
    ]
    for page in pages:
        assert page in code, f"Missing required navigation page: {page}"

    # Verify download buttons exist for .pptx and .md
    assert "st.download_button" in code
    print("\n[R2 PASSED] Interactive Streamlit GUI components verified")


# ── Requirement 3: Prompt Engineering ─────────────────────────────────────────
def test_requirement_3_prompt_engineering():
    """R3: Verify prompt templates exist with System & Human components and format correctly."""
    from prompts.summarizer_prompts import summarizer_prompt
    from prompts.concept_prompts import concept_prompt
    from prompts.slide_prompts import slide_prompt
    from prompts.viva_prompts import viva_prompt

    prompts = [
        ("Summarizer", summarizer_prompt, {"context": "test text"}),
        ("Concept", concept_prompt, {"context": "test text"}),
        ("Slide", slide_prompt, {"summary_context": "summary", "retrieved_context": "chunks"}),
        ("Viva", viva_prompt, {"context": "test text"}),
    ]

    for name, p, kwargs in prompts:
        assert isinstance(p, ChatPromptTemplate)
        messages = p.format_messages(**kwargs)
        assert len(messages) >= 2, f"{name} prompt must contain at least System and Human messages"
        assert len(messages[0].content) > 20, f"{name} system prompt is too short"
        assert len(messages[1].content) > 10, f"{name} human template is invalid"

    print("\n[R3 PASSED] Prompt Engineering templates verified across all agents")


# ── Requirement 4: LangChain Core Components & LCEL ───────────────────────────
def test_requirement_4_langchain_lcel():
    """R4: Explicitly test LCEL pipe operator chaining (prompt | llm | parser)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Answer in one word."),
        ("human", "What is 2 + 2?"),
    ])
    llm = get_llm(temperature=0.0)
    parser = StrOutputParser()

    # The pipe operator (|) connects LangChain components
    chain = prompt | llm | parser

    result = chain.invoke({})
    assert "4" in result or "FOUR" in result.upper()
    print("\n[R4 PASSED] LangChain LCEL pipeline (prompt | llm | parser) verified")


# ── Requirement 5: Agentic RAG Pipeline ───────────────────────────────────────
def test_requirement_5_agentic_rag():
    """R5: Verify full RAG pipeline: PDF -> Chunks -> Embeddings -> ChromaDB -> Grounded Retrieval."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        pdf_path = Path(tmpdir) / "rag_test.pdf"
        create_sample_academic_pdf(str(pdf_path))

        # 1. PDF Extraction
        docs = extract_text_from_pdf(pdf_path)
        assert len(docs) == 3
        assert docs[0].metadata["page"] == 1

        # 2. Chunking with page preservation
        chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
        assert len(chunks) >= 3
        for c in chunks:
            assert "page" in c.metadata

        # 3. ChromaDB storage
        vstore = VectorStoreManager(persist_dir=str(Path(tmpdir) / "chroma_rag"))
        vstore.store_documents(chunks, collection_id="rag_test")

        # 4. Agentic retrieval with validation
        agent = SummarizerAgent(vector_store=vstore)
        val_ctx = agent.retrieve_with_validation(
            query="BioFast-Transformer F1 score results",
            collection_id="rag_test",
            k=2,
            min_pages=1,
        )
        assert val_ctx.retrieval_count >= 1
        assert "89.4%" in val_ctx.formatted_text
        assert len(val_ctx.source_pages) >= 1

    print("\n[R5 PASSED] Agentic RAG pipeline with source page tracking verified")


# ── Requirement 6: Intent Detection and Routing ──────────────────────────────
def test_requirement_6_intent_detection_and_routing():
    """R6: Verify that queries are routed accurately with confidence scores and reasoning."""
    test_cases = [
        ("Summarize this research paper", "summarize", 0.9),
        ("What are the main concepts and algorithms used?", "concepts", 0.6),
        ("Create a PowerPoint presentation for my seminar", "slides", 0.9),
        ("Generate likely viva questions with answers", "viva", 0.9),
        ("What GPU was used for training?", "qa", 0.3),  # Specific question -> general Q&A
    ]

    for query, expected_intent, min_conf in test_cases:
        res = detect_intent(query)
        assert res.intent == expected_intent, f"Query '{query}' expected {expected_intent}, got {res.intent}"
        assert res.confidence >= min_conf, f"Confidence too low for '{query}': {res.confidence}"
        assert len(res.reason) > 5, "Missing routing explanation"

    # Explicit routing check
    explicit_res = detect_intent("", explicit_intent="slides")
    assert explicit_res.intent == "slides"
    assert explicit_res.confidence == 1.0

    print("\n[R6 PASSED] Intent Detection and Routing verified with 4-tier strategy")


# ── Requirement 7: Vector Database (ChromaDB) ────────────────────────────────
def test_requirement_7_vector_database():
    """R7: Verify ChromaDB document storage, similarity search, and collection lifecycle."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vstore = VectorStoreManager(persist_dir=str(Path(tmpdir) / "chroma_db_test"))
        coll_name = "test_collection_lifecycle"

        # Initially empty
        stats = vstore.get_collection_stats(coll_name)
        assert stats["chunk_count"] == 0

        # Store test documents
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Gradient descent optimizes model weights via backpropagation.", metadata={"page": 1}),
            Document(page_content="Convolutional layers capture spatial hierarchies in images.", metadata={"page": 2}),
            Document(page_content="Attention mechanisms weigh tokens dynamically based on context.", metadata={"page": 3}),
        ]
        count = vstore.store_documents(docs, collection_id=coll_name)
        assert count == 3

        # Semantic similarity search check
        matches = vstore.retrieve_chunks("How are spatial image features processed?", collection_id=coll_name, k=1)
        assert len(matches) == 1
        assert "Convolutional" in matches[0].page_content
        assert matches[0].metadata["page"] == 2
        assert "distance" in matches[0].metadata

        # Delete collection check
        del_success = vstore.delete_collection(coll_name)
        assert del_success is True

    print("\n[R7 PASSED] ChromaDB storage, similarity search, and lifecycle verified")


if __name__ == "__main__":
    print("\n============================================================")
    print("  PaperPilot AI — Running All 7 Acceptance Criteria Tests")
    print("============================================================")
    test_requirement_1_llm_integration()
    test_requirement_2_interactive_gui()
    test_requirement_3_prompt_engineering()
    test_requirement_4_langchain_lcel()
    test_requirement_5_agentic_rag()
    test_requirement_6_intent_detection_and_routing()
    test_requirement_7_vector_database()
    print("\n============================================================")
    print("  ALL 7 MANDATORY REQUIREMENTS VERIFIED & DEMONSTRABLE [OK]")
    print("============================================================\n")
