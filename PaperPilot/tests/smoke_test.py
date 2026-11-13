"""
tests/smoke_test.py — PaperPilot AI Phase 3 Environment Smoke Tests
====================================================================
PURPOSE:
    Verifies that every essential component of the environment is working
    correctly BEFORE we write any agent code.

    Think of these as "can the engine start?" tests — not full functionality tests.
    Full requirement tests live in test_pipeline.py (Phase 6).

HOW TO RUN:
    From the PaperPilot/ directory (with venv activated):
    python tests/smoke_test.py

WHAT EACH TEST CHECKS:
    1.  Python version (must be 3.10+)
    2.  All required packages importable
    3.  config/settings.py loads correctly
    4.  Ollama server reachable
    5.  qwen2.5-coder:7b model available
    6.  Ollama generates a response (with timing)
    7.  LangChain imports and prompt template works
    8.  LCEL pipeline (prompt | llm | parser) works
    9.  Sentence Transformer loads and embeds text
    10. ChromaDB can store and retrieve documents
    11. PyMuPDF can read a PDF
    12. python-pptx can create a presentation file
"""

import sys
import os
import time
import json
import tempfile
import traceback
from pathlib import Path

# Add project root to Python path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Colour helpers (works without any extra package) ──────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

results: list[dict] = []


def check(name: str, fn) -> bool:
    """Runs a single test and records pass/fail."""
    print(f"\n{CYAN}Testing:{RESET} {name} ...", end=" ", flush=True)
    start = time.time()
    try:
        detail = fn()
        elapsed = round(time.time() - start, 2)
        print(f"{GREEN}PASS{RESET} ({elapsed}s)")
        if detail:
            print(f"  -> {detail}")
        results.append({"name": name, "status": "PASS", "detail": detail or "", "time": elapsed})
        return True
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"{RED}FAIL{RESET} ({elapsed}s)")
        print(f"  -> {RED}{e}{RESET}")
        results.append({"name": name, "status": "FAIL", "detail": str(e), "time": elapsed})
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Python Version
# ══════════════════════════════════════════════════════════════════════════════
def test_python_version():
    v = sys.version_info
    assert v >= (3, 10), f"Python 3.10+ required, got {v.major}.{v.minor}"
    return f"Python {v.major}.{v.minor}.{v.micro}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Package Imports
# ══════════════════════════════════════════════════════════════════════════════
def test_package_imports():
    packages = {
        "streamlit": "streamlit",
        "langchain": "langchain",
        "langchain_core": "langchain_core",
        "langchain_community": "langchain_community",
        "langchain_ollama": "langchain_ollama",
        "langchain_groq": "langchain_groq",
        "ollama": "ollama",
        "sentence_transformers": "sentence_transformers",
        "chromadb": "chromadb",
        "fitz (PyMuPDF)": "fitz",
        "pptx (python-pptx)": "pptx",
        "dotenv": "dotenv",
        "pytest": "pytest",
    }
    failed = []
    for display_name, module_name in packages.items():
        try:
            __import__(module_name)
        except ImportError:
            failed.append(display_name)

    if failed:
        raise ImportError(f"Missing packages: {', '.join(failed)}")
    return f"All {len(packages)} packages importable"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Settings Module
# ══════════════════════════════════════════════════════════════════════════════
def test_settings():
    from config import settings
    assert settings.LLM_PROVIDER in ("ollama", "groq"), "LLM_PROVIDER must be 'ollama' or 'groq'"
    assert settings.OLLAMA_MODEL, "OLLAMA_MODEL must not be empty"
    assert settings.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE, "CHUNK_OVERLAP must be < CHUNK_SIZE"
    warnings = settings.validate_config()
    warn_str = f" | Warnings: {warnings}" if warnings else ""
    return f"Provider={settings.LLM_PROVIDER} | Model={settings.OLLAMA_MODEL}{warn_str}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Ollama Server Reachable
# ══════════════════════════════════════════════════════════════════════════════
def test_ollama_server():
    import urllib.request
    from config import settings
    url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        return f"Ollama running | Available models: {models}"
    except Exception as e:
        raise ConnectionError(f"Cannot reach Ollama at {settings.OLLAMA_BASE_URL}. Is Ollama running? Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: qwen2.5-coder:7b Model Available
# ══════════════════════════════════════════════════════════════════════════════
def test_model_available():
    import urllib.request
    from config import settings
    url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    model_names = [m["name"] for m in data.get("models", [])]
    # Check if our model (or a prefix match) is available
    target = settings.OLLAMA_MODEL
    matches = [m for m in model_names if m.startswith(target.split(":")[0])]
    if not matches:
        raise ValueError(f"Model '{target}' not found. Available: {model_names}")
    return f"Found: {matches}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Ollama Generates a Response
# ══════════════════════════════════════════════════════════════════════════════
def test_ollama_response():
    import urllib.request
    from config import settings
    payload = json.dumps({
        "model": settings.OLLAMA_MODEL,
        "prompt": "Reply with exactly three words: AI is powerful",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 20}
    }).encode()
    req = urllib.request.Request(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    response = data.get("response", "").strip()
    tps = data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001)
    assert response, "Empty response from Ollama"
    return f'Response: "{response[:60]}" | Speed: {tps:.1f} tok/s'


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: LangChain Prompt Template
# ══════════════════════════════════════════════════════════════════════════════
def test_langchain_prompt():
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage

    # This is exactly how we build prompts in every agent
    system = SystemMessagePromptTemplate.from_template(
        "You are a research assistant. Answer based only on the provided context."
    )
    human = HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_messages([system, human])

    # Format the template with actual values
    messages = prompt.format_messages(
        context="The paper proposes a new transformer model.",
        question="What does the paper propose?"
    )
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "transformer" in messages[1].content
    return "ChatPromptTemplate with System + Human messages works correctly"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: LangChain LCEL Pipeline (prompt | llm | parser)
# ══════════════════════════════════════════════════════════════════════════════
def test_lcel_pipeline():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    from config import settings

    # This is the LCEL pattern used in every agent:
    # prompt template → LLM → output parser (chain them with the pipe | operator)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be very brief."),
        ("human", "Say 'LCEL works' and nothing else.")
    ])

    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=10
    )

    parser = StrOutputParser()

    # The LCEL chain: prompt → llm → parser
    # The pipe operator (|) connects these components into a pipeline
    chain = prompt | llm | parser

    result = chain.invoke({})
    assert result.strip(), "LCEL chain returned empty result"
    return f'LCEL chain output: "{result.strip()[:60]}"'


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Sentence Transformers (Embedding Generation)
# ══════════════════════════════════════════════════════════════════════════════
def test_embeddings():
    from sentence_transformers import SentenceTransformer
    # Load the embedding model — downloads ~90MB on first use, then cached
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Test: embed two similar sentences and one different sentence
    sentences = [
        "The transformer model achieves high accuracy on NLP tasks.",
        "Transformer architecture performs well on natural language processing.",
        "The stock market crashed yesterday."  # unrelated
    ]
    embeddings = model.encode(sentences)

    assert embeddings.shape == (3, 384), f"Expected (3,384) got {embeddings.shape}"

    # Compute cosine similarities
    from numpy import dot
    from numpy.linalg import norm
    cosine = lambda a, b: dot(a, b) / (norm(a) * norm(b))

    sim_related = cosine(embeddings[0], embeddings[1])    # should be HIGH
    sim_unrelated = cosine(embeddings[0], embeddings[2])  # should be LOW

    assert sim_related > sim_unrelated, "Similar sentences should score higher than unrelated"
    return f"Embedding dim=384 | Similar pair: {sim_related:.3f} | Unrelated pair: {sim_unrelated:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: ChromaDB (Store + Retrieve)
# ══════════════════════════════════════════════════════════════════════════════
def test_chromadb():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    # Create a temporary in-memory ChromaDB client (no disk writes during test)
    client = chromadb.Client()

    # Create a test collection with our embedding model
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(name="smoke_test", embedding_function=ef)

    # Store 5 test documents
    collection.add(
        documents=[
            "The proposed CNN model achieves 94% accuracy on CIFAR-10 dataset.",
            "Data augmentation and batch normalization were used for training.",
            "The model uses residual connections to prevent vanishing gradients.",
            "Experiments were conducted on NVIDIA V100 GPUs with PyTorch.",
            "Future work includes extending the model to video classification tasks."
        ],
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
        metadatas=[{"page": i+1} for i in range(5)]
    )

    # Retrieve: query about accuracy — should return doc1 as most relevant
    results_q = collection.query(
        query_texts=["what accuracy does the model achieve?"],
        n_results=2
    )
    top_doc = results_q["documents"][0][0]
    assert "94%" in top_doc or "accuracy" in top_doc.lower(), \
        f"Expected accuracy-related doc, got: {top_doc}"

    # Check metadata was preserved
    top_meta = results_q["metadatas"][0][0]
    assert "page" in top_meta, "Metadata not preserved in ChromaDB"

    return f"Stored 5 docs | Top result: '{top_doc[:60]}...' | Metadata: {top_meta}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: PyMuPDF (PDF Reading)
# ══════════════════════════════════════════════════════════════════════════════
def test_pymupdf():
    import fitz  # PyMuPDF

    # Create a minimal in-memory PDF to test with (no file on disk needed)
    doc = fitz.open()  # new empty document
    page = doc.new_page()
    page.insert_text((72, 72), "PaperPilot AI: Test Research Paper\nAbstract: This is a test.")
    
    # Save to bytes and re-read (simulates upload)
    pdf_bytes = doc.tobytes()
    doc.close()

    # Re-open from bytes and extract text
    doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert doc2.page_count == 1, "Expected 1 page"

    page_text = doc2[0].get_text()
    assert "PaperPilot" in page_text, "Text extraction failed"

    # Test metadata extraction (what pdf_processor.py will do)
    page_meta = {"page": 1, "total_pages": doc2.page_count, "source": "test.pdf"}
    doc2.close()

    return f"Pages: {page_meta['total_pages']} | Text extracted: '{page_text[:40].strip()}...'"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: python-pptx (Presentation Creation)
# ══════════════════════════════════════════════════════════════════════════════
def test_pptx():
    from pptx import Presentation
    from pptx.util import Inches, Pt
    import tempfile

    prs = Presentation()

    # Slide 1: Title slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.shapes.title.text = "PaperPilot AI — Test Presentation"
    slide.placeholders[1].text = "Generated by PaperPilot AI"

    # Slide 2: Content slide with bullet points
    bullet_slide_layout = prs.slide_layouts[1]
    slide2 = prs.slides.add_slide(bullet_slide_layout)
    slide2.shapes.title.text = "Key Findings"
    tf = slide2.placeholders[1].text_frame
    tf.text = "Finding 1: Model achieves 94% accuracy"
    tf.add_paragraph().text = "Finding 2: 40% fewer parameters than baseline"

    # Save to a temp file and verify it exists
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        tmp_path = f.name
    prs.save(tmp_path)

    size_kb = round(os.path.getsize(tmp_path) / 1024, 1)
    assert size_kb > 0, "PPTX file is empty"
    os.unlink(tmp_path)  # clean up

    return f"Created 2-slide PPTX ({size_kb} KB) | Bullet points work | File deletable"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Run all tests and print summary
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  PaperPilot AI — Phase 3 Smoke Tests{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    tests = [
        ("1. Python Version",                    test_python_version),
        ("2. Package Imports",                   test_package_imports),
        ("3. Config / Settings",                 test_settings),
        ("4. Ollama Server Reachable",           test_ollama_server),
        ("5. Model Available (qwen2.5-coder)",   test_model_available),
        ("6. Ollama Generates Response",         test_ollama_response),
        ("7. LangChain Prompt Templates",        test_langchain_prompt),
        ("8. LCEL Pipeline (prompt|llm|parser)", test_lcel_pipeline),
        ("9. Sentence Transformers Embeddings",  test_embeddings),
        ("10. ChromaDB Store + Retrieve",        test_chromadb),
        ("11. PyMuPDF PDF Reading",              test_pymupdf),
        ("12. python-pptx Slide Creation",       test_pptx),
    ]

    for name, fn in tests:
        check(name, fn)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total  = len(results)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  RESULTS: {passed}/{total} passed{RESET}", end="  ")
    if failed == 0:
        print(f"{GREEN}ALL TESTS PASSED [OK]{RESET}")
    else:
        print(f"{RED}{failed} FAILED [X]{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    if failed > 0:
        print(f"\n{RED}Failed tests:{RESET}")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  [X] {r['name']}: {r['detail']}")
        print()

    # Return exit code for CI/CD use
    sys.exit(0 if failed == 0 else 1)
