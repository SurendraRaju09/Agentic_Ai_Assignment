# ✈️ PaperPilot AI — Autonomous Paper-to-Presentation & Viva Agent

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4%20LCEL-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-FF6F00?style=for-the-badge&logoColor=white)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5--coder%3A7b-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **An Agentic Multi-Agent RAG System that converts academic research papers into structured summaries, technical glossaries, downloadable 16:9 widescreen PowerPoint presentations (`.pptx`), and tiered external examiner viva defense simulations.**

---

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Key Highlights & Zero-Hallucination Guarantee](#-key-highlights--zero-hallucination-guarantee)
3. [System Architecture & Agent Workflow](#-system-architecture--agent-workflow)
4. [The 4 Specialized Agents & Intent Router](#-the-4-specialized-agents--intent-router)
5. [Seven Mandatory Acceptance Criteria](#-seven-mandatory-acceptance-criteria)
6. [Technology Stack](#-technology-stack)
7. [Installation & Setup](#-installation--setup)
8. [Running the Application](#-running-the-application)
9. [Automated Test Suite](#-automated-test-suite)
10. [Repository Structure](#-repository-structure)
11. [Configuration Reference](#-configuration-reference)
12. [Evaluation & Viva Defense Guide](#-evaluation--viva-defense-guide)
13. [Limitations & Future Roadmap](#-limitations--future-roadmap)
14. [License & Author](#-license--author)

---

## 💡 Overview

Reading, analyzing, presenting, and defending academic research papers requires dozens of hours of intense cognitive effort. Students and researchers must:
- Extract the core problem statement, methodologies, and quantitative baselines.
- Translate dense academic prose into structured, professional presentation slide decks.
- Anticipate rigorous, adversarial technical questions from faculty and external examiners during oral viva defenses.

**PaperPilot AI** solves this end-to-end challenge using an **Agentic Multi-Agent Retrieval-Augmented Generation (RAG)** pipeline. It parses research papers page-by-page, embeds chunks locally into a ChromaDB vector database, and coordinates four domain-specific AI agents orchestrated via **LangChain Expression Language (LCEL)** and powered by a local **Ollama `qwen2.5-coder:7b`** model (with zero API costs and total offline privacy).

---

## ✨ Key Highlights & Zero-Hallucination Guarantee

- **🔒 100% Local & Private:** Runs entirely on-device via Ollama without leaking unpublished manuscripts or academic drafts to third-party cloud APIs (with optional Groq cloud fallback).
- **🎯 Exact Page Citations:** PyMuPDF parses documents page-by-page, annotating every chunk with its physical source page number `[Page X]`.
- **🛡️ Strict Missing-Evidence Handling:** Prompts enforce that when evidence is absent from the paper, the agent explicitly marks it as *"Not reported in the paper"* rather than hallucinating plausible facts.
- **📊 Native 16:9 PowerPoint Export:** Compiles presentations into downloadable `.pptx` decks using `python-pptx` with standard academic slide hierarchies.
- **🎓 External Examiner Viva Simulator:** Generates tiered examination questions (Basic, Intermediate, and Advanced) with grounded answers and revision concepts.
- **⚡ Dynamic Intent Router:** Intelligently identifies whether the user wants a summary, concept glossary, slide generation, viva preparation, or direct paper Q&A.

---

## 🏗️ System Architecture & Agent Workflow

```text
                                 ┌─────────────────────────────────┐
                                 │       User (Web Browser)        │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │     Streamlit GUI (app.py)      │
                                 └──────┬───────────────────┬──────┘
                                        │                   │
                     1. Upload PDF      │                   │ 2. Query / Action
                                        ▼                   ▼
┌────────────────────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
│               Document Ingestion Pipeline              │  │          Intent Detection Router        │
│                                                        │  │        (core/intent_router.py)          │
│  [PDF File]                                            │  └───────────────────┬─────────────────────┘
│      │                                                 │                      │
│      ▼ (PyMuPDF with Page-Tracking)                    │                      ▼
│  [Document Pages with Metadata]                        │      ┌───────────────┴───────────────┐
│      │                                                 │      │                               │
│      ▼ (Recursive Chunker: 1000 chars, 150 overlap)    │      ▼                               ▼
│  [Text Chunks + Source Page Tags]                      │ [Pre-set Tab Actions]       [Free-form Prompt]
│      │                                                 │      │                               │
│      ▼ (Sentence-Transformers all-MiniLM-L6-v2)        │      └───────────────┬───────────────┘
│  [384-Dimensional Dense Vector Embeddings]             │                      │
│      │                                                 │                      ▼
│      ▼                                                 │       Routes to Targeted Agent Pipeline
│  [ChromaDB Vector Store (Local Persistent Collection)] │                      │
└───────────────────────────────┬────────────────────────┘                      │
                                │                                               │
                                └───────────────────────┬───────────────────────┘
                                                        │
                                                        ▼
                        ┌───────────────────────────────────────────────────────────────┐
                        │            LangChain LCEL Multi-Agent Pipeline                │
                        │               (agents/ & prompts/)                            │
                        └───────┬───────────────┬───────────────┬───────────────┬───────┘
                                │               │               │               │
        ┌───────────────────────┘               │               │               └───────────────────────┐
        ▼                                       ▼               ▼                                       ▼
┌───────────────────────┐       ┌───────────────────────┐ ┌───────────────────────┐     ┌───────────────────────┐
│     Agent 1:          │       │     Agent 2:          │ │     Agent 3:          │     │     Agent 4:          │
│   Summarizer Agent    │       │ Concept Extractor     │ │   Slide Generator     │     │   Viva Defense Agent  │
│                       │       │                       │ │                       │     │                       │
│ • Batched Map-Reduce  │       │ • Algorithmic metrics │ │ • 12-slide structure  │     │ • 3-Tier Examiner Q&A │
│ • 6 Academic Sections │       │ • Terminology glossary│ │ • Reuses Agent 1 & 2  │     │ • Basic/Inter/Adv     │
│ • Page-level evidence │       │ • Baseline validation │ │ • python-pptx builder │     │ • Revision concepts   │
└───────────┬───────────┘       └───────────┬───────────┘ └───────────┬───────────┘     └───────────┬───────────┘
            │                               │                         │                             │
            └───────────────────────────────┴────────────┬────────────┴─────────────────────────────┘
                                                         │
                                                         ▼
                                        ┌─────────────────────────────────┐
                                        │ Streamlit Interactive Dashboard │
                                        │  • Grounded Markdown Reports    │
                                        │  • Downloadable .pptx Slides    │
                                        │  • Interactive Defense Q&A      │
                                        └─────────────────────────────────┘
```

---

## 🤖 The 4 Specialized Agents & Intent Router

| Agent | Module | Retrieval Strategy | Output & Deliverables |
|---|---|---|---|
| **Agent 1: Summarizer** | `agents/summarizer_agent.py` | Broad retrieval (`k=10`), multi-page coverage, batched Map-Reduce for large documents. | 6 structured Markdown sections (Problem, Objectives, Methodology, Findings, Conclusion, Limitations) with page citations. |
| **Agent 2: Concept Extractor** | `agents/concept_agent.py` | Algorithmic & metric semantic queries (`k=6`), filtering non-technical chunks. | Structured technical glossary, algorithmic formulations, benchmarks, and quantitative metrics (Accuracy, F1, Latency). |
| **Agent 3: Slide Generator** | `agents/slide_agent.py` | Smart reuse of summary and concept outputs (conserving compute); queries missing details. | 12 standardized academic slides compiled into a downloadable widescreen 16:9 `.pptx` presentation file. |
| **Agent 4: Viva Examiner** | `agents/viva_agent.py` | Deep targeted retrieval (`k=8`) focusing on limitations, baselines, trade-offs, and failure modes. | Tiered external examiner defense questions (Basic, Intermediate, Advanced) with suggested answers and revision guides. |
| **Paper Q&A Agent** | `agents/qa_agent.py` | Targeted similarity search with page-context grounding. | Grounded conversational answers strictly citing page sources. |

### 🎯 Intent Router (`core/intent_router.py`)
Routes queries dynamically using a 4-tier strategy:
1. **Direct keyword intent matching** (Fast, deterministic)
2. **Action verb & context analysis** (Disambiguation)
3. **Task-specific confidence scoring**
4. **Default fallback to grounded Q&A**

---

## ✅ Seven Mandatory Acceptance Criteria

This project is built to satisfy seven formal academic engineering requirements, each verified by automated unit tests:

| # | Requirement | Implementation Module | Verification Test | Status |
|---|---|---|---|:---:|
| **R1** | **LLM Integration** | `utils/llm_factory.py` (Local Ollama primary, Groq API fallback) | `tests/test_pipeline.py::test_requirement_1_llm_integration` | ✅ Pass |
| **R2** | **Interactive GUI** | `app.py` (Streamlit web application with 8 distinct views and downloads) | `tests/test_pipeline.py::test_requirement_2_interactive_gui` | ✅ Pass |
| **R3** | **Prompt Engineering** | `prompts/*.py` (Documented System/Human prompt templates per agent) | `tests/test_pipeline.py::test_requirement_3_prompt_engineering` | ✅ Pass |
| **R4** | **LangChain & LCEL** | `agents/base_agent.py` (`prompt \| llm \| StrOutputParser()` pipe syntax) | `tests/test_pipeline.py::test_requirement_4_langchain_lcel` | ✅ Pass |
| **R5** | **Agentic RAG** | `core/pdf_processor.py`, `core/chunker.py`, `core/vector_store.py` | `tests/test_pipeline.py::test_requirement_5_agentic_rag` | ✅ Pass |
| **R6** | **Intent Routing** | `core/intent_router.py` (4-tier intent detection engine) | `tests/test_pipeline.py::test_requirement_6_intent_detection_and_routing` | ✅ Pass |
| **R7** | **Vector Database** | `core/vector_store.py` (ChromaDB + `all-MiniLM-L6-v2`) | `tests/test_pipeline.py::test_requirement_7_vector_database` | ✅ Pass |

---

## 🛠️ Technology Stack

| Layer | Component | Specification | Functionality |
|---|---|---|---|
| **Frontend** | Streamlit | v1.67+ | Responsive academic dashboard, metric displays, and `.pptx` download handling |
| **Orchestration** | LangChain Core | v1.4+ (LCEL) | Declarative pipe-based agent workflows (`prompt \| llm \| parser`) |
| **Primary LLM** | Ollama | `qwen2.5-coder:7b` | High-efficiency local inference, zero API cost, runs on CPU/GPU |
| **Fallback LLM** | Groq Cloud | `llama3-8b-8192` / `qwen` | Low-latency cloud fallback if local Ollama daemon is stopped |
| **Vector DB** | ChromaDB | v1.5+ | Local persistent vector storage with cosine similarity indexing |
| **Embeddings** | Sentence-Transformers | `all-MiniLM-L6-v2` | 384-dimensional dense semantic representations |
| **PDF Extraction**| PyMuPDF (fitz) | v2.1+ | Fast page-by-page text extraction with page metadata tracking |
| **Slide Engine** | python-pptx | v1.0.2+ | Programmatic PowerPoint generation with custom academic design layouts |
| **Testing** | Pytest | v8.0+ | Automated verification suite for the 7 acceptance criteria |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python:** 3.10, 3.11, or 3.12 installed.
- **Hardware:** Minimum 8GB RAM (16GB recommended for smooth local LLM inference).
- **Ollama:** Installed from [ollama.com](https://ollama.com).

Pull the local model:
```bash
ollama run qwen2.5-coder:7b
```

---

### 2. Clone the Repository & Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/Agentic_ai.git
cd Agentic_ai/PaperPilot

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Environment Configuration

Copy the template `.env.example` to create your active `.env`:
```bash
cp .env.example .env
```

Ensure your `.env` contains:
```ini
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.1

# Optional Groq Cloud Fallback (uncomment to enable)
# GROQ_API_KEY=your_groq_api_key_here
# GROQ_MODEL=llama3-8b-8192

# Storage Directories
CHROMA_PERSIST_DIR=./chroma_db
OUTPUT_DIR=./outputs
DATA_DIR=./data

# RAG & Chunking Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
MAX_CHUNKS_PER_RETRIEVAL=6
SUMMARIZER_K=10
VIVA_K=8
```

---

## 🚀 Running the Application

Make sure Ollama is running in the background, then launch Streamlit:

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

### Using the App:
1. **Upload:** Go to **Upload Research Paper** and drag & drop any academic PDF (or use `data/sample_biofast.pdf`).
2. **Index:** Click **🚀 Process & Index Paper** to extract text and generate ChromaDB vector embeddings.
3. **Analyze:**
   - **Paper Summary:** Generate the 6-point structured academic summary with page citations.
   - **Concept Extraction:** Extract technical keywords, algorithms, and metric tables.
   - **Presentation Generator:** Click **Generate 12-Slide Presentation (.pptx)** and download the widescreen `.pptx` deck.
   - **Viva Defense:** Review basic, intermediate, and advanced examiner questions with suggested answers.
   - **Ask Your Paper:** Ask any question; the dynamic intent router directs queries to the appropriate agent.

---

## 🧪 Automated Test Suite

The project includes thorough smoke tests and verification suites for all seven acceptance criteria:

### 1. Run the 12-Point Smoke Test
Verifies environment dependencies, local Ollama connectivity, PDF parser, and vector database:
```bash
python tests/smoke_test.py
```

### 2. Run the 7 Acceptance Criteria Tests
Executes the comprehensive Pytest suite:
```bash
pytest tests/test_pipeline.py -v
```

### 3. Run Backend Integration Tests
```bash
pytest tests/test_backend.py -v
```

---

## 📁 Repository Structure

```text
PaperPilot/
├── app.py                          # Streamlit interactive GUI application
├── requirements.txt                # Production Python dependencies
├── README.md                       # Project-level documentation
├── PROJECT_EVALUATION_GUIDE.md     # Faculty viva & defense evaluation guide
├── .env.example                    # Safe environment template
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Centralized configuration & hyper-parameters
│
├── core/                           # Core RAG & Ingestion Engines
│   ├── __init__.py
│   ├── pdf_processor.py            # PyMuPDF page-aware text extractor
│   ├── chunker.py                  # Recursive character chunking engine
│   ├── vector_store.py             # ChromaDB vector indexer & semantic search
│   └── intent_router.py            # 4-tier query intent detection & routing
│
├── agents/                         # LangChain Multi-Agent Architecture
│   ├── __init__.py
│   ├── base_agent.py               # Abstract agent base class (LCEL pipeline)
│   ├── summarizer_agent.py         # Agent 1: Batched Map-Reduce summarizer
│   ├── concept_agent.py            # Agent 2: Technical concept & metric extractor
│   ├── slide_agent.py              # Agent 3: 12-slide presentation content planner
│   ├── viva_agent.py               # Agent 4: Tiered external examiner defense agent
│   └── qa_agent.py                 # Grounded general paper Q&A agent
│
├── prompts/                        # Structured Agent Prompt Templates
│   ├── __init__.py
│   ├── summarizer_prompts.py       # 6-section academic summary prompts
│   ├── concept_prompts.py          # Algorithmic & metric extraction prompts
│   ├── slide_prompts.py            # 12-slide standard academic format prompts
│   └── viva_prompts.py             # Basic/Intermediate/Advanced viva prompts
│
├── generators/
│   ├── __init__.py
│   └── pptx_generator.py           # python-pptx widescreen presentation builder
│
├── utils/
│   ├── __init__.py
│   └── llm_factory.py              # Dynamic LLM provider factory (Ollama / Groq)
│
├── tests/                          # Automated Testing Suite
│   ├── __init__.py
│   ├── smoke_test.py               # 12-point environment validation test
│   ├── test_pipeline.py            # Tests for the 7 mandatory acceptance criteria
│   ├── test_backend.py             # Backend unit & integration tests
│   └── test_agent_live.py          # Live Ollama end-to-end integration test
│
├── data/                           # Sample research papers (.pdf)
└── outputs/                        # Generated presentation files (.pptx)
```

---

## ⚙️ Configuration Reference

All settings can be customized in `.env` or `config/settings.py`:

| Key | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Active LLM backend (`ollama` or `groq`) |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Model tag in local Ollama instance |
| `OLLAMA_BASE_URL`| `http://localhost:11434`| HTTP endpoint of local Ollama server |
| `OLLAMA_TEMPERATURE` | `0.1` | Low temperature ensures deterministic, non-hallucinated answers |
| `CHUNK_SIZE` | `1000` | Target character size per text chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between consecutive chunks |
| `SUMMARIZER_K` | `10` | Chunks retrieved for academic summarization |
| `VIVA_K` | `8` | Chunks retrieved for viva question generation |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Storage path for ChromaDB vector embeddings |

---

## 🎓 Evaluation & Viva Defense Guide

For academic defense or faculty presentations, refer to [`PROJECT_EVALUATION_GUIDE.md`](file:///c:/Dev/Projects/Agentic_ai/PaperPilot/PROJECT_EVALUATION_GUIDE.md):
- **3-Minute Opening Pitch:** Professional talking points explaining problem context and architectural novelty.
- **5-Minute Live Demo Script:** Exact click-by-click sequence to showcase automated tests, indexing, summarization, slide generation, and viva simulation.
- **Top 10 Faculty Viva Q&As:** Technical defenses on hallucination mitigation, chunking heuristics, LCEL design patterns, and embedding vector choices.

---

## 🔮 Limitations & Future Roadmap

- [ ] **OCR Support for Scanned PDFs:** Integrate `Tesseract` or `EasyOCR` to process legacy non-searchable scans.
- [ ] **Multimodal Diagram Ingestion:** Incorporate Vision-Language Models (e.g. `LLaVA` or `qwen2-vl`) to extract figures and architecture diagrams directly into generated slides.
- [ ] **Interactive Spoken Viva Simulator:** Integrate Speech-to-Text (Whisper) and Text-to-Speech (Coqui/Piper) for a realistic voice-driven mock examination session.

---

## 📄 License & Author

- **Author:** Surendra Raju
- **Academic Context:** 6th-Semester BE in Artificial Intelligence and Data Science
- **License:** Released under the [MIT License](https://opensource.org/licenses/MIT).
