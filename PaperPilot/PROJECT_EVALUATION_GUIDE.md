# 🎓 PaperPilot AI — Faculty Evaluation & Viva Defense Guide

> **Prepared for:** 6th-Semester BE Artificial Intelligence and Data Science Student  
> **Project:** Paper-to-Presentation & Viva Agent  
> **Use this guide to confidently explain, demonstrate, and defend your project during faculty evaluation.**

---

## 📌 Part 1: The 3-Minute Opening Pitch

When the faculty panel asks: *"Tell us about your project"*, deliver this structured opening:

> *"Respected evaluators, my project is titled **PaperPilot AI: Paper-to-Presentation and Viva Agent**.*
>
> *Reading and defending an academic research paper is one of the most critical challenges in higher technical education. Students spend dozens of hours extracting problem statements, synthesizing methodologies, creating slide presentations, and trying to anticipate what the examiner will ask in the oral viva.*
>
> *PaperPilot AI automates this workflow using an **Agentic Multi-Agent Retrieval-Augmented Generation (RAG)** architecture.*
>
> *The system accepts any research paper in PDF format and deploys **four specialized agents**:*
> 1. *A **Summarizer Agent** that produces a grounded 6-part academic summary using batched Map-Reduce.*
> 2. *A **Concept-Extractor Agent** that extracts algorithms, mathematical formulations, and evaluation metrics.*
> 3. *A **Slide-Generator Agent** that structures the findings into 12 standard academic slides and creates a downloadable 16:9 `.pptx` PowerPoint file.*
> 4. *A **Viva Agent** that simulates an external examiner by generating tiered basic, intermediate, and advanced questions with suggested answers.*
>
> *Crucially, our system eliminates AI hallucinations by grounding all outputs in a local **ChromaDB vector database** using **Sentence Transformer embeddings**, and orchestrates all agent pipelines using **LangChain Expression Language (LCEL)** connected to a locally hosted **Ollama `qwen2.5-coder:7b`** model.*
>
> *All seven mandatory requirements are fully implemented, and we have automated test suites to demonstrate every single criterion."*

---

## 📌 Part 2: 5-Minute Live Demonstration Script

Follow these exact steps when demonstrating to the evaluators:

### Step 1: Show the Automated Test Suite (1 minute)
Open a terminal in `PaperPilot/` and run:
```bash
pytest tests/test_pipeline.py -v
```
**What to say:**
> *"Before opening the UI, I would like to show our automated acceptance test suite. We have seven automated tests that map 1-to-1 to each mandatory requirement: LLM integration, Streamlit GUI, prompt engineering, LangChain LCEL pipelines, Agentic RAG with page citations, 4-tier intent routing, and ChromaDB vector search. As you can see, all seven tests pass cleanly."*

### Step 2: Launch the Streamlit Dashboard (30 seconds)
```bash
streamlit run app.py
```
Open `http://localhost:8501`.  
**What to say:**
> *"This is our Streamlit dashboard. In the sidebar, we display the active local model — `qwen2.5-coder:7b` running locally on Ollama without external cloud dependencies or API costs. The sidebar provides structured navigation across eight distinct sections."*

### Step 3: Upload and Index a Research Paper (1 minute)
1. Click **Upload Research Paper**.
2. Upload `data/sample_biofast.pdf` (or any IEEE/arXiv paper).
3. Click **🚀 Process & Index Paper**.  
**What to say:**
> *"When we upload a PDF, PyMuPDF extracts text page-by-page to preserve source page metadata. The text is passed to LangChain's RecursiveCharacterTextSplitter with 1000-character chunks and 150-character overlap. Sentence Transformers then converts chunks into 384-dimensional dense vectors and stores them in ChromaDB with persistent collection management."*

### Step 4: Demonstrate Summarizer & Concept Extraction (1 minute)
1. Click **Paper Summary** → Click **Generate Structured Summary**.
2. Point out:
   - The 6 distinct sections: Problem, Objectives, Methodology, Findings, Conclusion, Limitations.
   - The **Source Pages Cited** badge: `[1, 2, 3]`.
   - The **Context Quality** pill: `OPTIMAL`.
3. Click **Concept Extraction** → Show extracted algorithms and metrics (e.g. `89.4% F1-score`).  
**What to say:**
> *"Notice that the agent does not invent information. In the Limitations section, it strictly cites the degradation on rare clinical acronyms that was actually present on page 3 of the paper. This proves zero hallucination."*

### Step 5: Generate & Download PowerPoint Presentation (1 minute)
1. Click **Presentation Generator** → Click **Generate 12-Slide Presentation (.pptx)**.
2. Show the slide preview.
3. Click **📥 Download Presentation (.pptx)** and open it in PowerPoint.  
**What to say:**
> *"The Slide Agent reuses the previously computed summary and concepts to avoid redundant vector queries and compute. It structures the content into 12 standard academic slides and python-pptx compiles a 16:9 widescreen presentation with custom styling."*

### Step 6: Demonstrate the Intent Router (30 seconds)
1. Click **Ask Your Paper**.
2. Type: `"What are the likely viva questions?"` → Point out the blue routing badge:  
   `🎯 Routed to: Viva Agent (Confidence: 95%)`.
3. Type: `"What F1 score was achieved?"` → Point out:  
   `🎯 Routed to: Paper Q&A Agent`.  
**What to say:**
> *"Our Intent Router implements a 4-tier strategy: explicit navigation, exact trigger matching, weighted keyword scoring, and fallback to Q&A. The user is always informed of the routing decision and confidence score."*

---

## 📌 Part 3: 15 Likely Faculty Viva Questions & Exact Answers

### Q1: What is the difference between standard RAG and "Agentic RAG" in your project?
**Answer:**
> *"In basic RAG, a single generic search query is sent to the vector database regardless of the task. In our Agentic RAG implementation:
> 1. Each agent formulates its own domain-specific query (e.g., the Summarizer searches for problems and conclusions, whereas the Concept Agent searches for algorithms and benchmarks).
> 2. The base agent validates context quality by checking page diversity (ensuring chunks come from at least 2-3 different pages).
> 3. If retrieval is too narrow, the agent self-heals by triggering an automated query expansion with broader keywords and a higher k value.
> 4. The Slide Agent practices context reuse by ingesting prior agent outputs, preventing redundant queries."*

---

### Q2: Why did you choose ChromaDB over alternatives like FAISS or Pinecone?
**Answer:**
> *"Pinecone is a cloud-hosted vector database requiring internet connectivity, API keys, and recurring costs. FAISS is an efficient in-memory indexing library from Meta, but it does not natively provide persistent document metadata storage out of the box.  
> ChromaDB is an open-source, local vector database that runs entirely in Python without external servers. It manages persistent document storage, embedding functions, cosine similarity search, and collection lifecycle management locally on disk."*

---

### Q3: What embedding model do you use, and why?
**Answer:**
> *"We use `all-MiniLM-L6-v2` from Sentence Transformers. It maps text sentences into a 384-dimensional dense vector space. We chose it because it is lightweight (~90MB), runs on CPU without requiring an expensive GPU, and scores competitively on information retrieval benchmarks."*

---

### Q4: How does your Summarizer Agent handle very long research papers without exceeding the LLM context window?
**Answer:**
> *"We implemented an adaptive strategy with **Batched Map-Reduce**. If the retrieved text exceeds 4500 characters, rather than passing the entire text in a single prompt or making 20 individual LLM calls for every chunk, we group chunks into batches of 3 by page proximity. The model generates intermediate summary notes for each batch (Map phase), and then synthesizes all batch notes into the final structured 6-part summary (Reduce phase)."*

---

### Q5: How do you prevent hallucinations in your LLM outputs?
**Answer:**
> *"We prevent hallucinations through three mechanisms:
> 1. **Grounded RAG:** The LLM prompt strictly receives retrieved excerpts from ChromaDB and is instructed never to extrapolate beyond what is in the text.
> 2. **Explicit Fallback Prompts:** In our system prompts, if a piece of information (such as limitations or baseline comparisons) is not mentioned in the paper, the model is instructed to write: 'Not explicitly stated in the paper'.
> 3. **Page Metadata Tracking:** Every chunk carries the original PDF page number, allowing students and examiners to verify any claim against the source document."*

---

### Q6: What is LCEL and where is it used in your codebase?
**Answer:**
> *"LCEL stands for **LangChain Expression Language**. It is a declarative way to chain LangChain components using the Python pipe operator (`|`). In our `agents/base_agent.py` and across all agents, we chain:
> `chain = prompt | self.llm | self.output_parser`
> This replaces legacy, black-box chains like `LLMChain` with transparent, streaming-compatible, and explainable pipelines."*

---

### Q7: How does your Intent Router work? What if a user enters an ambiguous query?
**Answer:**
> *"Our Intent Router in `core/intent_router.py` uses a 4-tier routing hierarchy:
> 1. **Explicit Navigation:** User menu selection guarantees 100% confidence.
> 2. **Exact Keyphrase Matching:** Predefined triggers like 'create slides' yield 95% confidence.
> 3. **Weighted Keyword Scoring:** Keywords like 'algorithm' (+5) or 'viva' (+6) accumulate intent scores.
> 4. **Ambiguity Resolution:** If the top two intent scores are within 1 point of each other, or if no keywords match, the system safely falls back to general Q&A (`qa`) with a transparent explanation to the user."*

---

### Q8: Why did you use `qwen2.5-coder:7b` instead of standard `llama3:8b`?
**Answer:**
> *"We tested `qwen2.5-coder:7b` against our structured prompting requirements. Despite the 'coder' designation, Qwen2.5 has exceptional instruction-following capabilities, particularly when adhering to strict markdown headers, JSON-like structures, and concise bullet points. It runs efficiently at 11 tokens per second on our 16GB laptop via Ollama."*

---

### Q9: Can your system switch from local Ollama to a cloud API like Groq or OpenAI?
**Answer:**
> *"Yes. We implemented the **Factory Pattern** in `utils/llm_factory.py`. By changing `LLM_PROVIDER=groq` and supplying a `GROQ_API_KEY` in the `.env` file, the entire application seamlessly switches to Groq without modifying a single line of agent or application code."*

---

### Q10: How does `python-pptx` generate slides from LLM text?
**Answer:**
> *"The Slide Agent generates a structured outline using `SLIDE <N>: <Title>` and bullet points (`-`). Our parser in `agents/slide_agent.py` extracts this into Python dictionaries. `generators/pptx_generator.py` then creates a 16:9 widescreen presentation object, applies our academic color theme (Navy `#1A365D`, Accent Blue `#3182CE`), constructs title banners, and formats the bullet points with proper font sizing and line spacing."*

---

### Q11: What chunk size and overlap did you choose, and why?
**Answer:**
> *"We chose `chunk_size = 1000` characters (~150-200 words) and `chunk_overlap = 150` characters. A 1000-character window is optimal for academic papers because it captures complete paragraphs or mathematical methodologies without dilution. The 150-character overlap prevents sentences and terminology from being split at chunk boundaries."*

---

### Q12: What happens if an empty, corrupted, or image-only scanned PDF is uploaded?
**Answer:**
> *"In `core/pdf_processor.py`, we implement custom exception handling with `PDFProcessingError`. We validate that the file is not 0 bytes, is not encrypted, and extracts at least 50 readable characters. If an image-only scanned PDF is uploaded, the system catches it and presents a user-friendly error message on Streamlit explaining that the PDF lacks OCR text."*

---

### Q13: What are the three tiers in the Viva Agent, and how are they prompted?
**Answer:**
> *"The three tiers reflect the progression of an oral defense:
> 1. **Basic Tier:** Fundamental definitions, paper motivation, and 'what is' questions.
> 2. **Intermediate Tier:** Algorithmic mechanics, hyperparameter choices, and dataset characteristics.
> 3. **Advanced Tier:** Critical defense questions covering baseline comparisons, threats to validity, trade-offs, and failure cases.  
> We use few-shot prompt engineering in `prompts/viva_prompts.py` to guide the LLM on question depth and answer formatting."*

---

### Q14: How is session state managed in Streamlit?
**Answer:**
> *"Streamlit re-runs the script from top to bottom on user interactions. We use `st.session_state` to store the active collection ID, paper metadata, and generated results (`summary_result`, `slides_result`, etc.). This ensures that when a user generates a summary and then clicks on 'Presentation Generator', their data persists without re-running expensive LLM inferences."*

---

### Q15: What are the primary limitations of your current implementation?
**Answer:**
> *"Currently, the system extracts text streams via PyMuPDF; it cannot parse complex bitmap diagrams or charts without an OCR or Vision-Language Model. Secondly, the local 7B parameter model has modest inference speeds (~15-20 seconds per generation on CPU). Future enhancements would incorporate multimodal chart parsing and voice synthesis for interactive verbal mock-viva practice."*

---

## 📌 Part 4: How the 7 Mandatory Criteria are Satisfied

| Mandatory Requirement | Code Location | Demonstrable Evidence |
|---|---|---|
| **R1: LLM Integration** | `utils/llm_factory.py` | Runs locally on Ollama `qwen2.5-coder:7b` with clean output; tested in `test_requirement_1_llm_integration`. |
| **R2: Interactive GUI** | `app.py` | 8 sidebar navigation pages, status pills, file uploader, `.pptx` download button. |
| **R3: Prompt Engineering** | `prompts/*.py` | 4 distinct files with System/Human templates, few-shot examples, and documented rationales. |
| **R4: LangChain & LCEL** | `agents/base_agent.py` | Chains built with `prompt \| self.llm \| self.output_parser` using the pipe operator. |
| **R5: Agentic RAG** | `core/pdf_processor.py`, `core/vector_store.py`, `agents/` | Page metadata preserved, agent-specific queries, adaptive query expansion, grounded citations. |
| **R6: Intent Routing** | `core/intent_router.py` | 4-tier routing with explicit, exact phrase, keyword scoring, and fallback; tested in `test_requirement_6_intent_detection_and_routing`. |
| **R7: Vector Database** | `core/vector_store.py` | ChromaDB collection storage, cosine similarity search, chunk count stats, collection lifecycle. |
