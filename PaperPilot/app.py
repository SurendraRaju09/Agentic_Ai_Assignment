"""
app.py — PaperPilot AI Streamlit Application
=============================================
PURPOSE:
    Interactive academic AI dashboard for research paper analysis, presentation generation,
    and viva defense preparation.
    Implements Requirement 2 (Interactive GUI) and orchestrates all backend agents.
"""

import streamlit as st
from pathlib import Path
import time

# Core & Backend Imports
from config import settings
from core.pdf_processor import extract_text_from_pdf, PDFProcessingError
from core.chunker import chunk_documents
from core.vector_store import VectorStoreManager
from core.intent_router import detect_intent, AGENT_NAME_MAP
from agents.summarizer_agent import SummarizerAgent
from agents.concept_agent import ConceptAgent
from agents.slide_agent import SlideAgent
from agents.viva_agent import VivaAgent
from agents.qa_agent import QAAgent


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperPilot AI — Research Paper & Viva Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Professional Academic Styling ─────────────────────────────
st.markdown("""
<style>
    /* Metric & Card styling */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .agent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-bottom: 8px;
    }
    .page-cite {
        background-color: #f1f5f9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-optimal { background-color: #dcfce7; color: #15803d; }
    .status-limited { background-color: #fef9c3; color: #a16207; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
if "paper_loaded" not in st.session_state:
    st.session_state.paper_loaded = False
if "paper_filename" not in st.session_state:
    st.session_state.paper_filename = ""
if "paper_title" not in st.session_state:
    st.session_state.paper_title = ""
if "collection_id" not in st.session_state:
    st.session_state.collection_id = ""
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "concepts_result" not in st.session_state:
    st.session_state.concepts_result = None
if "slides_result" not in st.session_state:
    st.session_state.slides_result = None
if "viva_result" not in st.session_state:
    st.session_state.viva_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Lazy Resource Cache ───────────────────────────────────────────────────────
@st.cache_resource
def get_vector_store():
    return VectorStoreManager()


vstore = get_vector_store()


# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## ✈️ PaperPilot AI")

# LLM Engine Selector
provider_choice = st.sidebar.selectbox(
    "⚡ LLM Inference Engine",
    ["Local Ollama (Offline / CPU)", "Groq Cloud API (Lightning Fast)"],
    index=0 if settings.LLM_PROVIDER == "ollama" else 1,
    help="Ollama runs locally on CPU. Groq provides ultra-fast cloud inference (3-5s)."
)

if "Groq" in provider_choice:
    settings.LLM_PROVIDER = "groq"
    groq_input = st.sidebar.text_input("Groq API Key", value=settings.GROQ_API_KEY, type="password")
    if groq_input:
        settings.GROQ_API_KEY = groq_input
    st.sidebar.caption(f"Model: `{settings.GROQ_MODEL}` (Ultra-fast Cloud)")
else:
    settings.LLM_PROVIDER = "ollama"
    st.sidebar.caption(f"Model: `{settings.OLLAMA_MODEL}` (Local CPU)")

st.sidebar.divider()

nav = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard",
        "📤 Upload Research Paper",
        "📝 Paper Summary",
        "🔬 Concept Extraction",
        "📊 Presentation Generator",
        "🎓 Viva Preparation",
        "💬 Ask Your Paper",
        "ℹ️ About Project",
    ],
    index=0,
)

st.sidebar.divider()
if st.session_state.paper_loaded:
    st.sidebar.success(f"📄 **Active Paper:**\n{st.session_state.paper_filename}")
    st.sidebar.markdown(f"• **Pages:** {st.session_state.total_pages}")
    st.sidebar.markdown(f"• **ChromaDB Chunks:** {st.session_state.chunk_count}")
else:
    st.sidebar.info("No paper uploaded yet.\nGo to **Upload Research Paper** to begin.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Dashboard":
    st.title("Academic Paper-to-Presentation & Viva Agent")
    st.markdown("""
    Welcome to **PaperPilot AI**, an Agentic RAG system that transforms research papers into
    structured summaries, technical glossaries, professional PowerPoint slide decks, and comprehensive viva preparation guides.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Active Engine", value=settings.LLM_PROVIDER.upper())
    with col2:
        st.metric(label="Indexed Chunks", value=st.session_state.chunk_count)
    with col3:
        st.metric(label="Document Pages", value=st.session_state.total_pages)
    with col4:
        st.metric(label="Vector DB", value="ChromaDB (Local)")

    st.markdown("### 🤖 The Multi-Agent Collaboration Workflow")
    st.markdown("""
    ```
    [Uploaded Research Paper] 
               │
               ▼
    [PyMuPDF Page Parser] ──► [Recursive Chunker] ──► [all-MiniLM-L6-v2] ──► [ChromaDB Vector Store]
                                                                                   │
         ┌──────────────────────────────┬──────────────────────────────┬───────────┴───────────┐
         ▼                              ▼                              ▼                       ▼
    [Agent 1: Summarizer]     [Agent 2: Concept Extractor]   [Agent 3: Slide Generator]   [Agent 4: Viva Agent]
    • Problem & Objectives    • Algorithms & Math            • 12 Academic Slides         • Basic / Inter / Adv Q&A
    • Batched Map-Reduce      • Metrics & Baselines          • Direct .pptx Export        • Suggested Answers
    ```
    """)

    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. Navigate to **Upload Research Paper** in the sidebar.
    2. Upload any PDF and click **🚀 Process & Generate Complete Academic Package** to autonomously run all agents at once.
    3. Download your PowerPoint presentation, summary report, and viva guide in one click!
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Upload Research Paper
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📤 Upload Research Paper":
    st.title("📤 Upload and Index Research Paper")
    st.markdown("Upload an academic paper in PDF format. You can index the paper or generate the complete academic suite in one click.")

    uploaded_file = st.file_uploader("Select PDF Research Paper", type=["pdf"], help="Upload academic paper (IEEE, ACM, Springer, arXiv, etc.)")

    if uploaded_file is not None:
        filename = uploaded_file.name
        st.write(f"**Filename:** `{filename}` | **Size:** {round(uploaded_file.size / 1024, 1)} KB")

        col_btn1, col_btn2 = st.columns([2, 1])

        with col_btn1:
            run_all = st.button("🌟 Process & Generate Complete Package (All Agents)", type="primary", use_container_width=True)
        with col_btn2:
            index_only = st.button("📁 Index Paper Only", use_container_width=True)

        if run_all or index_only:
            try:
                with st.status("Executing PaperPilot AI Pipeline...", expanded=True) as status:
                    # Step 1: Extraction
                    status.write("📄 [1/5] Extracting text page-by-page with PyMuPDF...")
                    docs = extract_text_from_pdf(uploaded_file, filename=filename)

                    # Step 2: Chunking
                    status.write(f"✂️ [2/5] Creating semantic chunks with preserved page metadata ({len(docs)} pages)...")
                    chunks = chunk_documents(docs)

                    # Step 3: Embeddings & ChromaDB
                    status.write(f"🗄️ [3/5] Generating dense vectors and indexing {len(chunks)} chunks into ChromaDB...")
                    coll_id = vstore.sanitize_collection_name(filename)
                    count = vstore.store_documents(chunks, collection_id=coll_id)

                    # Update session state for document
                    st.session_state.paper_loaded = True
                    st.session_state.paper_filename = filename
                    st.session_state.paper_title = filename.replace(".pdf", "").replace("_", " ").title()
                    st.session_state.collection_id = coll_id
                    st.session_state.total_pages = len(docs)
                    st.session_state.chunk_count = count

                    if run_all:
                        # Agent 1: Summarizer
                        status.write("📝 [4/5] Running Agent 1: Generating structured academic summary...")
                        sum_agent = SummarizerAgent(vector_store=vstore)
                        st.session_state.summary_result = sum_agent.run(collection_id=coll_id)

                        # Agent 2: Concepts
                        status.write("🔬 [5/5] Running Agent 2: Extracting technical concepts, algorithms, and metrics...")
                        con_agent = ConceptAgent(vector_store=vstore)
                        st.session_state.concepts_result = con_agent.run(collection_id=coll_id)

                        # Agent 3: Slides & PPTX
                        status.write("📊 Running Agent 3: Structuring 12 academic slides and building .pptx presentation...")
                        slide_agent = SlideAgent(vector_store=vstore)
                        prior_sum = st.session_state.summary_result["summary"] if st.session_state.summary_result else None
                        prior_con = st.session_state.concepts_result["concepts"] if st.session_state.concepts_result else None
                        st.session_state.slides_result = slide_agent.run(
                            collection_id=coll_id,
                            prior_summary=prior_sum,
                            prior_concepts=prior_con,
                            paper_title=st.session_state.paper_title,
                        )

                        # Agent 4: Viva
                        status.write("🎓 Running Agent 4: Generating tiered viva examiner questions and suggested answers...")
                        viva_agent = VivaAgent(vector_store=vstore)
                        st.session_state.viva_result = viva_agent.run(collection_id=coll_id)

                        status.update(label="🎉 Complete Academic Package Generated Successfully!", state="complete", expanded=False)
                    else:
                        status.update(label=f"✅ Indexed {count} chunks across {len(docs)} pages in ChromaDB!", state="complete", expanded=False)

                st.success(f"✅ Successfully processed **{filename}**!")
                st.balloons()
                st.rerun()

            except PDFProcessingError as e:
                st.error(f"❌ PDF Processing Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Processing Error: {str(e)}")

    if st.session_state.paper_loaded:
        st.divider()
        st.markdown("### 📑 Active Document Status")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Title:** {st.session_state.paper_title}")
        c2.info(f"**Total Pages:** {st.session_state.total_pages}")
        c3.info(f"**ChromaDB Chunks:** {st.session_state.chunk_count}")

        # If complete package has been generated, show one-click download buttons right here!
        if st.session_state.slides_result or st.session_state.summary_result:
            st.markdown("### 📥 Instant Package Downloads")
            d_col1, d_col2, d_col3 = st.columns(3)

            with d_col1:
                if st.session_state.slides_result:
                    pptx_p = Path(st.session_state.slides_result["pptx_path"])
                    if pptx_p.exists():
                        with open(pptx_p, "rb") as f:
                            st.download_button(
                                label="📊 Download Presentation (.pptx)",
                                data=f.read(),
                                file_name=pptx_p.name,
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                type="primary",
                                use_container_width=True,
                            )
            with d_col2:
                if st.session_state.summary_result:
                    st.download_button(
                        label="📝 Download Summary (.md)",
                        data=st.session_state.summary_result["summary"],
                        file_name=f"Summary_{st.session_state.paper_filename.replace('.pdf', '')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
            with d_col3:
                if st.session_state.viva_result:
                    st.download_button(
                        label="🎓 Download Viva Guide (.md)",
                        data=st.session_state.viva_result["viva_guide"],
                        file_name=f"Viva_Prep_{st.session_state.paper_filename.replace('.pdf', '')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Paper Summary
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📝 Paper Summary":
    st.title("📝 Research Paper Summary")
    st.markdown("Generated by **Agent 1: Summarizer Agent** via LangChain LCEL & ChromaDB retrieval.")

    if not st.session_state.paper_loaded:
        st.warning("⚠️ Please upload a research paper first from the sidebar.")
    else:
        st.info(f"Active Paper: **{st.session_state.paper_filename}** ({st.session_state.total_pages} pages)")

        if st.session_state.summary_result is None:
            if st.button("Generate Structured Summary", type="primary"):
                with st.status("Summarizer Agent is working...", expanded=True) as status:
                    status.write("🔍 Searching and validating relevant paper chunks...")
                    agent = SummarizerAgent(vector_store=vstore)
                    
                    def on_progress(msg):
                        status.write(f"⚙️ {msg}")

                    res = agent.run(collection_id=st.session_state.collection_id, progress_callback=on_progress)
                    status.update(label="✅ Summary generated successfully!", state="complete", expanded=False)
                    st.session_state.summary_result = res
                st.rerun()
        else:
            res = st.session_state.summary_result

            # Quality and source metadata badges
            q_class = "status-optimal" if res["quality"] == "optimal" else "status-limited"
            st.markdown(
                f'<span class="status-pill {q_class}">Context Quality: {res["quality"].upper()}</span> '
                f'• Method: `{"Batched Map-Reduce" if "map_reduce" in res.get("method", "") else "Single-Pass LCEL"}` '
                f'• Source Pages: `{res["source_pages"]}`',
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown(res["summary"])

            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.download_button(
                    label="📥 Download Summary (.md)",
                    data=res["summary"],
                    file_name=f"Summary_{st.session_state.paper_filename.replace('.pdf', '')}.md",
                    mime="text/markdown",
                )
            with col_b:
                if st.button("🔄 Re-generate Summary"):
                    st.session_state.summary_result = None
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Concept Extraction
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔬 Concept Extraction":
    st.title("🔬 Concept & Methodology Extractor")
    st.markdown("Generated by **Agent 2: Concept-Extractor Agent** to extract technical terminology, algorithms, and experimental results.")

    if not st.session_state.paper_loaded:
        st.warning("⚠️ Please upload a research paper first from the sidebar.")
    else:
        st.info(f"Active Paper: **{st.session_state.paper_filename}**")

        if st.session_state.concepts_result is None:
            if st.button("Extract Concepts & Terminology", type="primary"):
                with st.status("Concept Agent is working...", expanded=True) as status:
                    status.write("🔍 Retrieving algorithmic and methodology chunks from ChromaDB...")
                    agent = ConceptAgent(vector_store=vstore)
                    status.write("🧠 Analyzing mathematical formulations and metrics with qwen2.5-coder:7b...")
                    res = agent.run(collection_id=st.session_state.collection_id)
                    status.update(label="✅ Concepts extracted successfully!", state="complete", expanded=False)
                    st.session_state.concepts_result = res
                st.rerun()
        else:
            res = st.session_state.concepts_result
            st.markdown(f"• **Source Pages Cited:** `{res['source_pages']}`")
            st.markdown("---")
            st.markdown(res["concepts"])

            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.download_button(
                    label="📥 Download Concepts (.md)",
                    data=res["concepts"],
                    file_name=f"Concepts_{st.session_state.paper_filename.replace('.pdf', '')}.md",
                    mime="text/markdown",
                )
            with col_b:
                if st.button("🔄 Re-extract Concepts"):
                    st.session_state.concepts_result = None
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Presentation Generator
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Presentation Generator":
    st.title("📊 PowerPoint Presentation Generator")
    st.markdown("Generated by **Agent 3: Slide-Generator Agent** using python-pptx (16:9 widescreen format).")

    if not st.session_state.paper_loaded:
        st.warning("⚠️ Please upload a research paper first from the sidebar.")
    else:
        st.info(f"Active Paper: **{st.session_state.paper_filename}**")

        if st.session_state.slides_result is None:
            st.markdown("""
            **Context Reuse Optimization:**
            The Slide Agent automatically incorporates output from the Summarizer and Concept agents
            if already computed, avoiding duplicate retrieval and saving compute.
            """)
            if st.button("Generate 12-Slide Presentation (.pptx)", type="primary"):
                with st.status("Slide Agent is working...", expanded=True) as status:
                    status.write("📊 Reusing synthesized summary and concepts context...")
                    agent = SlideAgent(vector_store=vstore)
                    prior_sum = st.session_state.summary_result["summary"] if st.session_state.summary_result else None
                    prior_con = st.session_state.concepts_result["concepts"] if st.session_state.concepts_result else None

                    status.write("🧠 Structuring 12 academic slides via LCEL...")
                    res = agent.run(
                        collection_id=st.session_state.collection_id,
                        prior_summary=prior_sum,
                        prior_concepts=prior_con,
                        paper_title=st.session_state.paper_title,
                    )
                    status.write(f"📁 Built presentation with {res['slide_count']} slides using python-pptx!")
                    status.update(label="✅ Presentation generated successfully!", state="complete", expanded=False)
                    st.session_state.slides_result = res
                st.rerun()
        else:
            res = st.session_state.slides_result
            pptx_path = Path(res["pptx_path"])

            # Download Button
            if pptx_path.exists():
                with open(pptx_path, "rb") as f:
                    pptx_bytes = f.read()

                st.success(f"✅ Generated {res['slide_count']} slides successfully! File size: {round(len(pptx_bytes)/1024, 1)} KB")
                st.download_button(
                    label="📥 Download Presentation (.pptx)",
                    data=pptx_bytes,
                    file_name=pptx_path.name,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    type="primary",
                )

            st.divider()
            st.markdown("### 🖥️ Slide Outline Preview")
            for s in res["slides"]:
                with st.expander(f"Slide {s.get('slide_num', '')}: {s.get('title', '')}", expanded=(s.get('slide_num') in (1, 2))):
                    for b in s.get("bullets", []):
                        st.markdown(f"• {b}")

            if st.button("🔄 Re-generate Slides"):
                st.session_state.slides_result = None
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Viva Preparation
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🎓 Viva Preparation":
    st.title("🎓 Viva & Defense Examination Agent")
    st.markdown("Generated by **Agent 4: Viva Agent** simulating external examiner questions across three difficulty tiers.")

    if not st.session_state.paper_loaded:
        st.warning("⚠️ Please upload a research paper first from the sidebar.")
    else:
        st.info(f"Active Paper: **{st.session_state.paper_filename}**")

        if st.session_state.viva_result is None:
            if st.button("Generate Examiner Questions & Suggested Answers", type="primary"):
                with st.status("Viva Agent is working...", expanded=True) as status:
                    status.write("🔍 Retrieving limitations, experimental baselines, and trade-offs...")
                    agent = VivaAgent(vector_store=vstore)
                    status.write("🎓 Simulating examiner questioning (Basic, Intermediate, Advanced) with qwen2.5-coder:7b...")
                    res = agent.run(collection_id=st.session_state.collection_id)
                    status.update(label="✅ Viva preparation guide generated successfully!", state="complete", expanded=False)
                    st.session_state.viva_result = res
                st.rerun()
        else:
            res = st.session_state.viva_result
            st.markdown(f"• **Source Pages Examined:** `{res['source_pages']}`")
            st.markdown("---")
            st.markdown(res["viva_guide"])

            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.download_button(
                    label="📥 Download Viva Guide (.md)",
                    data=res["viva_guide"],
                    file_name=f"Viva_Prep_{st.session_state.paper_filename.replace('.pdf', '')}.md",
                    mime="text/markdown",
                )
            with col_b:
                if st.button("🔄 Re-generate Viva Questions"):
                    st.session_state.viva_result = None
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: Ask Your Paper
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "💬 Ask Your Paper":
    st.title("💬 Ask Your Paper (Intent Detection Demo)")
    st.markdown("""
    Type any question about the paper. The **Intent Router** detects what you want and routes
    to the correct specialized agent, or answers via the grounded Q&A agent.
    """)

    if not st.session_state.paper_loaded:
        st.warning("⚠️ Please upload a research paper first from the sidebar.")
    else:
        user_query = st.chat_input("Ask a question (e.g. 'Summarize this', 'What algorithm was used?', 'What are the viva questions?')")

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if "intent_badge" in msg:
                    st.markdown(f'<span class="agent-badge">{msg["intent_badge"]}</span>', unsafe_allow_html=True)
                st.markdown(msg["content"])

        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Detect Intent
            intent_res = detect_intent(user_query)
            badge_text = f"🎯 Routed to: {intent_res.agent_name} (Confidence: {int(intent_res.confidence * 100)}%)"

            with st.chat_message("assistant"):
                st.markdown(f'<span class="agent-badge">{badge_text}</span>', unsafe_allow_html=True)
                st.caption(f"Reason: {intent_res.reason}")

                with st.spinner(f"Running {intent_res.agent_name}..."):
                    if intent_res.intent == "summarize":
                        agent = SummarizerAgent(vector_store=vstore)
                        ans = agent.run(st.session_state.collection_id)["summary"]
                    elif intent_res.intent == "concepts":
                        agent = ConceptAgent(vector_store=vstore)
                        ans = agent.run(st.session_state.collection_id)["concepts"]
                    elif intent_res.intent == "viva":
                        agent = VivaAgent(vector_store=vstore)
                        ans = agent.run(st.session_state.collection_id)["viva_guide"]
                    elif intent_res.intent == "slides":
                        agent = SlideAgent(vector_store=vstore)
                        res = agent.run(st.session_state.collection_id, paper_title=st.session_state.paper_title)
                        ans = f"✅ Created 12-slide presentation! You can preview and download it in the **Presentation Generator** tab."
                    else:
                        # General Q&A
                        agent = QAAgent(vector_store=vstore)
                        ans = agent.answer_question(user_query, st.session_state.collection_id)["answer"]

                st.markdown(ans)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ans,
                    "intent_badge": badge_text,
                })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: About Project
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "ℹ️ About Project":
    st.title("ℹ️ About PaperPilot AI")
    st.markdown("""
    **Project Title:** Paper-to-Presentation & Viva Agent  
    **Academic Level:** BE Artificial Intelligence and Data Science (Semester 6)  
    **Technology Stack:** Python, Streamlit, LangChain, LCEL, ChromaDB, Ollama (`qwen2.5-coder:7b`), PyMuPDF, python-pptx  
    """)

    st.markdown("### 🏆 Acceptance Criteria Checklist")
    criteria = [
        ("Requirement 1: LLM Integration", "Local Ollama (qwen2.5-coder:7b) primary with Groq cloud fallback via utils/llm_factory.py."),
        ("Requirement 2: Interactive GUI", "Complete Streamlit dashboard with upload, interactive navigation, and instant downloads."),
        ("Requirement 3: Prompt Engineering", "Distinct, documented system and user prompt templates across all 4 agents in prompts/ folder."),
        ("Requirement 4: LangChain Core Components", "ChatPromptTemplate, SystemMessage/HumanMessage, and LCEL pipe (|) chains throughout."),
        ("Requirement 5: Agentic RAG", "Targeted agent queries, context validation, self-healing query expansion, and page tracking."),
        ("Requirement 6: Intent Detection & Routing", "4-tier router (explicit, exact phrase, weighted keyword scoring, and fallback) in core/intent_router.py."),
        ("Requirement 7: Vector Database", "Local ChromaDB with all-MiniLM-L6-v2 embeddings, similarity search, and collection management."),
    ]

    for req, desc in criteria:
        with st.expander(f"✅ {req}", expanded=True):
            st.markdown(desc)
