"""
prompts/summarizer_prompts.py — Summarizer Agent Prompt Templates
==================================================================
PURPOSE:
    Provides structured system and user prompt templates for research paper summarization.
    Implements Requirement 3 (Prompt Engineering) & Requirement 4 (LangChain Core Components).

PROMPT ENGINEERING RATIONALE:
    - Structured Output: Instructs the model to output exact section headers
      (RESEARCH PROBLEM, OBJECTIVES, METHODOLOGY, KEY FINDINGS, CONCLUSION, LIMITATIONS).
    - Hallucination Control: Explicitly instructs that limitations or claims must only be
      stated if grounded in the provided paper context. If missing, state: "Not explicitly stated".
    - Batched Map Prompt: Focuses on condensing localized sections while retaining page numbers.
    - Reduce Prompt: Synthesizes multi-page mini-summaries into a coherent comprehensive overview.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ── 1. Single-Pass / Standard Summary Prompt ──────────────────────────────────
SUMMARIZER_SYSTEM_PROMPT = """You are an expert academic research assistant specialized in synthesizing scientific papers.
Your task is to analyze the provided research paper excerpts and generate an objective, highly accurate, and structured summary.

RULES:
1. Ground every statement strictly in the provided paper excerpts. Do not hallucinate or extrapolate beyond what is written.
2. If any requested item (like limitations or specific metrics) is not mentioned in the text, explicitly write: "Not explicitly stated in the paper."
3. Keep your language clear, academic, and accessible to a computer science undergraduate.
4. Follow the exact section headers below. Do not add or omit sections.
"""

SUMMARIZER_HUMAN_TEMPLATE = """Below are relevant excerpts from the research paper:

--- BEGIN EXCERPTS ---
{context}
--- END EXCERPTS ---

Generate a structured summary following this exact format:

### 1. Research Problem
(Explain the core problem, research gap, or challenge the paper addresses in 2-3 concise sentences.)

### 2. Objectives
(State the primary goals and aims of the study.)

### 3. Proposed Methodology
(Detail the algorithms, frameworks, architectures, datasets, and experimental methods used.)

### 4. Key Findings
(Highlight the main experimental results, metrics, benchmarks, and performance gains reported.)

### 5. Conclusion
(Summarize the final takeaways and academic contributions of the work.)

### 6. Limitations
(List explicit constraints, computational requirements, or assumptions mentioned by the authors. If none are cited, state: "Not explicitly stated in the paper.")
"""

summarizer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SUMMARIZER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(SUMMARIZER_HUMAN_TEMPLATE),
])


# ── 2. Batched Map-Reduce Prompts (For Long Papers) ───────────────────────────
MAP_SYSTEM_PROMPT = """You are an academic reader extracting key points from a specific section of a research paper.
Condense the excerpt into 3-5 bullet points covering any problem statement, methods, findings, or metrics mentioned.
Preserve specific model names, numbers, and facts. Avoid filler words."""

MAP_HUMAN_TEMPLATE = """Section excerpts [from Pages {pages}]:
{batch_text}

Provide concise key points from this section:"""

map_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(MAP_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(MAP_HUMAN_TEMPLATE),
])

REDUCE_SYSTEM_PROMPT = """You are a senior academic synthesizing section-by-section summaries into a final, unified paper summary.
Synthesize the provided section notes into a cohesive, structured summary. Do not lose specific metrics or methodology details."""

REDUCE_HUMAN_TEMPLATE = """Here are the section-by-section notes extracted across the entire research paper:

--- SECTION NOTES ---
{section_notes}
--- END SECTION NOTES ---

Produce the final comprehensive summary using the exact structure:
### 1. Research Problem
### 2. Objectives
### 3. Proposed Methodology
### 4. Key Findings
### 5. Conclusion
### 6. Limitations
"""

reduce_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REDUCE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(REDUCE_HUMAN_TEMPLATE),
])
