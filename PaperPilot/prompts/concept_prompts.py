"""
prompts/concept_prompts.py — Concept Extractor Agent Prompt Templates
=====================================================================
PURPOSE:
    Extracts technical concepts, terminology, algorithms, contributions, and results.
    Implements Requirement 3 & 4.

PROMPT ENGINEERING RATIONALE:
    - Educational Clarity: Requires explaining technical terminology in understandable language
      so a student can easily explain them in a viva examination.
    - Grounding: Only extract algorithms and metrics actually present in the paper text.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

CONCEPT_SYSTEM_PROMPT = """You are an AI and Data Science educator and technical concept extractor.
Your role is to identify and explain the core theoretical and engineering concepts in a research paper.

RULES:
1. Identify actual technical terms, algorithms, architectures, and metrics from the provided text.
2. Explain each term clearly so a 6th-semester undergraduate can explain it during a technical viva.
3. Ground all contributions and results strictly in the provided text.
4. If a category (e.g. named algorithms) has no mentions in the paper, write "None explicitly named".
"""

CONCEPT_HUMAN_TEMPLATE = """Analyze the following paper excerpts:

--- BEGIN EXCERPTS ---
{context}
--- END EXCERPTS ---

Extract technical information in the following structured format:

### 1. Core Technical Concepts
- **[Concept Name]**: [Clear explanation of what it is and why it is used in this paper]

### 2. Technical Terminology Glossary
- **[Term]**: [Definition and plain-language explanation]

### 3. Algorithms & Methodologies Used
- **[Algorithm/Architecture/Model]**: [Role and operational detail in the paper]

### 4. Main Contributions
- [Bullet points of novel contributions claimed by the authors]

### 5. Experimental Results & Performance Metrics
- [Specific datasets, benchmarks, evaluation metrics (e.g., accuracy, F1, latency, BLEU), and comparison against baselines]
"""

concept_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CONCEPT_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(CONCEPT_HUMAN_TEMPLATE),
])
