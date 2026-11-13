"""
prompts/viva_prompts.py — Viva Agent Prompt Templates
=====================================================
PURPOSE:
    Simulates academic examiner questioning based on the uploaded research paper.
    Implements Requirement 3 & 4.

PROMPT ENGINEERING RATIONALE:
    - Few-Shot Prompting: Includes example Q&A pairs to establish the expected level of technical rigor
      and answer structure.
    - Tiered Difficulty: Organizes questions into Basic (definitions, concepts),
      Intermediate (mechanism, comparisons), and Advanced (trade-offs, limitations, critical defense).
    - Suggested Answers: Provides answers grounded directly in the paper to prepare the student.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

VIVA_SYSTEM_PROMPT = """You are a senior academic professor and external viva examiner conducting an oral examination
for an AI & Data Science undergraduate student defending a research paper.

Your role:
1. Generate realistic, probing examiner questions at three distinct difficulty levels:
   - Basic Questions (4-5): Fundamental definitions, motivation, and "what is" questions.
   - Intermediate Questions (4-5): Methodology, algorithm design, hyperparameters, and "how does it work".
   - Advanced Questions (3-4): Trade-offs, baseline comparisons, limitations, and "why did you choose X over Y".
2. For EVERY question, provide a clear, accurate "Suggested Answer" grounded in the paper.
3. Conclude with a curated list of "Important Concepts to Revise Before the Exam".
"""

# Few-shot example embedded into the template
VIVA_HUMAN_TEMPLATE = """Research Paper Context:
--- BEGIN EXCERPTS ---
{context}
--- END EXCERPTS ---

Example format for each question:
Q: [Examiner Question]
A: [Concise, technically grounded suggested answer]

Generate the viva preparation guide following this exact structure:

### 1. Basic Level Questions (Foundations & Motivation)
(4-5 questions with suggested answers)

### 2. Intermediate Level Questions (Methodology & Architecture)
(4-5 questions with suggested answers)

### 3. Advanced Level Questions (Evaluation, Trade-offs & Critical Defense)
(3-4 questions with suggested answers)

### 4. Important Concepts to Revise Before the Viva
- [Key concept 1: 1-line refresher]
- [Key concept 2: 1-line refresher]
- [Key concept 3: 1-line refresher]
"""

viva_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(VIVA_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(VIVA_HUMAN_TEMPLATE),
])
