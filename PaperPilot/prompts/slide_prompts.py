"""
prompts/slide_prompts.py — Slide Generator Agent Prompt Templates
=================================================================
PURPOSE:
    Transforms synthesized research paper content into a 12-slide presentation outline.
    Implements Requirement 3 & 4.

PROMPT ENGINEERING RATIONALE:
    - Presentation Constraints: Enforces concise bullet points (max 15 words per bullet)
      to avoid wall-of-text slides.
    - Standard Slide Deck Structure: Strictly aligns with the 12 required academic presentation slides:
      1. Title & Authors
      2. Introduction
      3. Problem Statement
      4. Objectives
      5. Literature Review / Background
      6. Proposed Methodology
      7. System Architecture
      8. Implementation Details
      9. Results and Discussion
      10. Conclusion
      11. Future Scope
      12. References / Acknowledgements
    - Output Parsing: Instructs the model to output a clean markdown format (SLIDE N: Title / - bullet points)
      that can be reliably parsed into python-pptx objects.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

SLIDE_SYSTEM_PROMPT = """You are an academic presentation design specialist.
Your goal is to convert research paper content into a clean, 12-slide PowerPoint presentation outline.

RULES:
1. Generate EXACTLY 12 slides corresponding to the standard academic structure.
2. Under each slide, provide 3 to 5 concise, impactful bullet points.
3. Keep bullet points brief (under 15 words each). DO NOT write long paragraphs.
4. Base all slide content on the provided paper summary, concepts, and excerpts.
5. Format each slide strictly as:
   SLIDE <number>: <Slide Title>
   - <Bullet point 1>
   - <Bullet point 2>
   - <Bullet point 3>
"""

SLIDE_HUMAN_TEMPLATE = """Paper Context:
--- SUMMARY & CONCEPTS ---
{summary_context}

--- ADDITIONAL RETRIEVED EXCERPTS ---
{retrieved_context}
--- END CONTEXT ---

Generate the 12 slides:
SLIDE 1: Title & Overview
SLIDE 2: Introduction
SLIDE 3: Problem Statement
SLIDE 4: Objectives & Scope
SLIDE 5: Literature Review & Background
SLIDE 6: Proposed Methodology
SLIDE 7: System Architecture
SLIDE 8: Implementation Details
SLIDE 9: Results & Performance Evaluation
SLIDE 10: Conclusion & Contributions
SLIDE 11: Limitations & Future Scope
SLIDE 12: Key References & Takeaways
"""

slide_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SLIDE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(SLIDE_HUMAN_TEMPLATE),
])
