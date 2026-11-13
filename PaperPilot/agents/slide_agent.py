"""
agents/slide_agent.py — Agent 3: Slide Content Generator
========================================================
PURPOSE:
    Converts research paper findings into presentation-ready content structured into
    the standard 12 academic slides.
    Implements Context Reuse (takes prior agent outputs to avoid redundant queries)
    and interfaces with pptx_generator.py.
"""

import re
from agents.base_agent import BaseAgent
from prompts.slide_prompts import slide_prompt
from generators.pptx_generator import create_presentation_from_slides
from config import settings


class SlideAgent(BaseAgent):
    """Specialized Agent for structuring academic slide content and triggering PPTX creation."""

    SLIDE_QUERY = (
        "system architecture implementation details experimental setup "
        "hardware software hyperparameter evaluation baseline comparison future work"
    )

    @staticmethod
    def parse_slides_output(raw_text: str) -> list[dict]:
        """
        Parses LLM output into a structured list of slide dictionaries:
        [{"slide_num": 1, "title": "...", "bullets": [...]}, ...]
        """
        slides = []
        # Pattern looks for "SLIDE <num>: <Title>" followed by bullets
        slide_blocks = re.split(r'SLIDE\s+(\d+)[:\.\-]\s*', raw_text, flags=re.IGNORECASE)

        if len(slide_blocks) > 1:
            # slide_blocks[0] is preamble; subsequent elements are pairs of (number, content)
            for i in range(1, len(slide_blocks), 2):
                slide_num = int(slide_blocks[i])
                block_content = slide_blocks[i + 1].strip()

                lines = block_content.split('\n')
                title = lines[0].strip().strip('*#')
                bullets = []

                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith(('-', '*', '•')) or (len(line) > 2 and line[0].isdigit() and line[1] in ('.', ')')):
                        clean_bullet = re.sub(r'^[\-\*•\d\.\)\s]+', '', line).strip()
                        if clean_bullet:
                            bullets.append(clean_bullet)

                if not bullets and len(lines) > 1:
                    # Fallback for non-bulleted lines
                    bullets = [l.strip() for l in lines[1:] if l.strip()][:4]

                slides.append({
                    "slide_num": slide_num,
                    "title": title or f"Slide {slide_num}",
                    "bullets": bullets if bullets else ["Key technical points covered in paper."]
                })
        else:
            # Fallback parser if LLM did not strictly follow the "SLIDE N:" format
            sections = raw_text.split("###")
            for idx, sec in enumerate(sections[1:], 1):
                lines = sec.strip().split("\n")
                title = lines[0].strip()
                bullets = [l.strip("- *•").strip() for l in lines[1:] if l.strip().startswith(("-", "*", "•"))]
                slides.append({
                    "slide_num": idx,
                    "title": title or f"Slide {idx}",
                    "bullets": bullets or [lines[1].strip() if len(lines) > 1 else "Key findings."]
                })

        return slides

    def run(
        self,
        collection_id: str,
        prior_summary: str | None = None,
        prior_concepts: str | None = None,
        paper_title: str = "Research Presentation",
        **kwargs
    ) -> dict:
        """
        Generates slide content and builds the PPTX presentation.
        
        Args:
            collection_id: ChromaDB collection identifier.
            prior_summary: Optional summary text from Summarizer Agent (Context Reuse).
            prior_concepts: Optional concepts text from Concept Agent (Context Reuse).
            paper_title: Presentation title for Slide 1 and file naming.
            
        Returns:
            dict: Parsed slides list, PPTX file path, raw text, and source pages.
        """
        # Step 1: Context Reuse Optimization
        summary_context = ""
        if prior_summary:
            summary_context += f"--- PAPER SUMMARY ---\n{prior_summary}\n\n"
        if prior_concepts:
            summary_context += f"--- KEY CONCEPTS & RESULTS ---\n{prior_concepts}\n\n"

        # Step 2: Agentic retrieval only for architecture/implementation details
        val_ctx = self.retrieve_with_validation(
            query=self.SLIDE_QUERY,
            collection_id=collection_id,
            k=settings.MAX_CHUNKS_PER_RETRIEVAL,
            min_pages=2,
        )

        # Step 3: Run LCEL chain
        chain = self._build_lcel_chain(slide_prompt)
        raw_slide_text = chain.invoke({
            "summary_context": summary_context or "Summary not previously computed; rely on retrieved excerpts.",
            "retrieved_context": val_ctx.formatted_text,
        })

        # Step 4: Parse into structured slide dictionary
        parsed_slides = self.parse_slides_output(raw_slide_text)

        # Step 5: Generate PPTX file
        pptx_path = create_presentation_from_slides(
            slides=parsed_slides,
            presentation_title=paper_title,
        )

        return {
            "slides": parsed_slides,
            "raw_text": raw_slide_text,
            "pptx_path": str(pptx_path),
            "source_pages": val_ctx.source_pages,
            "quality": val_ctx.quality,
            "slide_count": len(parsed_slides),
        }
