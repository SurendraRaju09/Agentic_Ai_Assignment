"""
agents/viva_agent.py — Agent 4: Academic Viva & Defense Preparation Agent
========================================================================
PURPOSE:
    Anticipates likely questions from an external academic examiner based on the uploaded paper.
    Generates structured Q&A across 3 tiers (Basic, Intermediate, Advanced) with suggested answers
    grounded in the paper's text.
    Implements Requirement 3, 4, and 5.
"""

from agents.base_agent import BaseAgent
from prompts.viva_prompts import viva_prompt
from config import settings


class VivaAgent(BaseAgent):
    """Specialized Agent simulating academic viva examinations."""

    # Targeted query to unearth potential critique areas, limitations, and design trade-offs
    VIVA_QUERY = (
        "limitations assumptions trade-offs threats to validity baseline comparison "
        "hyperparameters dataset bias future work experimental errors failure cases"
    )

    def run(self, collection_id: str, **kwargs) -> dict:
        """
        Executes the viva agent examination workflow.
        
        Args:
            collection_id: ChromaDB collection identifier for the uploaded paper.
            
        Returns:
            dict: Structured viva questions and answers, source pages, and context quality.
        """
        # Step 1: Agentic retrieval targeting critical evaluation and limitation chunks
        val_ctx = self.retrieve_with_validation(
            query=self.VIVA_QUERY,
            collection_id=collection_id,
            k=settings.VIVA_K,
            min_pages=3,
        )

        if val_ctx.quality == "empty":
            return {
                "viva_guide": "Could not generate viva questions because no relevant text was found in the paper.",
                "source_pages": [],
                "quality": "empty",
            }

        # Step 2: Build and invoke LCEL chain
        chain = self._build_lcel_chain(viva_prompt)
        viva_output = chain.invoke({"context": val_ctx.formatted_text})

        return {
            "viva_guide": viva_output,
            "source_pages": val_ctx.source_pages,
            "quality": val_ctx.quality,
            "chunks_retrieved": val_ctx.retrieval_count,
        }
