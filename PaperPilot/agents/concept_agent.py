"""
agents/concept_agent.py — Agent 2: Technical Concept & Methodology Extractor
============================================================================
PURPOSE:
    Extracts key technical terminology, algorithms, architectures, contributions,
    and experimental benchmarks from the paper.
    Implements Requirement 3, 4, and 5.
"""

from agents.base_agent import BaseAgent
from prompts.concept_prompts import concept_prompt
from config import settings


class ConceptAgent(BaseAgent):
    """Specialized Agent for technical concept extraction and plain-language explanation."""

    # Targeted query specifically seeking mathematical, algorithmic, and experimental text
    CONCEPT_QUERY = (
        "algorithm mathematical formulation architecture model framework "
        "terminology baseline dataset benchmark accuracy F1 score precision recall latency"
    )

    def run(self, collection_id: str, **kwargs) -> dict:
        """
        Executes the concept extraction agent workflow.
        
        Args:
            collection_id: ChromaDB collection identifier for the uploaded paper.
            
        Returns:
            dict: Extracted concepts text, source pages, and context quality.
        """
        # Step 1: Agentic retrieval targeting technical and algorithmic sections
        val_ctx = self.retrieve_with_validation(
            query=self.CONCEPT_QUERY,
            collection_id=collection_id,
            k=settings.MAX_CHUNKS_PER_RETRIEVAL,
            min_pages=2,
        )

        if val_ctx.quality == "empty":
            return {
                "concepts": "Could not extract concepts because no relevant content was found in the paper.",
                "source_pages": [],
                "quality": "empty",
            }

        # Step 2: Build and invoke LCEL chain
        chain = self._build_lcel_chain(concept_prompt)
        extracted_text = chain.invoke({"context": val_ctx.formatted_text})

        return {
            "concepts": extracted_text,
            "source_pages": val_ctx.source_pages,
            "quality": val_ctx.quality,
            "chunks_retrieved": val_ctx.retrieval_count,
        }
