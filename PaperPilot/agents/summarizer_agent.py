"""
agents/summarizer_agent.py — Agent 1: Research Paper Summarizer
==============================================================
PURPOSE:
    Extracts core research problem, objectives, methodology, key findings, conclusions,
    and grounded limitations from the research paper.
    Implements Requirement 3, 4, and 5 with batched Map-Reduce optimization.
"""

from agents.base_agent import BaseAgent
from prompts.summarizer_prompts import summarizer_prompt, map_prompt, reduce_prompt
from config import settings


class SummarizerAgent(BaseAgent):
    """Specialized Agent for structured academic paper summarization."""

    # Targeted query to pull chunks across all core paper sections
    SUMMARIZER_QUERY = (
        "research problem objective proposed methodology system architecture "
        "experimental results performance evaluation conclusion limitations"
    )

    def _batched_map_reduce(self, raw_chunks: list, collection_id: str, progress_callback=None) -> str:
        """
        Batched Map-Reduce: Selects representative chunks across the entire paper sequence,
        groups them into 3-4 batches, summarizes each batch, and synthesizes into the final summary.
        Keeps LLM calls to 3-4 total so generation finishes in 1-2 minutes even for 30-page papers.
        """
        map_chain = self._build_lcel_chain(map_prompt)
        reduce_chain = self._build_lcel_chain(reduce_prompt)

        # If a paper has many chunks (e.g. 100+ chunks), select up to 12 evenly-distributed
        # representative chunks across the beginning, middle, and end of the document.
        max_chunks = 12
        if len(raw_chunks) > max_chunks:
            step = len(raw_chunks) / max_chunks
            selected_chunks = [raw_chunks[int(i * step)] for i in range(max_chunks)]
        else:
            selected_chunks = raw_chunks

        batch_size = 3
        batches = [selected_chunks[i : i + batch_size] for i in range(0, len(selected_chunks), batch_size)]
        section_notes = []

        for idx, batch in enumerate(batches):
            pages = sorted(list(set(c.metadata.get("page", 0) for c in batch)))
            batch_text = "\n\n".join(c.page_content for c in batch)
            
            if progress_callback:
                progress_callback(f"Summarizing section {idx + 1} of {len(batches)} (Pages {pages})...")

            # Map step on the batch
            note = map_chain.invoke({
                "pages": ", ".join(map(str, pages)),
                "batch_text": batch_text,
            })
            section_notes.append(f"Pages {pages} Notes:\n{note}")

        if progress_callback:
            progress_callback("Synthesizing final comprehensive paper summary...")

        # Reduce step: synthesize all section notes
        combined_notes = "\n\n---\n\n".join(section_notes)
        final_summary = reduce_chain.invoke({"section_notes": combined_notes})
        return final_summary

    def run(self, collection_id: str, progress_callback=None, **kwargs) -> dict:
        """
        Executes the summarizer agent workflow.
        
        Args:
            collection_id: ChromaDB collection identifier for the uploaded paper.
            progress_callback: Optional callable for UI progress updates.
            
        Returns:
            dict: Structured summary text, source pages, and metadata.
        """
        # Step 1: Agentic retrieval with multi-page validation
        val_ctx = self.retrieve_with_validation(
            query=self.SUMMARIZER_QUERY,
            collection_id=collection_id,
            k=settings.SUMMARIZER_K,
            min_pages=3,
        )

        if val_ctx.quality == "empty":
            return {
                "summary": "Could not find any content in the specified research paper collection.",
                "source_pages": [],
                "quality": "empty",
                "method": "none",
            }

        # Step 2: Determine full paper vs targeted single-pass
        all_chunks = self.vector_store.get_all_documents(collection_id)
        total_text_len = sum(len(c.page_content) for c in all_chunks) if all_chunks else 0

        if len(all_chunks) > 6 or total_text_len > 4000:
            summary_text = self._batched_map_reduce(all_chunks, collection_id, progress_callback=progress_callback)
            method = "batched_map_reduce_all_chunks"
            source_pages = sorted(list(set(c.metadata.get("page", 0) for c in all_chunks if "page" in c.metadata)))
            chunks_processed = min(len(all_chunks), 12)
        else:
            if progress_callback:
                progress_callback("Generating structured summary via LCEL...")
            chain = self._build_lcel_chain(summarizer_prompt)
            summary_text = chain.invoke({"context": val_ctx.formatted_text})
            method = "single_pass"
            source_pages = val_ctx.source_pages
            chunks_processed = val_ctx.retrieval_count

        return {
            "summary": summary_text,
            "source_pages": source_pages,
            "quality": val_ctx.quality,
            "method": method,
            "chunks_retrieved": chunks_processed,
        }
