"""
agents/qa_agent.py — General Paper Question-Answering Agent
===========================================================
PURPOSE:
    Answers arbitrary user questions regarding the uploaded paper.
    Grounds answers strictly in ChromaDB retrieved context. If not found, explicitly states so.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from agents.base_agent import BaseAgent
from config import settings

QA_SYSTEM = """You are an academic assistant answering questions about an uploaded research paper.
RULES:
1. Answer strictly using the provided paper excerpts.
2. If the answer is not contained in the text, respond clearly:
   "Based on the provided research paper, this information is not mentioned or available."
3. Cite the source page numbers where you found the information.
"""

QA_HUMAN = """Relevant paper excerpts:
{context}

User Question: {question}

Provide an accurate, grounded answer:"""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(QA_SYSTEM),
    HumanMessagePromptTemplate.from_template(QA_HUMAN),
])


class QAAgent(BaseAgent):
    """Answers general questions about the paper using similarity search."""

    def answer_question(self, question: str, collection_id: str) -> dict:
        val_ctx = self.retrieve_with_validation(
            query=question,
            collection_id=collection_id,
            k=settings.MAX_CHUNKS_PER_RETRIEVAL,
            min_pages=1,
        )

        if val_ctx.quality == "empty":
            return {
                "answer": "No relevant excerpts could be retrieved from the paper.",
                "source_pages": [],
                "quality": "empty",
            }

        chain = self._build_lcel_chain(qa_prompt)
        response = chain.invoke({
            "context": val_ctx.formatted_text,
            "question": question,
        })

        return {
            "answer": response,
            "source_pages": val_ctx.source_pages,
            "quality": val_ctx.quality,
        }

    def run(self, collection_id: str, **kwargs) -> dict:
        question = kwargs.get("question", "What is the primary contribution of this paper?")
        return self.answer_question(question=question, collection_id=collection_id)
