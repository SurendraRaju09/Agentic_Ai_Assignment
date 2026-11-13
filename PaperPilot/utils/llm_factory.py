"""
utils/llm_factory.py — LLM Provider Factory
===========================================
PURPOSE:
    Provides a centralized factory function to instantiate LLM clients.
    Implements Requirement 1 (Flexible LLM Integration: Local Ollama primary, Groq fallback).

DESIGN PATTERN:
    Factory Pattern — Modules ask for an LLM without needing to know provider details,
    API keys, or connection parameters.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from config import settings


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """
    Returns a configured LangChain ChatModel based on application settings.
    
    Args:
        temperature: Optional float override for generation randomness (0.0 to 1.0).
                     If None, uses settings.OLLAMA_TEMPERATURE.
    
    Returns:
        BaseChatModel: An instantiated LangChain chat model (ChatOllama or ChatGroq).
    """
    temp = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE

    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temp,
        )
    elif settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in .env. Please provide a key or switch LLM_PROVIDER to 'ollama'.")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=temp,
            max_tokens=800,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. Use 'ollama' or 'groq'.")
