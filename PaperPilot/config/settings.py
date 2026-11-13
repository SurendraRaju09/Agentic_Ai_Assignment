"""
config/settings.py — PaperPilot AI Configuration Manager
=========================================================
PURPOSE:
    This is the SINGLE SOURCE OF TRUTH for all configuration values.
    Every other module imports constants from here — nobody calls os.getenv() directly.

HOW IT WORKS:
    1. python-dotenv reads the .env file
    2. Values are loaded into Python environment variables
    3. This module reads those variables and exports clean Python constants
    4. If a variable is missing from .env, a sensible default is used

WHY THIS MATTERS (Faculty Explanation):
    If you need to change the model from qwen2.5-coder:7b to something else,
    you only change ONE line in .env — not scattered throughout the codebase.
    This is the "Single Responsibility Principle" and "DRY (Don't Repeat Yourself)" in practice.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ─────────────────────────────────────────────────────────────
# Find the project root (the folder containing this config/ directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Load the .env file from the project root
# override=False means: don't overwrite variables already set in the OS environment
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ── LLM Configuration ─────────────────────────────────────────────────────────
# Which LLM provider to use: "ollama" (local) or "groq" (cloud API)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

# Ollama settings — your locally running model
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Groq settings — cloud API (optional fallback)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


# ── ChromaDB Configuration ─────────────────────────────────────────────────────
# Directory where ChromaDB persists vector embeddings on disk
# Using absolute path to avoid any working-directory confusion
CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))


# ── File Path Configuration ────────────────────────────────────────────────────
# Directory for generated PowerPoint files
OUTPUT_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "outputs")
# Directory for temporarily stored uploaded PDFs
DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

# Create these directories if they don't exist yet
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── RAG (Retrieval-Augmented Generation) Parameters ───────────────────────────
# How many characters each text chunk contains
# 1000 chars ≈ 150-200 words — good granularity for research papers
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))

# How many characters overlap between adjacent chunks
# Prevents important sentences from being split across chunk boundaries
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# Default number of chunks to retrieve from ChromaDB per query
MAX_CHUNKS_PER_RETRIEVAL: int = int(os.getenv("MAX_CHUNKS_PER_RETRIEVAL", "6"))

# Summarizer agent retrieves more chunks for broad paper coverage
SUMMARIZER_K: int = int(os.getenv("SUMMARIZER_K", "10"))

# Viva agent retrieves more chunks to cover limitations/future work
VIVA_K: int = int(os.getenv("VIVA_K", "8"))


# ── Embedding Model Configuration ─────────────────────────────────────────────
# The sentence-transformer model used to convert text to vectors
# "all-MiniLM-L6-v2" is 90MB, runs on CPU, good accuracy for academic text
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


# ── Validation ────────────────────────────────────────────────────────────────
def validate_config() -> list[str]:
    """
    Checks the configuration and returns a list of warnings.
    Called at application startup to catch misconfigurations early.
    """
    warnings = []

    if LLM_PROVIDER not in ("ollama", "groq"):
        warnings.append(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'ollama' or 'groq'.")

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        warnings.append("LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set in .env")

    if CHUNK_SIZE < 200:
        warnings.append(f"CHUNK_SIZE={CHUNK_SIZE} is very small. Recommended: 500-1500.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        warnings.append("CHUNK_OVERLAP must be less than CHUNK_SIZE.")

    return warnings


# ── Debug Helper ──────────────────────────────────────────────────────────────
def print_config() -> None:
    """Prints current configuration (safe — does not print API keys)."""
    print("=" * 50)
    print("PaperPilot AI — Current Configuration")
    print("=" * 50)
    print(f"  LLM Provider   : {LLM_PROVIDER}")
    print(f"  Model          : {OLLAMA_MODEL if LLM_PROVIDER == 'ollama' else GROQ_MODEL}")
    print(f"  Ollama URL     : {OLLAMA_BASE_URL}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  ChromaDB Dir   : {CHROMA_PERSIST_DIR}")
    print(f"  Output Dir     : {OUTPUT_DIR}")
    print(f"  Chunk Size     : {CHUNK_SIZE} chars")
    print(f"  Chunk Overlap  : {CHUNK_OVERLAP} chars")
    print(f"  Default K      : {MAX_CHUNKS_PER_RETRIEVAL}")
    print("=" * 50)

    issues = validate_config()
    if issues:
        print("⚠️  Configuration warnings:")
        for w in issues:
            print(f"   - {w}")
    else:
        print("✅ Configuration looks good.")
    print("=" * 50)
