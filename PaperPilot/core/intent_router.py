"""
core/intent_router.py — Intent Detection and Agent Routing
==========================================================
PURPOSE:
    Determines user intent from input queries or navigation events and routes
    requests to the appropriate specialized agent.
    Implements Requirement 6 (Intent Detection and Routing).

ROUTING ARCHITECTURE:
    1. Explicit Navigation (confidence: 1.0)
    2. Exact Phrase Match (confidence: 0.95)
    3. Weighted Keyword Scoring (confidence: 0.6 - 0.9)
    4. Ambiguity Resolution & Fallback to Q&A (confidence: 0.3 - 0.5)
"""

from typing import NamedTuple


class IntentResult(NamedTuple):
    intent: str            # "summarize" | "concepts" | "slides" | "viva" | "qa"
    agent_name: str        # Human-readable agent name
    confidence: float      # 0.0 to 1.0
    reason: str            # Explanation of routing decision for faculty demo


# Exact phrases that directly map to specific intents
EXACT_PHRASES: dict[str, list[str]] = {
    "summarize": [
        "summarize this paper",
        "summarize this research paper",
        "give me a summary",
        "what is this paper about",
        "summarise the paper",
        "paper overview",
        "brief summary",
        "executive summary",
    ],
    "concepts": [
        "extract concepts",
        "list algorithms",
        "what concepts are used",
        "technical terms",
        "what methods are used",
        "key contributions",
        "explain methodology",
    ],
    "slides": [
        "create slides",
        "make presentation",
        "generate pptx",
        "create a powerpoint",
        "make slides",
        "build presentation",
        "slide deck",
    ],
    "viva": [
        "viva questions",
        "exam questions",
        "prepare for viva",
        "what questions will they ask",
        "examiner questions",
        "quiz me",
        "defense questions",
    ],
}

# Weighted keywords for ambiguous or partial text queries
KEYWORD_WEIGHTS: dict[str, dict[str, int]] = {
    "summarize": {
        "summary": 5,
        "summarize": 5,
        "summarise": 5,
        "overview": 4,
        "abstract": 3,
        "conclusion": 3,
        "problem": 2,
        "objective": 2,
    },
    "concepts": {
        "concept": 5,
        "terminology": 5,
        "algorithm": 5,
        "methodology": 4,
        "technique": 4,
        "contribution": 4,
        "architecture": 3,
        "metric": 3,
        "result": 2,
    },
    "slides": {
        "slide": 6,
        "slides": 6,
        "presentation": 6,
        "pptx": 6,
        "powerpoint": 6,
        "deck": 5,
    },
    "viva": {
        "viva": 6,
        "examiner": 5,
        "exam": 5,
        "quiz": 5,
        "defense": 5,
        "limitation": 3,
        "future work": 3,
    },
}

AGENT_NAME_MAP = {
    "summarize": "Summarizer Agent",
    "concepts": "Concept-Extractor Agent",
    "slides": "Slide-Generator Agent",
    "viva": "Viva Agent",
    "qa": "Paper Q&A Agent",
}


def detect_intent(query: str, explicit_intent: str | None = None) -> IntentResult:
    """
    Detects user intent from query text or explicit user selection.
    
    Args:
        query: User input query text.
        explicit_intent: Optional explicit navigation intent (e.g. from Streamlit sidebar).
        
    Returns:
        IntentResult: Detected intent, agent name, confidence score, and explanation.
    """
    # Priority 1: Explicit user action/selection
    if explicit_intent and explicit_intent in AGENT_NAME_MAP:
        return IntentResult(
            intent=explicit_intent,
            agent_name=AGENT_NAME_MAP[explicit_intent],
            confidence=1.0,
            reason=f"Direct user navigation to {AGENT_NAME_MAP[explicit_intent]}.",
        )

    clean_query = query.strip().lower()
    if not clean_query:
        return IntentResult(
            intent="qa",
            agent_name=AGENT_NAME_MAP["qa"],
            confidence=0.0,
            reason="Empty query provided; routed to default Q&A.",
        )

    # Priority 2: Exact phrase matching
    for intent, phrases in EXACT_PHRASES.items():
        for phrase in phrases:
            if phrase in clean_query:
                return IntentResult(
                    intent=intent,
                    agent_name=AGENT_NAME_MAP[intent],
                    confidence=0.95,
                    reason=f"Matched exact trigger phrase: '{phrase}'.",
                )

    # Priority 3: Weighted keyword scoring
    scores: dict[str, int] = {intent: 0 for intent in KEYWORD_WEIGHTS}
    matched_words: dict[str, list[str]] = {intent: [] for intent in KEYWORD_WEIGHTS}

    for intent, kw_dict in KEYWORD_WEIGHTS.items():
        for word, weight in kw_dict.items():
            if word in clean_query:
                scores[intent] += weight
                matched_words[intent].append(word)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = sorted_scores[0]
    runner_up_intent, runner_up_score = sorted_scores[1]

    # Ambiguity check: if top two intents have close scores, fall back to Q&A
    if top_score > 0 and (top_score - runner_up_score) <= 1 and runner_up_score >= 3:
        return IntentResult(
            intent="qa",
            agent_name=AGENT_NAME_MAP["qa"],
            confidence=0.5,
            reason=(
                f"Ambiguous intent between {top_intent} (score {top_score}) and "
                f"{runner_up_intent} (score {runner_up_score}); routed to Q&A for general answering."
            ),
        )

    # Threshold for routing
    if top_score >= 3:
        confidence = min(0.6 + (top_score / 20.0), 0.9)
        matches = ", ".join(matched_words[top_intent])
        return IntentResult(
            intent=top_intent,
            agent_name=AGENT_NAME_MAP[top_intent],
            confidence=round(confidence, 2),
            reason=f"Matched keywords [{matches}] with routing score {top_score}.",
        )

    # Priority 4: Fallback to General Q&A
    return IntentResult(
        intent="qa",
        agent_name=AGENT_NAME_MAP["qa"],
        confidence=0.35,
        reason="No specific agent intent keywords detected; routed to General Paper Q&A.",
    )
