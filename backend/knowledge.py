"""
knowledge.py
Deliberately NOT a vector database. This is step 1 of RAG: split the
knowledge file into chunks (one per line/paragraph) and pick the ones
that share the most words with the question. Once this makes sense,
swap this module for embeddings + a vector store — the rest of the
app (context builder, prompt, endpoints) doesn't need to change.
"""

import re
from pathlib import Path

KNOWLEDGE_PATH = Path(__file__).parent / "company_knowledge.txt"

STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "to", "in", "on", "for", "and",
    "what", "how", "many", "can", "i", "do", "does", "my", "our", "we",
    "you", "your", "it", "at", "be", "will", "with",
}


def _load_chunks() -> list[str]:
    if not KNOWLEDGE_PATH.exists():
        return []
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    # one chunk per non-empty line/paragraph
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    return chunks


def _keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def retrieve_relevant_knowledge(question: str, top_k: int = 3) -> list[str]:
    """Return up to top_k knowledge chunks that overlap most with the question."""
    chunks = _load_chunks()
    if not chunks:
        return []

    q_words = _keywords(question)
    if not q_words:
        return []

    scored = []
    for chunk in chunks:
        c_words = _keywords(chunk)
        overlap = len(q_words & c_words)
        if overlap > 0:
            scored.append((overlap, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def append_text_to_knowledge(text: str, source: str | None = None) -> None:
    """Append plain text to the `company_knowledge.txt` store as new chunks.

    The text will be split into non-empty paragraphs and appended to the
    file with an optional source header so it's easier to track later.
    """
    if not text:
        return

    header = f"\n\n# Source: {source}\n" if source else "\n\n"
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    if not chunks:
        return

    with open(KNOWLEDGE_PATH, "a", encoding="utf-8") as f:
        f.write(header)
        for p in chunks:
            f.write(p + "\n")
