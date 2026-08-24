from .embeddings import get_embedding
from .vector_store import search


# Number of chunks returned for each question.
DEFAULT_TOP_K = 5


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks
    for a user's question.
    """

    question = question.strip()

    if not question:
        return []

    # --------------------------------------------------
    # Convert question into an embedding
    # --------------------------------------------------

    query_embedding = get_embedding(question)

    # --------------------------------------------------
    # Semantic similarity search
    # --------------------------------------------------

    results = search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    return results


def retrieve_context(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    """
    Return only the text content of the retrieved chunks.

    This is convenient when passing the context
    to the LLM.
    """

    results = retrieve(
        question,
        top_k=top_k,
    )

    return [
        result["content"]
        for result in results
    ]


def retrieve_with_sources(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Return retrieved chunks along with their source
    document, page number, and similarity distance.

    Useful for displaying citations/debugging.
    """

    return retrieve(
        question,
        top_k=top_k,
    )