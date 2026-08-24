import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(ENV_PATH)


GATEWAY_URL = os.getenv("GATEWAY_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


if not GATEWAY_URL:
    raise ValueError("GATEWAY_URL is not set")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


# --------------------------------------------------
# OpenAI client
# --------------------------------------------------

client = OpenAI(
    base_url=GATEWAY_URL,
    api_key=API_KEY,
)


# Gateway-supported embedding model
EMBEDDING_MODEL = "text-embedding-3-small"


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks.

    Example:

        texts = [
            "Employees are entitled to annual leave.",
            "Leave requests must be approved by the manager."
        ]

    Returns:

        [
            [0.012, -0.032, ...],
            [0.021, -0.014, ...]
        ]
    """

    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # OpenAI returns embeddings associated with an index.
    # Sorting makes sure the output order matches the input order.
    data = sorted(response.data, key=lambda item: item.index)

    return [item.embedding for item in data]


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single piece of text.
    """

    embeddings = get_embeddings([text])

    if not embeddings:
        raise ValueError("Embedding generation returned no result")

    return embeddings[0]