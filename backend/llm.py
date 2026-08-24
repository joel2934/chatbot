import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


GATEWAY_URL = os.getenv("GATEWAY_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


if not GATEWAY_URL:
    raise ValueError("GATEWAY_URL is not set")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


# --------------------------------------------------
# OpenAI client through organization gateway
# --------------------------------------------------

client = OpenAI(
    base_url=GATEWAY_URL,
    api_key=API_KEY,
)


# Organization gateway exposes gpt-4o-mini
MODEL = "gpt-4o-mini"


# --------------------------------------------------
# System prompt
# --------------------------------------------------

def build_system_prompt(
    knowledge_chunks: list[str],
) -> str:

    if knowledge_chunks:

        knowledge_block = "\n\n".join(
            f"Context {index + 1}:\n{chunk}"
            for index, chunk in enumerate(
                knowledge_chunks
            )
        )

    else:

        knowledge_block = (
            "No relevant company policy information "
            "was retrieved from the knowledge base."
        )

    return f"""
You are a helpful and professional company assistant chatbot.

Your job is to answer the user's question using the
retrieved company policy information provided below.

IMPORTANT RULES:

1. For company-specific questions, use the retrieved
   company policy context as your primary source.

2. Do not invent, guess, or assume company policies.

3. If the retrieved context does not contain enough
   information to answer a company-specific question,
   clearly say that the provided company documents do
   not contain that information.

4. You may answer general non-company questions using
   your general knowledge.

5. Keep answers concise, clear, professional, and friendly.

6. Do not mention internal implementation details such
   as ChromaDB, embeddings, vector databases, or RAG
   unless the user specifically asks about them.

7. When multiple retrieved contexts are provided, use
   the context that is most relevant to the question.

Retrieved Company Policy Context:

{knowledge_block}
"""


# --------------------------------------------------
# Generate response
# --------------------------------------------------

def generate_response(
    question: str,
    history: list[dict],
    knowledge_chunks: list[str]
) -> str:

    system_prompt = build_system_prompt(knowledge_chunks)

    messages = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in history
    ]

    messages.append({
        "role": "user",
        "content": question
    })

    print("\nSending request to OpenAI Gateway...")
    print("Model:", MODEL)
    print("Gateway:", GATEWAY_URL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ],
        max_tokens=500,
    )

    print("OpenAI Gateway response received.")

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "OpenAI Gateway returned an empty response"
        )

    return answer