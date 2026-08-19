import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# Load environment variables from backend/.env
# ---------------------------------------------------------

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


GATEWAY_URL = os.getenv("GATEWAY_URL")
API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in backend/.env")

if not GATEWAY_URL:
    raise ValueError("GATEWAY_URL is not set in backend/.env")


# ---------------------------------------------------------
# Create OpenAI client
#
# The organization gateway is used instead of the
# public OpenAI endpoint.
# ---------------------------------------------------------

client = OpenAI(
    base_url=GATEWAY_URL,
    api_key=API_KEY
)


# ---------------------------------------------------------
# Model
#
# Change this if your organization gateway requires
# a specific model name.
# ---------------------------------------------------------

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_system_prompt(knowledge_chunks: list[str]) -> str:

    if knowledge_chunks:
        knowledge_block = "\n".join(
            f"- {chunk}" for chunk in knowledge_chunks
        )
    else:
        knowledge_block = (
            "(no directly relevant company information found)"
        )

    return f"""
You are a helpful company assistant chatbot.

Answer the user's question using the conversation history
and company information when relevant.

If the company information does not cover the question,
say so honestly and answer generally if you can.

Keep answers concise, clear and friendly.

Company Information:
{knowledge_block}
"""


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

    print("\n========== OPENAI REQUEST ==========")
    print("Gateway:", GATEWAY_URL)
    print("Model:", MODEL)
    print("Question:", question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ],
        max_tokens=1000
    )

    print("OpenAI response received.")

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("OpenAI returned an empty response")

    return answer