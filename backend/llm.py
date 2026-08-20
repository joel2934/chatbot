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

Keep answers concise and friendly.

Company Information:
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