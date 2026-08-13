import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# load .env located in the same folder as this file (works regardless of CWD)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

MODEL = "openrouter/free"


def build_system_prompt(knowledge_chunks: list[str]) -> str:

    if knowledge_chunks:
        knowledge_block = "\n".join(
            f"- {c}" for c in knowledge_chunks
        )
    else:
        knowledge_block = "(no directly relevant company information found)"

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


def generate_response(
    question: str,
    history: list[dict],
    knowledge_chunks: list[str]
) -> str:

    system_prompt = build_system_prompt(knowledge_chunks)

    messages = [
        {
            "role": m["role"],
            "content": m["content"]
        }
        for m in history
    ]

    messages.append({
        "role": "user",
        "content": question
    })

    print("\nSending request to OpenRouter...")
    print("Model:", MODEL)

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

    print("OpenRouter response received.")

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("OpenRouter returned an empty response")

    return answer