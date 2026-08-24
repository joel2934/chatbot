import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


print("====================================")
print("OpenAI Gateway Configuration Test")
print("====================================")


# Load .env from backend folder
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


gateway_url = os.getenv("GATEWAY_URL")
api_key = os.getenv("OPENAI_API_KEY")

print("Gateway loaded:", bool(gateway_url))
print("API key loaded:", bool(api_key))
print("Model:", "gpt-4o-mini")
print("Gateway:", gateway_url)
print(
    "Key prefix:",
    api_key[:8] + "..." if api_key else None
)


if not gateway_url:
    raise ValueError("GATEWAY_URL is not set")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = OpenAI(
    base_url=gateway_url,
    api_key=api_key,
)


print("\nSending test request...")


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one short sentence."
        }
    ],
    max_tokens=500,
)


print("\n====================================")
print("SUCCESS")
print("====================================")
print("AI RESPONSE:")
print(response.choices[0].message.content)