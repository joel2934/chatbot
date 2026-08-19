import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load backend/.env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


gateway_url = os.getenv("GATEWAY_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


print("====================================")
print("OpenAI Gateway Configuration Test")
print("====================================")

print("Gateway loaded:", bool(gateway_url))
print("API key loaded:", bool(api_key))
print("Model:", model)

if gateway_url:
    print("Gateway:", gateway_url)

if api_key:
    print("Key prefix:", api_key[:8] + "...")


if not gateway_url:
    raise ValueError("GATEWAY_URL is missing")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing")


client = OpenAI(
    base_url=gateway_url,
    api_key=api_key
)


print("\nSending test request...")


response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Say hello in one short sentence."
        }
    ],
    max_tokens=100
)


print("\n====================================")
print("AI RESPONSE")
print("====================================")

print(response.choices[0].message.content)