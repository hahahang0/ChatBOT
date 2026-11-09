
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY was not found in .env")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Explain how A* path planning works in robotics."
        }
    ],
}

try:
    response = httpx.post(
        url,
        headers=headers,
        json=payload,
        timeout=90.0,
    )

    response.raise_for_status()

    result = response.json()

    print("\nModel:", result["model"])
    print("\nResponse:")
    print(result["choices"][0]["message"]["content"])

except httpx.HTTPStatusError as error:
    print("API error:", error.response.status_code)
    print(error.response.text)

except httpx.RequestError as error:
    print("Connection error:", error)
