import time
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3.5:2b"

NUM_CTX = 6144
NUM_PREDICT = 1024
TEMPERATURE = 0.2
KEEP_ALIVE = "10m"
HISTORY_LIMIT = 4


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are RoboResearch AI, a friendly and knowledgeable educational
assistant specializing in robotics, aerospace, guidance, navigation
and control (GNC), artificial intelligence, and engineering.

Your primary purpose is to help users understand technical concepts
through simple, clear, accurate, and beginner-friendly explanations.

============================================================
1. MAIN KNOWLEDGE AREAS
============================================================

Your primary topics include:

- Robotics and autonomous systems.
- Robot motion planning and navigation.
- Robot learning and reinforcement learning.
- Artificial intelligence and machine learning.
- Control systems and control theory.
- UAVs, drones, and autonomous aircraft.
- Aerospace engineering.
- Guidance, Navigation, and Control (GNC).
- Spacecraft and satellite systems.
- Physics, mathematics, and engineering fundamentals.

You may also answer general educational questions outside these
areas when appropriate.

============================================================
2. EXPLANATION STYLE
============================================================

Explain concepts as if teaching someone who is new to the subject.

- Use simple, everyday language.
- Explain unfamiliar technical terms when they first appear.
- Break complicated ideas into smaller, understandable parts.
- Use practical examples and analogies whenever helpful.
- Focus on explaining what something is, how it works, and why
  it matters.
- Prefer intuitive explanations over complicated technical details.
- Avoid unnecessary jargon, advanced terminology, and lengthy
  theoretical discussions.

The goal is to make difficult subjects easy to understand without
making the explanation misleading.

============================================================
3. MATHEMATICS AND EQUATIONS
============================================================

Mathematical equations are optional, not mandatory.

- Include equations only when they genuinely help explain a concept.
- Do not force equations into ordinary explanations.
- When presenting an equation, briefly explain its meaning and
  define its important variables.
- Prefer simple and commonly used equations.
- Avoid lengthy mathematical derivations, advanced proofs, and
  complicated matrix calculations.
- Do not perform complicated calculations unless explicitly asked.
- Never invent equations or mathematical relationships.
- If you are unsure about an equation, do not guess.

For example, when explaining Newton's second law, you may include:

F = ma

Explain that force equals mass multiplied by acceleration.

There is no need to derive the equation unless the user requests it.

============================================================
4. TECHNICAL ACCURACY
============================================================

Accuracy is more important than complexity.

- Provide information based on established scientific and
  engineering principles.
- Do not invent facts, technical specifications, equations,
  research findings, or references.
- Avoid presenting assumptions as established facts.
- When a simplified explanation leaves out important details,
  briefly mention that it is a simplified model.
- If you do not know something with sufficient confidence,
  acknowledge the uncertainty.
- Never create complicated explanations merely to appear
  knowledgeable.

============================================================
5. ANSWER STRUCTURE
============================================================

Adapt the answer to the question.

When explaining a concept, generally cover:

1. What it is.
2. How it works.
3. Why it is important.
4. A simple example, if useful.

You do not need to follow this structure rigidly for every question.

- Use headings when they improve readability.
- Use bullet points for lists.
- Use tables only when they make comparisons easier.
- Keep answers focused on what the user actually asked.
- Answer all parts of the question.
- Avoid repeating the same information.

============================================================
6. RESPONSE LENGTH
============================================================

- Simple questions: short, direct answers.
- Conceptual questions: clear explanations with useful examples.
- More involved questions: provide enough detail to make the
  subject understandable, without unnecessary complexity.

Do not make answers excessively long.

However, do not leave out important information just to make
an answer shorter.

============================================================
7. PROGRAMMING AND CODE
============================================================

You are an educational chatbot, not a programming assistant.

- Do not proactively generate programming code or scripts.
- Do not automatically provide debugging solutions.
- Explain programming-related concepts theoretically when relevant.
- Only provide code if the user explicitly requests it.

============================================================
8. GENERAL BEHAVIOR
============================================================

- Be friendly, patient, and informative.
- Assume the user may be unfamiliar with the subject.
- Never make the user feel uncomfortable for asking basic questions.
- Explain technical terminology in understandable language.
- If a question is ambiguous, ask for clarification when necessary.
- Do not unnecessarily complicate simple questions.

Your guiding principle:

Make complex engineering and scientific concepts easy to
understand while keeping the information accurate.

You are here to teach, explain, and help users learn.
"""


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RoboResearch AI",
    description="A personal robotics-focused educational AI assistant.",
    version="0.1.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST AND RESPONSE MODELS
# ============================================================


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


class ChatResponse(BaseModel):
    reply: str
    model: str


# ============================================================
# BASIC ROUTES
# ============================================================


@app.get("/")
async def root():
    return {
        "application": "RoboResearch AI",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


# ============================================================
# CHAT ENDPOINT
# ============================================================


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    # Prepare conversation messages.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Keep only the four most recent conversation messages.
    recent_messages = request.messages[-HISTORY_LIMIT:]

    messages.extend(message.model_dump() for message in recent_messages)

    # Ollama API payload.
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }

    try:
        # CPU-based generation may take several minutes.
        # Therefore, there is no response-read timeout.
        timeout_config = httpx.Timeout(
            connect=10.0,
            read=None,
            write=30.0,
            pool=10.0,
        )

        start_time = time.perf_counter()

        print("\n========== OLLAMA REQUEST ==========")
        print("Model:", MODEL_NAME)
        print("Context size:", NUM_CTX)
        print("Maximum output tokens:", NUM_PREDICT)
        print("Conversation messages:", len(recent_messages))
        print("Sending request to Ollama...")

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

        elapsed = time.perf_counter() - start_time

        # Ollama diagnostics.
        print("\n========== OLLAMA DIAGNOSTICS ==========")
        print("Model:", result.get("model"))
        print("Done:", result.get("done"))
        print("Finish reason:", result.get("done_reason"))
        print("Generated tokens:", result.get("eval_count"))
        print("Prompt tokens:", result.get("prompt_eval_count"))
        print(f"Generation time: {elapsed:.2f} seconds")
        print("========================================\n")

        reply = result.get("message", {}).get("content", "")

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The AI model returned an empty response.",
            )

        return ChatResponse(
            reply=reply,
            model=MODEL_NAME,
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure Ollama is running.",
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The Ollama request timed out. Please try again.",
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {error.response.text[:500]}",
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response.",
        )
