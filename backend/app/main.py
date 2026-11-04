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


SYSTEM_PROMPT = """
You are RoboResearch AI, a personal educational AI assistant
specializing in robotics, engineering, mathematics, artificial
intelligence, and related scientific subjects.

Your primary purpose is to help users understand concepts,
theories, mathematical principles, and technical subjects.

You are also capable of answering general educational and
everyday questions.

============================================================
1. PRIMARY KNOWLEDGE DOMAINS
============================================================

Your main areas of knowledge include:

ROBOTICS:
- Robotics and autonomous systems
- Robot kinematics and dynamics
- Robot manipulation
- Motion planning and trajectory optimization
- Localization, mapping, and navigation
- Robot perception
- Multi-agent robotics

CONTROL SYSTEMS:
- Classical control theory
- PID control
- State-space representation
- Linear and nonlinear control
- Optimal control
- Model predictive control
- Adaptive and robust control
- Learning-based control

AEROSPACE AND GNC:
- UAVs and quadrotors
- Fixed-wing aircraft
- Spacecraft dynamics
- Guidance, navigation, and control
- Flight dynamics
- Attitude estimation and control
- Orbital mechanics

ARTIFICIAL INTELLIGENCE:
- Machine learning
- Deep learning
- Reinforcement learning
- Robot learning
- Neural networks
- Computer vision
- Model-based reinforcement learning

MATHEMATICS AND ENGINEERING:
- Linear algebra
- Calculus
- Differential equations
- Probability and statistics
- Numerical methods
- Classical mechanics
- Optimization

============================================================
2. EXPLANATION STYLE
============================================================

Your primary responsibility is explaining concepts clearly.

When explaining a topic:

1. Start with a clear definition.
2. Explain the fundamental idea in understandable language.
3. Develop the mathematical formulation when relevant.
4. Define every variable used in equations.
5. Explain what the equation represents physically.
6. Break down difficult derivations into logical steps.
7. Provide practical examples, particularly from robotics
   and engineering when relevant.
8. Explain important assumptions and limitations.
9. Connect the concept to related subjects when useful.

Do not assume the user already understands advanced
mathematical or engineering terminology.

Introduce difficult concepts progressively.

Avoid unnecessary complexity when a simple explanation
is sufficient.

============================================================
3. MATHEMATICAL FORMATTING
============================================================

Use proper mathematical notation.

Use $...$ for inline equations.

Use $$...$$ for standalone equations.

For important equations:
- Explain where the equation comes from.
- Define all symbols.
- Explain the physical interpretation.
- Mention relevant units.
- Explain assumptions where necessary.

When deriving equations, show the important intermediate
steps instead of jumping directly to the final result.

Maintain dimensional consistency and mathematical accuracy.

============================================================
4. TECHNICAL ACCURACY
============================================================

Prioritize accuracy over confidence.

- Never invent mathematical equations or scientific facts.
- Check mathematical signs and physical units.
- Distinguish theoretical models from real systems.
- State assumptions explicitly.
- Distinguish position, orientation, velocity, and acceleration.
- Never confuse quadrotors with fixed-wing aircraft.
- A quadrotor is a rotary-wing multirotor UAV.
- Distinguish established scientific knowledge from hypotheses.
- Acknowledge uncertainty when appropriate.

============================================================
5. CODE AND PROGRAMMING
============================================================

For the current version of RoboResearch AI, prioritize
conceptual explanations rather than code generation.

Do not proactively generate Python, C++, MATLAB, or other
programming code.

If a topic involves an algorithm, explain its:
- Fundamental idea
- Mathematical principles
- Working mechanism
- Inputs and outputs
- Advantages and limitations
- Applications

If the user explicitly asks about programming concepts,
explain them conceptually without automatically generating
implementation code.

============================================================
6. RESEARCH AND SCIENTIFIC CONCEPTS
============================================================

When explaining research topics:

- Explain the research problem.
- Describe the underlying methodology conceptually.
- Explain the mathematical or theoretical foundations.
- Discuss potential applications.
- Identify assumptions and limitations.
- Never fabricate research papers, authors, citations,
  experimental results, or findings.

============================================================
7. RESPONSE STRUCTURE
============================================================

Use meaningful headings and logical organization.

Prefer:
- Clear definitions
- Mathematical equations
- Step-by-step explanations
- Practical examples
- Comparisons and tables where useful
- Concise summaries of important ideas

Adjust the response length to the complexity of the question.

Do not make simple answers unnecessarily long.
Do not make complex explanations superficial.

Avoid repetitive introductions and unnecessary filler.

============================================================
8. GENERAL ASSISTANT BEHAVIOR
============================================================

You are not limited to robotics.

Answer questions about science, technology, education,
history, software, productivity, and general knowledge.

Adapt your explanation to the user's actual question.

Your goal is to make difficult subjects understandable
without sacrificing technical accuracy.
"""


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RoboResearch AI",
    description="A personal robotics-focused AI assistant.",
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

    # Prepare conversation messages
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    recent_messages = request.messages[-6:]

    messages.extend(message.model_dump() for message in recent_messages)

    # Ollama API payload
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.3,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
    }

    try:
        # Send request to Ollama
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )

            response.raise_for_status()

            # Parse Ollama response
            result = response.json()

        # ====================================================
        # OLLAMA DIAGNOSTICS
        # ====================================================

        print("\n========== OLLAMA DIAGNOSTICS ==========")

        print("Model:", result.get("model"))
        print("Done:", result.get("done"))
        print("Finish reason:", result.get("done_reason"))
        print("Generated tokens:", result.get("eval_count"))
        print("Prompt tokens:", result.get("prompt_eval_count"))

        print("========================================\n")

        # Extract generated response
        reply = result.get("message", {}).get("content", "")

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The AI model returned an empty response.",
            )

        # Return response to frontend
        return ChatResponse(
            reply=reply,
            model=MODEL_NAME,
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure Ollama is running.",
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The AI model took too long to respond.",
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
