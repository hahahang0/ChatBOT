import json

import httpx
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.chat_service import (
    build_chat_messages,
    generate_chat_response,
    groq_error_detail,
    list_available_models,
    resolve_chat_model,
    stream_chat_response,
)
from app.config import (
    CHAT_PROVIDER,
    CORS_ORIGINS,
    GROQ_MAX_COMPLETION_TOKENS,
    NUM_PREDICT,
)
from app.schemas import (
    ChatMessage,
    ChatResponse,
)

app = FastAPI(
    title="RoboResearch AI",
    description="A personal robotics-focused educational AI assistant.",
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST VALIDATION
# ============================================================

def parse_chat_messages(
    messages_json: str,
) -> list[ChatMessage]:

    try:
        raw_messages = json.loads(messages_json)

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid messages JSON.",
        ) from error

    if not isinstance(raw_messages, list):
        raise HTTPException(
            status_code=400,
            detail="Messages must be a JSON array.",
        )

    if not raw_messages:
        raise HTTPException(
            status_code=400,
            detail="At least one message is required.",
        )

    if len(raw_messages) > 40:
        raise HTTPException(
            status_code=400,
            detail="Too many messages in one request.",
        )

    try:
        messages = [
            ChatMessage.model_validate(message)
            for message in raw_messages
        ]

    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=error.errors(),
        ) from error

    if messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="The latest message must be a user message.",
        )

    return messages


# ============================================================
# ROOT AND HEALTH
# ============================================================

@app.get("/")
async def root():

    return {
        "application": "RoboResearch AI",
        "status": "running",
        "version": "0.2.0",
    }


@app.get("/api/health")
async def health():

    return {
        "status": "healthy",
    }


# ============================================================
# AVAILABLE MODELS
# ============================================================

@app.get("/api/models")
async def list_chat_models():

    return await list_available_models()


# ============================================================
# NORMAL CHAT
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
async def chat(
    messages: str = Form(...),
    model: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),  # noqa: B008
):

    recent_messages = parse_chat_messages(messages)

    selected_model = await resolve_chat_model(model)

    try:
        chat_messages, sources = await build_chat_messages(
            recent_messages,
            file,
        )
    finally:
        if file:
            await file.close()

    print("\n========== CHAT REQUEST ==========")
    print("Provider:", CHAT_PROVIDER)
    print("Model:", selected_model)
    print("Retrieved sources:", len(sources))
    print("==================================\n")

    try:

        reply = await generate_chat_response(
            selected_model,
            chat_messages,
        )

        return ChatResponse(
            reply=reply,
            model=selected_model,
            sources=sources,
        )

    except HTTPException:
        raise

    except httpx.ConnectError as error:

        detail = (
            "Could not connect to the Groq API."
            if CHAT_PROVIDER == "groq"
            else "Cannot connect to Ollama."
        )

        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from error

    except httpx.TimeoutException as error:

        raise HTTPException(
            status_code=504,
            detail=f"The {CHAT_PROVIDER} request timed out.",
        ) from error

    except httpx.HTTPStatusError as error:

        if CHAT_PROVIDER == "groq":

            detail = groq_error_detail(error)

            status_code = (
                429
                if error.response.status_code == 429
                else 401
                if error.response.status_code == 401
                else 403
                if error.response.status_code == 403
                else 502
            )

        else:

            try:
                await error.response.aread()
                error_detail = error.response.text[:500]

            except Exception:
                error_detail = (
                    "Could not read Ollama error details."
                )

            detail = (
                f"Ollama returned an error: {error_detail}"
            )

            status_code = 502

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from error


# ============================================================
# STREAMING CHAT
# ============================================================

@app.post("/api/chat/stream")
async def chat_stream(
    messages: str = Form(...),
    model: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),  # noqa: B008
):

    recent_messages = parse_chat_messages(messages)

    selected_model = await resolve_chat_model(model)

    try:
        chat_messages, sources = await build_chat_messages(
            recent_messages,
            file,
        )
    finally:
        if file:
            await file.close()

    print("\n========== CHAT STREAM REQUEST ==========")
    print("Provider:", CHAT_PROVIDER)
    print("Model:", selected_model)
    print("Retrieved sources:", len(sources))

    print(
        "Maximum completion tokens:",
        (
            GROQ_MAX_COMPLETION_TOKENS
            if CHAT_PROVIDER == "groq"
            else NUM_PREDICT
        ),
    )

    print("=========================================\n")

    async def generate():

        try:

            async for event in stream_chat_response(
                selected_model,
                chat_messages,
            ):

                if event.get("type") == "done":

                    event["sources"] = sources

                    # Keep the event compatible with the
                    # existing frontend's streaming format.

                    event.pop("reply", None)

                yield json.dumps(
                    event,
                    ensure_ascii=False,
                ) + "\n"

        except httpx.ConnectError:

            print("[Streaming] Connection error.")

            message = (
                "Could not connect to the Groq API."
                if CHAT_PROVIDER == "groq"
                else "Cannot connect to Ollama."
            )

            yield json.dumps({
                "type": "error",
                "message": message,
            }) + "\n"

        except httpx.TimeoutException:

            print("[Streaming] Request timed out.")

            yield json.dumps({
                "type": "error",
                "message": (
                    f"The {CHAT_PROVIDER} request timed out."
                ),
            }) + "\n"

        except httpx.HTTPStatusError as error:

            print(
                "[Streaming] Provider HTTP status:",
                error.response.status_code,
            )

            if CHAT_PROVIDER == "groq":
                message = groq_error_detail(error)

            else:
                message = (
                    f"Ollama returned HTTP "
                    f"{error.response.status_code}."
                )

            yield json.dumps({
                "type": "error",
                "message": message,
            }) + "\n"

        except httpx.RequestError:

            print("[Streaming] Network error.")

            yield json.dumps({
                "type": "error",
                "message": (
                    "A network error interrupted the AI response. "
                    "Please try again."
                ),
            }) + "\n"

        except Exception:

            print("[Streaming] Unexpected generation error.")

            yield json.dumps({
                "type": "error",
                "message": (
                    "An unexpected error occurred during generation."
                ),
            }) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
