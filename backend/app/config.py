import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "groq").strip().lower()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434",
).rstrip("/")

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen3.5:2b",
)

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "gemini",
).strip().lower()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

MAX_PDF_SIZE = 20 * 1024 * 1024

CHUNK_SIZE = 450
CHUNK_OVERLAP = 75

RAG_SELECTED_TOP_K = 5
MAX_SELECTED_DOCUMENT_CONTEXT_CHARS = 4500

GEMINI_TOKENS_PER_MINUTE_BUDGET = 25000
GEMINI_RATE_WINDOW_SECONDS = 60

GEMINI_RETRY_ATTEMPTS = 3
GEMINI_MAX_RETRY_DELAY = 300

EMBEDDING_BATCH_SIZE = 5
NUM_CTX = 6144
NUM_PREDICT = 1024

TEMPERATURE = 0.2

KEEP_ALIVE = "10m"

GROQ_MAX_COMPLETION_TOKENS = 2048
GROQ_REASONING_EFFORT = "low"

HISTORY_LIMIT = 4
MAX_CHAT_HISTORY_CHARS = 6000

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
