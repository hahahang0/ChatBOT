import asyncio
import json
import math
import re
import time
from collections import deque

import httpx

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MAX_RETRY_DELAY,
    GEMINI_RATE_WINDOW_SECONDS,
    GEMINI_RETRY_ATTEMPTS,
    GEMINI_TOKENS_PER_MINUTE_BUDGET,
    OLLAMA_URL,
)

_gemini_usage = deque()
_gemini_usage_lock = asyncio.Lock()


def estimate_gemini_tokens(texts: list[str]) -> int:
    total_characters = sum(len(text) for text in texts)

    return max(
        1,
        math.ceil(total_characters / 3),
    )


async def wait_for_gemini_token_budget(
    texts: list[str],
) -> None:

    estimated_tokens = estimate_gemini_tokens(texts)

    if estimated_tokens > GEMINI_TOKENS_PER_MINUTE_BUDGET:
        raise ValueError(
            "A Gemini embedding batch exceeds the configured "
            "internal token-per-minute budget. Reduce the "
            "embedding batch size or text chunk size."
        )

    while True:
        wait_seconds = 0.0

        async with _gemini_usage_lock:
            now = time.monotonic()

            while _gemini_usage:
                oldest_timestamp, _ = _gemini_usage[0]

                if (
                    now - oldest_timestamp
                    >= GEMINI_RATE_WINDOW_SECONDS
                ):
                    _gemini_usage.popleft()
                else:
                    break

            used_tokens = sum(
                token_count
                for _, token_count in _gemini_usage
            )

            if (
                used_tokens + estimated_tokens
                <= GEMINI_TOKENS_PER_MINUTE_BUDGET
            ):
                _gemini_usage.append((
                    now,
                    estimated_tokens,
                ))

                return

            tokens_to_release = (
                used_tokens
                + estimated_tokens
                - GEMINI_TOKENS_PER_MINUTE_BUDGET
            )

            released_tokens = 0

            for timestamp, token_count in _gemini_usage:
                released_tokens += token_count

                if released_tokens >= tokens_to_release:
                    wait_seconds = max(
                        1.0,
                        timestamp
                        + GEMINI_RATE_WINDOW_SECONDS
                        - now
                        + 1.0,
                    )
                    break

        print(
            "[Gemini Rate Limiter] Token budget reached. "
            f"Waiting {wait_seconds:.1f} seconds."
        )

        await asyncio.sleep(wait_seconds)


def get_embedding_error_detail(
    response: httpx.Response,
) -> str:

    try:
        return json.dumps(
            response.json(),
            indent=2,
            ensure_ascii=False,
        )
    except ValueError:
        return response.text or "No error details returned."


def is_daily_quota_error(error_detail: str) -> bool:

    normalized = error_detail.lower()

    daily_indicators = (
        "requestsperday",
        "request_per_day",
        "requests_per_day",
        "perday",
        "per_day",
        "per day",
        "daily quota",
        "daily limit",
        "perdayperproject",
        "perdayperprojectpermodel",
    )

    return any(
        indicator in normalized
        for indicator in daily_indicators
    )


def get_retry_delay(
    response: httpx.Response,
    error_detail: str,
    attempt: int,
) -> float:

    retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            return min(
                max(float(retry_after) + 1.0, 1.0),
                GEMINI_MAX_RETRY_DELAY,
            )
        except ValueError:
            pass

    try:
        data = json.loads(error_detail)

        details = (
            data.get("error", {})
            .get("details", [])
        )

        for detail in details:
            if not isinstance(detail, dict):
                continue

            type_name = detail.get("@type", "")

            if type_name.endswith("RetryInfo"):
                retry_delay = detail.get("retryDelay", "")

                match = re.search(
                    r"([\d.]+)s",
                    retry_delay,
                )

                if match:
                    return min(
                        max(
                            math.ceil(float(match.group(1))) + 1,
                            1,
                        ),
                        GEMINI_MAX_RETRY_DELAY,
                    )

    except (
        ValueError,
        TypeError,
        AttributeError,
    ):
        pass

    match = re.search(
        r"retry in ([\d.]+)\s*s",
        error_detail,
        re.IGNORECASE,
    )

    if match:
        return min(
            max(
                math.ceil(float(match.group(1))) + 1,
                1,
            ),
            GEMINI_MAX_RETRY_DELAY,
        )

    return min(
        2 ** (attempt + 1),
        30,
    )


async def generate_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:

    if not texts:
        return []

    embeddings = []

    timeout_config = httpx.Timeout(
        connect=15.0,
        read=120.0,
        write=30.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout_config,
    ) as client:

        for start in range(
            0,
            len(texts),
            EMBEDDING_BATCH_SIZE,
        ):

            batch = texts[
                start:start + EMBEDDING_BATCH_SIZE
            ]

            if EMBEDDING_PROVIDER == "gemini":

                if not GEMINI_API_KEY:
                    raise ValueError(
                        "GEMINI_API_KEY is not configured."
                    )

                url = (
                    "https://generativelanguage.googleapis.com/"
                    f"v1beta/models/{GEMINI_EMBEDDING_MODEL}:"
                    "batchEmbedContents"
                )

                payload = {
                    "requests": [
                        {
                            "model": (
                                f"models/{GEMINI_EMBEDDING_MODEL}"
                            ),
                            "content": {
                                "parts": [
                                    {"text": text}
                                ]
                            },
                            "taskType": task_type,
                        }
                        for text in batch
                    ]
                }

                response = None

                for attempt in range(GEMINI_RETRY_ATTEMPTS):

                    await wait_for_gemini_token_budget(batch)

                    response = await client.post(
                        url,
                        headers={
                            "x-goog-api-key": GEMINI_API_KEY,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )

                    if response.status_code != 429:
                        response.raise_for_status()
                        break

                    error_detail = get_embedding_error_detail(
                        response
                    )

                    print("[Gemini Embeddings] HTTP 429 response received.")

                    if is_daily_quota_error(error_detail):
                        raise httpx.HTTPStatusError(
                            "Gemini daily quota exhausted: "
                            + error_detail,
                            request=response.request,
                            response=response,
                        )

                    if attempt == GEMINI_RETRY_ATTEMPTS - 1:
                        raise httpx.HTTPStatusError(
                            "Gemini embedding request failed "
                            "after maximum retry attempts: "
                            + error_detail,
                            request=response.request,
                            response=response,
                        )

                    retry_delay = get_retry_delay(
                        response,
                        error_detail,
                        attempt,
                    )

                    print(
                        "[Gemini Embeddings] Rate limit reached. "
                        f"Retrying in {retry_delay} seconds."
                    )

                    await asyncio.sleep(retry_delay)

                if response is None:
                    raise RuntimeError(
                        "Gemini embedding request did not return "
                        "a response."
                    )

                result = response.json()

                batch_embeddings = [
                    item["values"]
                    for item in result.get("embeddings", [])
                ]

            elif EMBEDDING_PROVIDER == "ollama":

                response = await client.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": batch,
                    },
                )

                response.raise_for_status()

                result = response.json()

                batch_embeddings = result.get(
                    "embeddings",
                    [],
                )

            else:
                raise ValueError(
                    "Unsupported embedding provider: "
                    f"{EMBEDDING_PROVIDER}"
                )

            if not batch_embeddings:
                raise ValueError(
                    f"{EMBEDDING_PROVIDER} returned empty embeddings."
                )

            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    "Embedding count does not match input count."
                )

            embeddings.extend(batch_embeddings)

    return embeddings


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b, strict=True)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )
