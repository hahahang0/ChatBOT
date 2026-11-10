import httpx
from fastapi import HTTPException

from app.config import (
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    MAX_SELECTED_DOCUMENT_CONTEXT_CHARS,
    RAG_SELECTED_TOP_K,
)
from app.embeddings import (
    cosine_similarity,
    generate_embeddings,
)
from app.pdf_service import extract_pdf_chunks

METADATA_QUERY_TERMS = (
    "title",
    "paper title",
    "research title",
    "author",
    "authors",
    "written by",
    "who wrote",
    "abstract",
    "journal",
    "publication",
    "published",
    "publication year",
    "year of publication",
    "conference",
    "keywords",
    "affiliation",
)


def is_metadata_question(query: str) -> bool:

    normalized_query = query.lower()

    return any(
        term in normalized_query
        for term in METADATA_QUERY_TERMS
    )


async def retrieve_relevant_chunks(
    query: str,
    chunks: list[dict],
) -> list[dict]:

    if not chunks:
        return []

    query_embeddings = await generate_embeddings(
        [query],
        task_type="RETRIEVAL_QUERY",
    )

    if not query_embeddings:
        return []

    query_embedding = query_embeddings[0]

    results = []

    for chunk in chunks:

        chunk_embedding = chunk.get("embedding")

        if not chunk_embedding:
            continue

        similarity = cosine_similarity(
            query_embedding,
            chunk_embedding,
        )

        results.append({
            **chunk,
            "similarity": similarity,
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    print("\n========== RAG RETRIEVAL ==========")
    print("Chunks scored:", len(results))

    for result in results[:5]:
        print(
            "Page:",
            result["page_number"],
            "| Chunk:",
            result["chunk_index"],
            "| Similarity:",
            round(result["similarity"], 4),
        )

    selected_results = results[:RAG_SELECTED_TOP_K]

    if is_metadata_question(query):

        first_page_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("page_number") == 1
        ]

        if first_page_chunks:

            first_page_chunk = min(
                first_page_chunks,
                key=lambda item: item.get("chunk_index", 0),
            )

            first_page_result = next(
                (
                    result
                    for result in results
                    if result["chunk_index"]
                    == first_page_chunk["chunk_index"]
                ),
                None,
            )

            if first_page_result:

                selected_results = [
                    first_page_result
                ] + [
                    result
                    for result in selected_results
                    if result["chunk_index"]
                    != first_page_result["chunk_index"]
                ]

                selected_results = selected_results[
                    :RAG_SELECTED_TOP_K
                ]

                print(
                    "[RAG] Metadata question detected. "
                    "Prioritized first-page chunk."
                )

    print(
        "[RAG] Selected chunks:",
        len(selected_results),
    )

    print("==================================\n")

    return selected_results


async def build_rag_context(
    query: str,
    file,
) -> tuple[str, list[dict]]:

    if file is None:
        print(
            "[RAG] No PDF attached. "
            "Skipping document retrieval."
        )

        return "", []

    if EMBEDDING_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini embedding API key is not configured.",
        )

    filename, chunks, page_count = await extract_pdf_chunks(file)

    try:
        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        embeddings = await generate_embeddings(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count does not match extracted chunk count."
            )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk["embedding"] = embedding

        relevant_chunks = await retrieve_relevant_chunks(
            query,
            chunks,
        )

    except httpx.ConnectError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot connect to the configured embedding provider."
            ),
        ) from error

    except httpx.HTTPStatusError as error:
        from app.embeddings import get_embedding_error_detail

        error_detail = get_embedding_error_detail(
            error.response
        )

        raise HTTPException(
            status_code=error.response.status_code,
            detail=(
                "Embedding provider error: "
                + error_detail
            ),
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        print(
            "[RAG] Processing error:",
            repr(error),
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not process or retrieve the attached PDF. "
                "Check the configured embedding provider."
            ),
        ) from error

    if not relevant_chunks:
        return "", []

    context_parts = []
    sources = []
    total_characters = 0

    for index, chunk in enumerate(
        relevant_chunks,
        start=1,
    ):

        source_label = (
            f"[Source {index}: {filename}, "
            f"Page {chunk['page_number']}]"
        )

        section = (
            f"{source_label}\n"
            f"{chunk['content']}"
        )

        remaining = (
            MAX_SELECTED_DOCUMENT_CONTEXT_CHARS
            - total_characters
        )

        if remaining <= 0:
            break

        if len(section) > remaining:
            section = section[:remaining]

        context_parts.append(section)

        total_characters += len(section)

        sources.append({
            "filename": filename,
            "page_number": chunk["page_number"],
            "similarity": round(
                chunk["similarity"],
                4,
            ),
            "excerpt": chunk["content"][:500],
        })

    print("[RAG] PDF pages:", page_count)
    print("[RAG] Context chunks:", len(sources))
    print("[RAG] Context characters:", total_characters)

    return "\n\n".join(context_parts), sources
