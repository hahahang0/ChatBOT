import io
from pathlib import Path

from fastapi import HTTPException
from pypdf import PdfReader

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_PDF_SIZE,
)


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunks.append(
            " ".join(words[start:end])
        )

        if end == len(words):
            break

        start = end - overlap

    return chunks


async def extract_pdf_chunks(
    file,
) -> tuple[str, list[dict], int]:

    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a PDF file.",
        )

    filename = Path(
        file.filename.replace("\\", "/")
    ).name

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_content = await file.read(MAX_PDF_SIZE + 1)

    if len(file_content) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB size limit.",
        )

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if b"%PDF-" not in file_content[:1024]:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not appear to be a valid PDF.",
        )

    try:
        reader = PdfReader(
            io.BytesIO(file_content)
        )

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported.",
            )

        page_count = len(reader.pages)

        if page_count == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain any pages.",
            )

        all_chunks = []
        chunk_index = 0

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            page_text = page.extract_text() or ""

            if not page_text.strip():
                continue

            page_chunks = split_text_into_chunks(page_text)

            for chunk_text in page_chunks:
                all_chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "filename": filename,
                    "content": chunk_text,
                })

                chunk_index += 1

        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No extractable text was found. "
                    "This PDF may contain scanned images."
                ),
            )

        print("\n========== IN-MEMORY PDF PROCESSING ==========")
        print("Filename:", filename)
        print("File size:", len(file_content))
        print("Pages:", page_count)
        print("Extracted chunks:", len(all_chunks))
        print("No PDF file was saved to disk.")
        print("==============================================\n")

        return filename, all_chunks, page_count

    except HTTPException:
        raise

    except Exception as error:
        print(
            "[PDF Processing] Error:",
            repr(error),
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not process the PDF. "
                "Check that it is a valid, text-based PDF."
            ),
        ) from error