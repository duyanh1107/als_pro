from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .toc_provider import build_page_metadata, load_toc_text, parse_toc


def clean_text(text: str) -> str:
    # Normalize whitespace and common PDF ligatures so later parsing is more stable.
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"\n\d+\n", "\n", text)
    return text.strip()


def extract_and_clean_pdf(file_path: str) -> tuple[list[dict[str, Any]], int]:
    reader = PdfReader(file_path)
    cleaned_pages: list[dict[str, Any]] = []

    for i, page in enumerate(reader.pages):
        # Keep page numbers so TOC metadata can be attached later.
        text = page.extract_text()
        if not text:
            continue
        cleaned_pages.append({"page": i, "content": clean_text(text)})

    return cleaned_pages, len(reader.pages)


def build_documents(
    cleaned_pages: list[dict[str, Any]],
    page_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for page_data in cleaned_pages:
        page_number = page_data["page"]
        if page_number >= len(page_meta):
            continue

        # Merge page text with chapter/section/subsection labels for semantic chunking.
        meta = page_meta[page_number]
        documents.append(
            {
                "page": page_number,
                "chapter": meta["chapter"],
                "section": meta["section"],
                "subsection": meta["subsection"],
                "content": page_data["content"],
            }
        )

    return documents


def process_pdf_to_documents(
    file_path: str,
    toc_text: str | None = None,
    toc_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    End-to-end PDF preprocessing for the RAG pipeline.

    Returns:
    - documents: page-level documents enriched with chapter/section/subsection metadata
    - total_pages: total page count of the PDF
    - toc_entries: parsed TOC entries
    - page_meta: metadata assigned to every page
    """
    cleaned_pages, total_pages = extract_and_clean_pdf(file_path)
    toc_content = toc_text if toc_text is not None else load_toc_text(file_path, toc_path)
    toc_entries = parse_toc(toc_content)
    page_meta = build_page_metadata(toc_entries, total_pages)
    documents = build_documents(cleaned_pages, page_meta)
    return documents, total_pages, toc_entries, page_meta
