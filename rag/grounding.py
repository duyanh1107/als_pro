from __future__ import annotations

from typing import Any


def build_grounded_context(
    query: str,
    chunks: list[dict[str, Any]],
    max_chars: int | None = None,
) -> dict[str, Any]:
    normalized_query = " ".join(query.strip().split())
    if not normalized_query or not chunks:
        return {
            "query": normalized_query,
            "sources": [],
            "context_text": "",
        }

    sources: list[dict[str, Any]] = []
    context_parts: list[str] = []
    total_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        source = {
            "source_id": index,
            "chapter": chunk.get("chapter"),
            "section": chunk.get("section"),
            "subsection": chunk.get("subsection"),
            "page_range": chunk.get("page_range"),
            "start_page": chunk.get("start_page"),
        }
        sources.append(source)

        # Keep the final prompt context explicitly grounded in retrieved text and
        # source metadata so downstream answer generation can cite and trace it.
        chunk_block = (
            f"[Source {index}]\n"
            f"Chapter: {source['chapter'] or 'Unknown chapter'}\n"
            f"Section: {source['section'] or 'Unknown section'}\n"
            f"Subsection: {source['subsection'] or 'Unknown subsection'}\n"
            f"Pages: {source['page_range'] or source['start_page'] or 'unknown'}\n"
            f"Content:\n{(chunk.get('content') or '').strip()}"
        )

        if max_chars is not None:
            remaining = max_chars - total_chars
            if remaining <= 0:
                break

            if len(chunk_block) > remaining:
                chunk_block = chunk_block[:remaining].rsplit(" ", 1)[0]
        context_parts.append(chunk_block)
        total_chars += len(chunk_block) + 2

        if max_chars is not None and total_chars >= max_chars:
            break

    return {
        "query": normalized_query,
        "sources": sources[: len(context_parts)],
        "context_text": "\n\n".join(context_parts),
    }
