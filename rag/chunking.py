from __future__ import annotations

from collections import defaultdict
import re
from typing import Any


def chunk_by_subsection_fallback(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)

    for doc in documents:
        chapter = doc.get("chapter")
        section = doc.get("section")
        subsection = doc.get("subsection")
        # If a page has no subsection label, group it at section level instead of dropping structure.
        key = (chapter, section, subsection if subsection else section)
        groups[key].append(doc)

    chunks: list[dict[str, Any]] = []

    for (chapter, section, sub_or_sec), docs in groups.items():
        docs_sorted = sorted(docs, key=lambda x: x["page"])
        # Merge all pages belonging to the same semantic unit into one retrievable chunk.
        content = "\n".join(d["content"] for d in docs_sorted if d["content"])
        pages = [d["page"] for d in docs_sorted]

        chunks.append(
            {
                "chapter": chapter,
                "section": section,
                "subsection": sub_or_sec,
                "content": content,
                "page_range": f"{min(pages)}-{max(pages)}",
                "num_pages": len(pages),
                "start_page": min(pages),
            }
        )

    chunks.sort(key=lambda x: x["start_page"])
    return chunks


def split_large_chunks(
    chunks: list[dict[str, Any]],
    max_chars: int = 8000,
) -> list[dict[str, Any]]:
    new_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        text = chunk["content"]
        if len(text) <= max_chars:
            new_chunks.append(chunk)
            continue

        # Oversized subsection chunks are recursively split near the midpoint,
        # preferring paragraph boundaries so each piece stays readable and the
        # split is less likely to cut through a definition or worked example.
        pieces = _split_text_by_paragraph_boundary(text, max_chars=max_chars)
        chunk_group_id = _build_chunk_group_id(chunk)
        total_parts = len(pieces)
        for part_index, piece in enumerate(pieces):
            new_chunks.append(
                {
                    **chunk,
                    "content": piece,
                    "chunk_group_id": chunk_group_id,
                    "chunk_part_index": part_index,
                    "chunk_part_count": total_parts,
                }
            )

    return new_chunks


def analyze_chunk_sizes(chunks: list[dict[str, Any]]) -> None:
    # This is only a debugging helper to understand how aggressive chunking needs to be.
    lengths = [len(c["content"]) for c in chunks]
    if not lengths:
        print("No chunks available for analysis.")
        return

    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    token_estimates = [length / 4 for length in lengths]
    max_tokens = max(token_estimates)
    avg_tokens = sum(token_estimates) / len(token_estimates)

    print("\n" + "=" * 80)
    print("CHUNK SIZE ANALYSIS")
    print("=" * 80)
    print(f"Total chunks: {len(chunks)}")
    print(f"Max length (chars): {max_len}")
    print(f"Min length (chars): {min_len}")
    print(f"Avg length (chars): {avg_len:.2f}")
    print("\nApprox token stats:")
    print(f"Max tokens ~ {int(max_tokens)}")
    print(f"Avg tokens ~ {int(avg_tokens)}")

    for i, tokens in enumerate(token_estimates):
        if tokens > 8000:
            print(f"Warning: chunk {i} exceeds token limit: ~{int(tokens)} tokens")


def _split_text_by_paragraph_boundary(text: str, max_chars: int) -> list[str]:
    cleaned_text = text.strip()
    if len(cleaned_text) <= max_chars:
        return [cleaned_text]

    # Recurse until every piece is under the max size while keeping sibling pieces
    # tied together through chunk_group_id / chunk_part_index later in the pipeline.
    split_index = _find_split_boundary(cleaned_text)
    if split_index is None or split_index <= 0 or split_index >= len(cleaned_text):
        # Fallback to a hard midpoint only when no paragraph/line boundary is available.
        split_index = len(cleaned_text) // 2

    left = cleaned_text[:split_index].strip()
    right = cleaned_text[split_index:].strip()

    if not left or not right:
        midpoint = len(cleaned_text) // 2
        left = cleaned_text[:midpoint].strip()
        right = cleaned_text[midpoint:].strip()

    pieces: list[str] = []
    if left:
        pieces.extend(_split_text_by_paragraph_boundary(left, max_chars))
    if right:
        pieces.extend(_split_text_by_paragraph_boundary(right, max_chars))
    return pieces


def _find_split_boundary(text: str) -> int | None:
    midpoint = len(text) // 2

    # Prefer blank-line paragraph boundaries. If PDF cleanup has already collapsed
    # blank lines, fall back to the nearest single line break.
    paragraph_breaks = [match.end() for match in re.finditer(r"\n\s*\n", text)]
    if not paragraph_breaks:
        paragraph_breaks = [match.end() for match in re.finditer(r"\n", text)]
    if not paragraph_breaks:
        return None

    return min(paragraph_breaks, key=lambda index: abs(index - midpoint))


def _build_chunk_group_id(chunk: dict[str, Any]) -> str:
    # This stable identifier lets retrieval know that split parts still belong to
    # the same original subsection-level chunk.
    chapter = chunk.get("chapter") or "unknown_chapter"
    section = chunk.get("section") or "unknown_section"
    subsection = chunk.get("subsection") or "unknown_subsection"
    start_page = chunk.get("start_page") or 0
    return f"{chapter}|{section}|{subsection}|{start_page}"
