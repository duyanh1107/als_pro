from __future__ import annotations

import re
from functools import lru_cache

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
    model: str = DEFAULT_CROSS_ENCODER_MODEL,
    debug: bool = False,
    ) -> dict[str, object]:
    normalized_query = " ".join(query.strip().split())
    if not normalized_query or not chunks:
        return {
            "query": normalized_query,
            "selected_indices": [],
            "chunks": [],
        }

    try:
        cross_encoder = _load_cross_encoder(model)
        pairs = [(normalized_query, _build_chunk_text(chunk)) for chunk in chunks]
        raw_scores = cross_encoder.predict(pairs)
        scored_indices = sorted(
            enumerate(raw_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        selected_indices = [index for index, _ in scored_indices[:top_k]]
        result = {
            "query": normalized_query,
            "selected_indices": selected_indices,
            "chunks": [chunks[index] for index in selected_indices],
        }
        if debug:
            _print_rerank_debug(result, chunks, source="cross_encoder")
        return result
    except Exception as exc:
        # Fall back to lexical overlap if the local cross-encoder stack or model
        # is unavailable. This keeps the pipeline running until dependencies are installed.
        selected_indices = _fallback_rerank_indices(normalized_query, chunks, top_k)
        result = {
            "query": normalized_query,
            "selected_indices": selected_indices,
            "chunks": [chunks[index] for index in selected_indices],
        }
        if debug:
            _print_rerank_debug(
                result,
                chunks,
                source="fallback",
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
        return result


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    # Import lazily so the codebase can still run in environments where the
    # cross-encoder dependencies are not installed yet.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _build_chunk_text(chunk: dict) -> str:
    # Provide both metadata and content so the cross-encoder can judge direct
    # answer relevance instead of only lexical overlap on the chunk body.
    parts = [
        f"Chapter: {chunk.get('chapter') or 'Unknown'}",
        f"Section: {chunk.get('section') or 'Unknown'}",
        f"Subsection: {chunk.get('subsection') or 'Unknown'}",
        f"Pages: {chunk.get('page_range') or chunk.get('start_page') or 'unknown'}",
        chunk.get("content") or "",
    ]
    return "\n".join(parts)


def _fallback_rerank_indices(query: str, chunks: list[dict], top_k: int) -> list[int]:
    query_terms = set(_tokenize(query))
    scored: list[tuple[int, int]] = []

    for index, chunk in enumerate(chunks):
        text = _build_chunk_text(chunk)
        overlap = len(query_terms.intersection(_tokenize(text)))
        scored.append((overlap, index))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [index for _, index in scored[:top_k]]


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _print_rerank_debug(
    result: dict[str, object],
    chunks: list[dict],
    source: str,
    fallback_reason: str | None = None,
) -> None:
    print("\n" + "=" * 80)
    print("RERANK DEBUG")
    print("=" * 80)
    print(f"Source: {source}")
    if fallback_reason:
        print(f"Fallback reason: {fallback_reason}")
    print(f"Query: {result['query']}")
    print(f"Selected indices: {result['selected_indices']}")

    for rank, index in enumerate(result["selected_indices"], start=1):
        chunk = chunks[index]
        section = chunk.get("section") or "Unknown section"
        subsection = chunk.get("subsection") or "Unknown subsection"
        page_range = chunk.get("page_range") or "unknown pages"
        print(f"{rank}. {section} | {subsection} | pages {page_range}")
