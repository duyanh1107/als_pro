from __future__ import annotations

import json

from .client import get_openai_client


def decompose_query(
    query: str,
    module_title: str | None = None,
    chapter_title: str | None = None,
    model: str = "gpt-4o-mini",
    debug: bool = False,
) -> dict[str, list[str] | str | bool]:
    normalized_query = " ".join(query.strip().split())
    if not normalized_query:
        return {
            "original": query,
            "normalized": "",
            "needs_decomposition": False,
            "subqueries": [],
        }

    prompt = f"""
You are decomposing a student question for retrieval in a tutoring RAG system.

Student question:
{normalized_query}

Current module title:
{module_title or "Unknown"}

Current chapter title:
{chapter_title or "Unknown"}

Task:
- First clean the question by fixing only obvious typos or wording issues.
- Decide whether the question contains multiple distinct sub-problems.
- If yes, break it into a few focused subqueries for retrieval.
- If no, keep one subquery equal to the normalized question.
- Do not answer the question.
- Do not broaden the topic.

Return JSON only in this shape:
{{
  "normalized": "cleaned question",
  "needs_decomposition": true,
  "subqueries": [
    "subquery 1",
    "subquery 2"
  ]
}}
"""

    try:
        response = get_openai_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        normalized = str(payload.get("normalized") or normalized_query).strip()
        raw_subqueries = payload.get("subqueries", [])
        if not isinstance(raw_subqueries, list):
            raw_subqueries = []
        subqueries = _dedupe_queries([str(item).strip() for item in raw_subqueries]) or [normalized]
        result = {
            "original": query,
            "normalized": normalized,
            "needs_decomposition": bool(payload.get("needs_decomposition")) and len(subqueries) > 1,
            "subqueries": subqueries[:4],
        }
        if debug:
            _print_query_decomposition_debug(result, source="llm")
        return result
    except Exception:
        result = {
            "original": query,
            "normalized": normalized_query,
            "needs_decomposition": False,
            "subqueries": [normalized_query],
        }
        if debug:
            _print_query_decomposition_debug(result, source="fallback")
        return result


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for query in queries:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def _print_query_decomposition_debug(
    result: dict[str, list[str] | str | bool],
    source: str,
) -> None:
    print("\n" + "=" * 80)
    print("QUERY DECOMPOSITION DEBUG")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Original: {result['original']}")
    print(f"Normalized: {result['normalized']}")
    print(f"Needs decomposition: {result['needs_decomposition']}")
    print("Subqueries:")
    for index, query in enumerate(result["subqueries"], start=1):
        print(f"{index}. {query}")
