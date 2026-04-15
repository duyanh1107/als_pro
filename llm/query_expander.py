from __future__ import annotations

import json

from .client import get_openai_client


def expand_query(
    query: str,
    module_title: str | None = None,
    chapter_title: str | None = None,
    model: str = "gpt-4o-mini",
    debug: bool = False,
) -> dict[str, list[str] | str]:
    # Normalize whitespace up front so both the LLM path and fallback path start
    # from the same cleaned query text.
    normalized_query = " ".join(query.strip().split())
    if not normalized_query:
        return {"original": query, "normalized": "", "expanded_query": ""}

    prompt = f"""
You are expanding a student question for retrieval in a tutoring RAG system.

Student question:
{normalized_query}

Current module title:
{module_title or "Unknown"}

Current chapter title:
{chapter_title or "Unknown"}

Task:
- Clean the query by fixing typos or wording issues.
- Identify the keywords in the query and enrich the same query with synonyms or alternate keyword phrasings, along with the original cleaned query.
- Return one expanded retrieval query, not multiple separate queries.
- Keep the expanded query semantically equivalent to the cleaned question.
- Do not broaden the topic.
- Do not answer the question.

Return JSON only in this shape:
{{
  "normalized": "cleaned question",
  "expanded_query": "one enriched retrieval query that keeps the same meaning"
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
        expanded_query = str(payload.get("expanded_query") or normalized).strip()
        result = {
            "original": query,
            "normalized": normalized,
            "expanded_query": expanded_query,
        }
        if debug:
            _print_query_expansion_debug(result, source="llm")
        return result
    except Exception:
        # Fall back to a conservative local expansion instead of blocking retrieval
        # when the LLM is unavailable.
        result = {
            "original": query,
            "normalized": normalized_query,
            "expanded_query": _fallback_expand_query(normalized_query, module_title),
        }
        if debug:
            _print_query_expansion_debug(result, source="fallback")
        return result


def _fallback_expand_query(query: str, module_title: str | None = None) -> str:
    # Keep the fallback narrow and maintainable: preserve one cleaned query and
    # optionally anchor it to the module in the same retrieval string.
    if module_title:
        return f"{query} | {module_title}"
    return query


def _print_query_expansion_debug(result: dict[str, list[str] | str], source: str) -> None:
    print("\n" + "=" * 80)
    print("QUERY EXPANSION DEBUG")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Original: {result['original']}")
    print(f"Normalized: {result['normalized']}")
    print(f"Expanded query: {result['expanded_query']}")
