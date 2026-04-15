from __future__ import annotations

import json

from .client import get_openai_client


def step_back_query(
    query: str,
    module_title: str | None = None,
    chapter_title: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, str]:
    normalized_query = " ".join(query.strip().split())
    if not normalized_query:
        return {"original": query, "normalized": "", "step_back_query": ""}

    prompt = f"""
You are generating a step-back retrieval query for a tutoring RAG system.

Student question:
{normalized_query}

Current module title:
{module_title or "Unknown"}

Current chapter title:
{chapter_title or "Unknown"}

Task:
- First clean the question by fixing only obvious typos or wording issues.
- Rewrite the question into a higher-level conceptual query that gives broader context.
- Keep it within the same topic.
- Do not answer the question.
- Do not make it too broad.

Return JSON only in this shape:
{{
  "normalized": "cleaned question",
  "step_back_query": "higher-level conceptual query"
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
        step_back = str(payload.get("step_back_query") or normalized).strip()
        return {
            "original": query,
            "normalized": normalized,
            "step_back_query": step_back,
        }
    except Exception:
        # Keep the fallback modest by anchoring the query to the current module
        # instead of inventing a broad abstraction locally.
        if module_title:
            return {
                "original": query,
                "normalized": normalized_query,
                "step_back_query": f"core idea of {module_title}",
            }
        return {
            "original": query,
            "normalized": normalized_query,
            "step_back_query": normalized_query,
        }
