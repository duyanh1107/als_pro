from __future__ import annotations

import json

from .client import get_openai_client


def generate_hypothetical_document(
    query: str,
    module_title: str | None = None,
    chapter_title: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, str]:
    normalized_query = " ".join(query.strip().split())
    if not normalized_query:
        return {"original": query, "normalized": "", "hypothetical_document": ""}

    prompt = f"""
You are generating a hypothetical answer passage for retrieval (HyDE) in a tutoring RAG system.

Student question:
{normalized_query}

Current module title:
{module_title or "Unknown"}

Current chapter title:
{chapter_title or "Unknown"}

Task:
- First clean the question by fixing only obvious typos or wording issues.
- Write a short hypothetical passage that would likely answer the student's question.
- Keep it factual in tone and aligned with textbook language.
- Do not mention that it is hypothetical.
- Keep it short and focused.

Return JSON only in this shape:
{{
  "normalized": "cleaned question",
  "hypothetical_document": "short hypothetical textbook-like answer passage"
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
        hypothetical_document = str(payload.get("hypothetical_document") or normalized).strip()
        return {
            "original": query,
            "normalized": normalized,
            "hypothetical_document": hypothetical_document,
        }
    except Exception:
        # Fall back to the normalized query when HyDE generation is unavailable.
        return {
            "original": query,
            "normalized": normalized_query,
            "hypothetical_document": normalized_query,
        }
