from __future__ import annotations

import json

from .client import get_openai_client

ALLOWED_TRANSFORMS = {"none", "expand", "decompose", "step_back", "hyde"}

#Should extend examples
def select_query_transform(
    query: str,
    module_title: str | None = None,
    chapter_title: str | None = None,
    model: str = "gpt-4o-mini",
    debug: bool = False,
) -> dict[str, str | list[str]]:
    normalized_query = " ".join(query.strip().split())

    if not normalized_query:
        result = {
            "normalized": "",
            "transform": "none",
            "reason": "Empty query.",
            "signals": [],
        }
        if debug:
            _print_query_transform_debug(query, result, source="fallback")
        return result

    prompt = f"""
You are selecting the best query transformation for retrieval in a tutoring RAG system.

Student question:
{normalized_query}

Current module title:
{module_title or "Unknown"}

Current chapter title:
{chapter_title or "Unknown"}

Available transformations:
- none: use the cleaned question directly only when it is already clear, specific, and retrieval-ready
- expand: enrich the query with synonyms or alternate phrasings
- decompose: split a question that contains multiple distinct information needs
- step_back: rewrite a narrow/local question into a broader conceptual query
- hyde: generate a short hypothetical answer passage for retrieval

Decision rules:
- First clean obvious typos or wording issues and return the cleaned version in `normalized`.
- Choose `none` only if the cleaned question is already a clear natural-language retrieval query.
- Do not choose `none` for short keyword fragments, vague phrases, telegraphic wording, or typo-heavy input.
- Prefer `expand` for short fragment queries, weak wording, keyword-only input, or same-meaning synonym reformulation.
- Prefer `decompose` only when the student is clearly asking more than one thing.
- Prefer `step_back` when the question is about a local step but the answer likely needs broader conceptual context.
- Prefer `hyde` when the question is conceptual/open-ended and would benefit from an answer-like passage for retrieval.
- Choose exactly one transform from: none, expand, decompose, step_back, hyde.
- Return a short reason and a few short signals.

Examples:
- "add matrix" -> expand
- "wat is matrx addtion" -> expand
- "what is row echelon form and how do I use it?" -> decompose
- "why do we swap rows here?" -> step_back
- "explain matrix addition" -> hyde
- "what is matrix addition?" -> none or hyde, depending on whether direct retrieval is already sufficient

Return JSON only in this shape:
{{
  "normalized": "cleaned question",
  "transform": "expand",
  "reason": "short reason",
  "signals": ["signal 1", "signal 2"]
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
        transform = str(payload.get("transform") or "none").strip().lower()
        if transform not in ALLOWED_TRANSFORMS:
            transform = "none"

        raw_signals = payload.get("signals", [])
        if not isinstance(raw_signals, list):
            raw_signals = []

        result = {
            "normalized": normalized,
            "transform": transform,
            "reason": str(payload.get("reason") or "LLM selector did not provide a reason.").strip(),
            "signals": [str(signal).strip() for signal in raw_signals if str(signal).strip()][:4],
        }
        if debug:
            _print_query_transform_debug(query, result, source="llm")
        return result
    except Exception:
        # Fall back to direct retrieval with the cleaned query rather than using
        # brittle local heuristics when the routing model is unavailable.
        result = {
            "normalized": normalized_query,
            "transform": "none",
            "reason": "Selector fallback: use the cleaned query directly when routing is unavailable.",
            "signals": ["selector_fallback"],
        }
        if debug:
            _print_query_transform_debug(query, result, source="fallback")
        return result


def _print_query_transform_debug(
    query: str,
    result: dict[str, str | list[str]],
    source: str,
) -> None:
    print("\n" + "=" * 80)
    print("QUERY TRANSFORM SELECTION DEBUG")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Original: {query}")
    print(f"Normalized: {result['normalized']}")
    print(f"Transform: {result['transform']}")
    print(f"Reason: {result['reason']}")
    print(f"Signals: {', '.join(result['signals']) if result['signals'] else 'none'}")
