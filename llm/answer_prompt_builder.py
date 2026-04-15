from __future__ import annotations

from typing import Any


def build_answer_generation_prompt(
    query: str,
    grounded_context: str,
    grounded_sources: list[dict[str, Any]],
) -> str:
    source_lines: list[str] = []
    for source in grounded_sources:
        source_lines.append(
            f"- Source {source['source_id']}: "
            f"{source.get('chapter') or 'Unknown chapter'} -> "
            f"{source.get('subsection') or source.get('section') or 'Unknown section'} "
            f"| pages {source.get('page_range') or source.get('start_page') or 'unknown'}"
        )

    sources_block = "\n".join(source_lines) if source_lines else "No grounded sources provided."

    # Keep the prompt explicit about grounding so the model cites the full source
    # label shown to the user instead of a shorter internal-only source id.
    return f"""
You are answering a student question in a tutoring RAG system.

Student question:
{query}

Grounded sources:
{sources_block}

Grounded context:
{grounded_context}

Instructions:
- Answer using only the grounded context above.
- Add source with exact page and section.
- If the context is insufficient, say what is missing instead of inventing facts.
- Prefer a direct, concise explanation first.
- Always include examples, definitions, or conditions from the context if they materially support the answer.
- Cite the supporting sources using the exact full source label from the Grounded sources list.
- Example citation format: Source 2: Linear Equations and Matrices -> Matrix Addition and Vectors | pages 17-17
- Keep citations out of the Answer and Explanation fields.
- Put all citations only in the final Sources field.
- Do not mention information that does not appear in the grounded context.

Return your answer in this format:
Question: <student question>
Answer: <direct answer>
Explanation: <short grounded explanation without inline citations>
Sources: <comma-separated full source labels copied from the Grounded sources list>
"""
