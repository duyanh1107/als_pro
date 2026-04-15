from __future__ import annotations

from typing import Any

from .client import get_openai_client


DIRECT_CONTEXT_CHAR_LIMIT = 12000


def get_mastery_band(mastery: float) -> str:
    # Translate a continuous mastery score into a coarse pacing mode for lesson generation.
    if mastery < 0.4:
        return "low"
    if mastery < 0.75:
        return "medium"
    return "high"


def get_pacing_instruction(mastery_band: str) -> str:
    # High-mastery learners get compressed explanations, while low-mastery learners get more scaffolding.
    return {
        "low": "Teach slowly, define terms clearly, build intuition, and include a simple worked example.",
        "medium": "Teach at a balanced pace, reinforce the main idea, and include one concise example.",
        "high": "Teach quickly, compress repetition, focus on the essential ideas, and highlight what is new or subtle.",
    }[mastery_band]


def generate_module_lesson(
    module,
    mastery: float,
    chunks: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> str:
    mastery_band = get_mastery_band(mastery)
    pacing_instruction = get_pacing_instruction(mastery_band)

    # Use direct generation when the combined subsection context is still reasonably small.
    total_context_chars = sum(len(chunk["content"]) for chunk in chunks)
    if total_context_chars <= DIRECT_CONTEXT_CHAR_LIMIT:
        context = "\n\n".join(chunk["content"] for chunk in chunks)
        return _generate_lesson_from_context(
            module,
            mastery,
            mastery_band,
            pacing_instruction,
            context,
            model,
        )

    # For large subsections, use a map-reduce style flow:
    # summarize each split chunk separately, then synthesize one final lesson.
    chunk_summaries = [
        _summarize_chunk_for_module(module, mastery_band, pacing_instruction, chunk, index, model)
        for index, chunk in enumerate(chunks, start=1)
    ]
    summarized_context = "\n\n".join(chunk_summaries)
    return _generate_lesson_from_summaries(
        module,
        mastery,
        mastery_band,
        pacing_instruction,
        summarized_context,
        model,
    )


def _generate_lesson_from_context(
    module,
    mastery: float,
    mastery_band: str,
    pacing_instruction: str,
    context: str,
    model: str,
) -> str:
    prompt = f"""
You are generating a lesson for an adaptive learning system.

Module title: {module.title}
Chapter: {module.chapter_title or "Unknown"}
Primary skill: {module.primary_skill}
Learner mastery score: {mastery:.2f}
Learner mastery band: {mastery_band}

Instruction:
{pacing_instruction}

Use the provided source context to produce a short learner-facing lesson.
The lesson should:
- stay faithful to the source context
- explain the module clearly
- adapt depth and speed to the learner mastery level
- end with a short "What to focus on" list with 2-3 bullets

Source context:
{context}
"""

    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _summarize_chunk_for_module(
    module,
    mastery_band: str,
    pacing_instruction: str,
    chunk: dict[str, Any],
    chunk_index: int,
    model: str,
) -> str:
    # Each chunk summary preserves the important teaching content without forcing
    # the whole subsection into one oversized prompt.
    prompt = f"""
You are summarizing one chunk of a larger lesson source.

Module title: {module.title}
Chapter: {module.chapter_title or "Unknown"}
Primary skill: {module.primary_skill}
Learner mastery band: {mastery_band}
Instruction: {pacing_instruction}
Chunk number: {chunk_index}

Produce a concise summary that preserves:
- main concepts
- important definitions
- methods or procedures
- examples or results that matter for teaching

Chunk content:
{chunk["content"]}
"""

    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = response.choices[0].message.content or ""
    return f"Chunk {chunk_index} summary:\n{summary}"


def _generate_lesson_from_summaries(
    module,
    mastery: float,
    mastery_band: str,
    pacing_instruction: str,
    summarized_context: str,
    model: str,
) -> str:
    # The final lesson is generated from chunk summaries, not raw chunks, to stay within limits.
    prompt = f"""
You are generating a lesson for an adaptive learning system from chunk summaries.

Module title: {module.title}
Chapter: {module.chapter_title or "Unknown"}
Primary skill: {module.primary_skill}
Learner mastery score: {mastery:.2f}
Learner mastery band: {mastery_band}

Instruction:
{pacing_instruction}

Use the summaries below to produce a short learner-facing lesson.
The lesson should:
- merge the important ideas across all summaries
- preserve progression across the module
- adapt depth and speed to the learner mastery level
- end with a short "What to focus on" list with 2-3 bullets

Chunk summaries:
{summarized_context}
"""

    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""
