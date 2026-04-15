from __future__ import annotations

import json

from .client import get_openai_client


HEURISTIC_EXCLUDE_TERMS = {
    "appendix",
    "summary",
    "preface",
    "foreword",
    "bibliography",
    "references",
    "index",
    "solutions",
}


def heuristic_module_decision(title: str, chapter_title: str | None = None) -> tuple[bool, str]:
    # Cheap fallback for obvious non-lesson sections such as appendices or summaries.
    combined = f"{chapter_title or ''} {title}".lower()
    for term in HEURISTIC_EXCLUDE_TERMS:
        if term in combined:
            return False, f"Excluded by heuristic because it looks like `{term}` content."
    return True, "Included by heuristic because it looks like a teachable module."


def llm_module_decision(
    *,
    course_title: str,
    course_description: str,
    chapter_title: str | None,
    section_title: str,
    level: str,
) -> tuple[bool, str]:
    # The LLM is only deciding "keep as module or not", not generating lesson content here.
    prompt = f"""
You are an expert curriculum designer.

Your task is to analyze a table-of-contents entry and decide whether it should become a teaching module or be treated like appendix/reference material.

Definitions:
- Teaching module: content that teaches core concepts, builds mental models, or enables learners to apply knowledge.
- Appendix: reference material, supporting details, or content that is not essential for core learning.

Course title: {course_title}
Course description: {course_description}
Parent chapter: {chapter_title or "None"}
Candidate title: {section_title}
TOC level: {level}

Decision criteria:
1. Learning Value
- Does this teach a core concept?
- Will learners reuse this knowledge later?
- Does it change how they think or solve problems?

2. Actionability
- Can learners apply this immediately?
- Does it include steps, workflows, or patterns?

3. Frequency of Use
- Will learners use this frequently?
- Is it part of the main workflow?

4. Cognitive Load
- Is it difficult to understand without guidance?
- Does it require explanation or examples?

5. Dependency
- Do later topics depend on this?
- Will skipping this break understanding?

6. Content Type
- Core concepts, frameworks, tutorials -> Module
- Pure reference, administrative, navigational, or non-essential supporting content -> Appendix

7. Goal Alignment
- Does this directly support the course objective?

Heuristic:
Ask: "If this section is removed, will learners still succeed?"
- If NO -> Module
- If YES -> Appendix

Important guidance:
- Be conservative about excluding material.
- Normal concept sections, methods, definitions, examples, and theorem/result sections should usually be modules.
- Summary, appendix, bibliography, acknowledgements, index, and clearly supporting/reference sections should usually be appendix.
- Return concise reasoning tied to the criteria above.

Return strict JSON only:
{{
  "section_name": "{section_title}",
  "classification": "Module",
  "include": true,
  "reason": "1-2 concise sentences"
}}
"""
    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    classification = str(data.get("classification", "")).strip().lower()
    include = bool(data.get("include"))
    if classification in {"module", "appendix"}:
        include = classification == "module"
    return include, str(data.get("reason", "")).strip()


def decide_module_candidate(
    *,
    course_title: str,
    course_description: str,
    chapter_title: str | None,
    section_title: str,
    level: str,
    use_llm: bool,
) -> tuple[bool, str]:
    if not use_llm:
        return heuristic_module_decision(section_title, chapter_title)

    try:
        return llm_module_decision(
            course_title=course_title,
            course_description=course_description,
            chapter_title=chapter_title,
            section_title=section_title,
            level=level,
        )
    except Exception as exc:
        # Keep the system usable when API/network/auth issues happen.
        include, reason = heuristic_module_decision(section_title, chapter_title)
        return include, f"{reason} Fallback used because LLM filtering failed: {exc}"
