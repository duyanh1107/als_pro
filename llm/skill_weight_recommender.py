from __future__ import annotations

import json

from .client import get_openai_client


def recommend_skill_weights(course, module, chunks: list[dict], model: str = "gpt-4o-mini") -> dict[str, float]:
    allowed_skills = list(dict.fromkeys(course.skills))
    if not allowed_skills:
        return {}

    context = "\n\n".join((chunk.get("content") or "")[:1200] for chunk in chunks[:4])

    prompt = f"""
You are assigning skill weights for one course module.

Course title: {course.title}
Course description: {course.description}
Module title: {module.title}
Chapter title: {module.chapter_title or "Unknown"}
Primary skill: {module.primary_skill}

Allowed skills:
{json.dumps(allowed_skills, ensure_ascii=False)}

Source content excerpt:
{context}

Task:
- Assign a weight to every allowed skill.
- Weights must be decimals between 0 and 1.
- Give higher weight to skills that are central to this module.
- Keep some weight on relevant global skills when justified by the content.
- The weights are independent relevance scores; they do not need to sum to 1.

Return JSON only in this shape:
{{
  "weights": {{
    "skill_name": 0.42
  }}
}}
"""

    try:
        response = get_openai_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        raw_weights = payload.get("weights", {})
        if isinstance(raw_weights, dict):
            parsed = {
                skill: _clamp_weight(raw_weights.get(skill, 0.0))
                for skill in allowed_skills
            }
            if any(value > 0 for value in parsed.values()):
                return parsed
    except Exception:
        pass

    return heuristic_skill_weights(allowed_skills, module.primary_skill)


def heuristic_skill_weights(allowed_skills: list[str], primary_skill: str) -> dict[str, float]:
    if not allowed_skills:
        return {}

    weights: dict[str, float] = {}
    other_skills = [skill for skill in allowed_skills if skill != primary_skill]

    if not other_skills:
        return {primary_skill: 1.0}

    primary_prefix = primary_skill.split(":", 1)[0] if ":" in primary_skill else primary_skill
    same_domain = [
        skill for skill in other_skills if skill.split(":", 1)[0] == primary_prefix
    ]
    global_skills = [skill for skill in other_skills if skill.startswith("global:")]
    remaining = [skill for skill in other_skills if skill not in same_domain and skill not in global_skills]

    weights[primary_skill] = 0.9

    if same_domain:
        for skill in same_domain:
            weights[skill] = 0.65

    if global_skills:
        for skill in global_skills:
            weights[skill] = 0.35

    if remaining:
        for skill in remaining:
            weights[skill] = 0.25

    for skill in allowed_skills:
        weights.setdefault(skill, 0.0)

    return {skill: _clamp_weight(weights.get(skill, 0.0)) for skill in allowed_skills}


def _clamp_weight(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
