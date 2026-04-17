from __future__ import annotations

import json
from pathlib import Path

from core.models import CourseConceptReview


BASE_DIR = Path(__file__).resolve().parent.parent
CONCEPTS_DIR = BASE_DIR / "data" / "concepts"


def _normalize_key(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def get_course_concept_review_path(course_id: str) -> Path:
    """Return the teacher review file path for one course concept catalog."""
    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
    return CONCEPTS_DIR / f"{_normalize_key(course_id)}__concept_review.json"


def save_course_concept_review(review: CourseConceptReview) -> Path:
    """Persist a teacher-reviewed concept configuration for one course."""
    path = get_course_concept_review_path(review.course_id)
    path.write_text(json.dumps(review.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_course_concept_review(course_id: str) -> CourseConceptReview | None:
    """Load a saved teacher review if one exists."""
    path = get_course_concept_review_path(course_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return CourseConceptReview.from_dict(payload)
