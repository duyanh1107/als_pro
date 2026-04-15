from __future__ import annotations

import json
from pathlib import Path

from core.models import LessonDraft


BASE_DIR = Path(__file__).resolve().parent.parent
LESSONS_DIR = BASE_DIR / "data" / "lessons"


def _normalize_key(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def get_lesson_draft_path(course_id: str, module_id: str, mastery_band: str) -> Path:
    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    # Include mastery band in the filename so the same module can have separate
    # teacher-approved variants for low / medium / high mastery learners.
    filename = (
        f"{_normalize_key(course_id)}__{_normalize_key(module_id)}__{_normalize_key(mastery_band)}.json"
    )
    return LESSONS_DIR / filename


def save_lesson_draft(draft: LessonDraft) -> Path:
    path = get_lesson_draft_path(draft.course_id, draft.module_id, draft.mastery_band)
    path.write_text(json.dumps(draft.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def delete_lesson_draft(course_id: str, module_id: str, mastery_band: str) -> None:
    path = get_lesson_draft_path(course_id, module_id, mastery_band)
    path.unlink(missing_ok=True)


def load_lesson_draft(course_id: str, module_id: str, mastery_band: str) -> LessonDraft | None:
    path = get_lesson_draft_path(course_id, module_id, mastery_band)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return LessonDraft.from_dict(payload)


def list_lesson_drafts() -> list[LessonDraft]:
    LESSONS_DIR.mkdir(parents=True, exist_ok=True)
    drafts: list[LessonDraft] = []
    for path in sorted(LESSONS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            drafts.append(LessonDraft.from_dict(payload))
        except (json.JSONDecodeError, TypeError):
            continue
    return drafts


def list_lesson_drafts_for_module(course_id: str, module_id: str) -> list[LessonDraft]:
    return [
        draft
        for draft in list_lesson_drafts()
        if draft.course_id == course_id and draft.module_id == module_id
    ]
