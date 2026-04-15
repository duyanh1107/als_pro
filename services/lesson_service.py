from __future__ import annotations

from core.models import LessonDraft
from llm.lesson_generator import generate_module_lesson
from llm.lesson_generator import get_mastery_band
from llm.skill_weight_recommender import recommend_skill_weights
from services.lesson_store import load_lesson_draft
from services.lesson_store import save_lesson_draft


def create_teacher_lesson_draft(course, module, rag_service, mastery_score: float) -> LessonDraft:
    # Teacher drafts are grounded in the same module retrieval used for learners,
    # but the output is saved for review/editing before release.
    chunks = rag_service.retrieve_for_module(module, k=5)
    lesson = generate_module_lesson(module, mastery_score, chunks)
    mastery_band = get_mastery_band(mastery_score)
    recommended_weights = recommend_skill_weights(course, module, chunks)
    return LessonDraft(
        # The draft id is descriptive only; persistence keys are handled by lesson_store.
        draft_id=f"{course.course_id}:{module.module_id}:{mastery_band}",
        course_id=course.course_id,
        module_id=module.module_id,
        module_title=module.title,
        chapter_title=module.chapter_title,
        primary_skill=module.primary_skill,
        mastery_score=mastery_score,
        mastery_band=mastery_band,
        source_chunks=[
            {
                "chapter": chunk.get("chapter"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "page_range": chunk.get("page_range"),
                "start_page": chunk.get("start_page"),
                # Save both preview and full content so the web reviewer can choose
                # between a quick skim and a full source inspection without re-running retrieval.
                "preview": (chunk.get("content") or "")[:1200],
                "content": chunk.get("content") or "",
            }
            for chunk in chunks
        ],
        generated_lesson=lesson,
        ai_recommended_skill_weights=recommended_weights,
        # Teacher edits start from the AI recommendation and can diverge later in the UI.
        skill_weights=dict(recommended_weights),
        edited_lesson=lesson,
    )


def save_teacher_lesson_draft(draft: LessonDraft) -> LessonDraft:
    save_lesson_draft(draft)
    return draft


def load_teacher_lesson_draft(course_id: str, module_id: str, mastery_band: str) -> LessonDraft | None:
    return load_lesson_draft(course_id, module_id, mastery_band)
