from __future__ import annotations

from llm.lesson_generator import generate_module_lesson
from llm.lesson_generator import get_mastery_band
from services.lesson_service import load_teacher_lesson_draft
from services.lesson_store import list_lesson_drafts_for_module


class AdaptiveService:
    def __init__(self, learner, engine, question_bank=None, rag_service=None):
        self.learner = learner
        self.engine = engine
        self.question_bank = question_bank
        self.rag_service = rag_service

    def get_lesson(self, module):
        mastery = self.get_module_mastery(module)
        mastery_band = get_mastery_band(mastery)

        # Prefer a teacher-reviewed lesson draft when one exists for this module and mastery band.
        # This makes lesson authoring teacher-owned first, while preserving a live
        # generation fallback for modules that have not been reviewed yet.
        stored_draft = load_teacher_lesson_draft(module.course_id, module.module_id, mastery_band)
        if stored_draft is not None and stored_draft.status in {"reviewed", "approved"}:
            return stored_draft.edited_lesson or stored_draft.generated_lesson

        if self.rag_service is None:
            raise RuntimeError("RAG service is required to generate a lesson.")

        # Fall back to live generation only when no teacher-approved draft exists yet.
        chunks = self.rag_service.retrieve_for_module(module, k=5)
        return generate_module_lesson(module, mastery, chunks)

    def get_module_mastery(self, module) -> float:
        skill_weights = self._get_module_skill_weights(module)
        if not skill_weights:
            return self.learner.get_mastery(module.primary_skill)

        weighted_sum = 0.0
        total_weight = 0.0
        for skill, weight in skill_weights.items():
            safe_weight = max(0.0, float(weight))
            if safe_weight <= 0:
                continue
            weighted_sum += self.learner.get_mastery(skill) * safe_weight
            total_weight += safe_weight

        if total_weight <= 0:
            return self.learner.get_mastery(module.primary_skill)
        return weighted_sum / total_weight

    def _get_module_skill_weights(self, module) -> dict[str, float]:
        drafts = list_lesson_drafts_for_module(module.course_id, module.module_id)
        for draft in drafts:
            if draft.skill_weights:
                return draft.skill_weights
        for draft in drafts:
            if draft.ai_recommended_skill_weights:
                return draft.ai_recommended_skill_weights

        if getattr(module, "skills", None):
            return {skill: 1.0 for skill in module.skills}
        return {module.primary_skill: 1.0}

    def get_next_question(self, module):
        if self.question_bank is None:
            raise RuntimeError("Question bank is required to select questions.")

        skill = module.primary_skill
        difficulty = self.engine.get_difficulty(self.learner, skill)
        return self.question_bank.sample(skill, difficulty)

    def submit_answer(self, question, correct, attempt):
        self.learner.update(correct, question.skills)
        primary_skill = question.skills[0]

        give_hint = self.engine.should_give_hint(
            self.learner,
            primary_skill,
            attempt,
        )

        return {"give_hint": give_hint}
