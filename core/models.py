from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict


@dataclass
class Question:
    content: str
    options: list[str]
    correct: str
    skills: list[str]


@dataclass
class Course:
    # `source_name` links a course to external assets such as TOC/PDF files.
    course_id: str
    title: str
    description: str
    primary_skill: str
    skills: list[str] = field(default_factory=list)
    source_name: str | None = None


@dataclass
class Module:
    # A module is a learner-facing lesson unit derived from the course TOC.
    module_id: str
    course_id: str
    title: str
    primary_skill: str
    skills: list[str]
    chapter_title: str | None = None
    toc_number: str | None = None
    start_page: int | None = None
    level: str = "section"
    # Stores why the heuristic or LLM accepted this TOC entry as a module.
    selection_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Module":
        return cls(**data)


@dataclass
class LessonDraft:
    # A lesson draft is teacher-owned content generated from module-aligned RAG sources.
    draft_id: str
    course_id: str
    module_id: str
    module_title: str
    chapter_title: str | None
    primary_skill: str
    mastery_score: float
    mastery_band: str
    source_chunks: list[dict]
    generated_lesson: str
    ai_recommended_skill_weights: dict[str, float] = field(default_factory=dict)
    skill_weights: dict[str, float] = field(default_factory=dict)
    edited_lesson: str | None = None
    status: str = "draft"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LessonDraft":
        return cls(**data)
