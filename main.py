from __future__ import annotations

from core.engine import DecisionEngine
from core.learner import Learner
from core.runner import choose_course, choose_module, run_teacher_console, show_learner_summary
from rag.pipeline import build_rag_pipeline
from services.adaptive_service import AdaptiveService
from services.content_service import get_document_path_for_course
from services.learner_store import create_new_learner, load_profile, save_learner


def main() -> None:
    print("Adaptive Learning AI")
    print("====================")

    mode = input("Select mode (1=learner, 2=teacher): ").strip()
    if mode == "2":
        # Teacher mode is used to curate reusable lesson drafts before learners consume them.
        run_teacher_console()
        return

    name = input("Enter your name: ").strip()
    simple_id = input("Enter your simple ID: ").strip()

    if not name or not simple_id:
        print("Name and simple ID are required.")
        return

    existing_profile = load_profile(simple_id)
    if existing_profile is None:
        # First login creates a learner profile with the default low initial mastery.
        learner = create_new_learner(simple_id)
        profile_name = name
        save_learner(simple_id, learner, profile_name)
        print("\nNew learner profile created.")
    else:
        learner = Learner.from_dict(existing_profile["learner"])
        profile_name = existing_profile.get("name", name)
        print("\nExisting learner profile loaded.")

    show_learner_summary(profile_name, learner)

    # For now the startup flow stops at course/module selection; later this feeds lesson generation.
    selected_course = choose_course()
    selected_module = choose_module(selected_course)
    save_learner(simple_id, learner, profile_name)

    print("\nSelected course")
    print(f"Course: {selected_course.title}")
    print(f"Primary skill: {selected_course.primary_skill}")

    if selected_module is not None:
        print("\nSelected module")
        print(f"Module: {selected_module.title}")
        if selected_module.chapter_title:
            print(f"Chapter: {selected_module.chapter_title}")
        print(f"Primary skill: {selected_module.primary_skill}")

        document_path = get_document_path_for_course(selected_course)
        if document_path is not None:
            # Learner mode reuses the cached RAG index, but AdaptiveService will first
            # try to serve a teacher-reviewed draft for the learner's mastery band.
            print("\nLoading lesson content...")
            try:
                rag_service, _, _ = build_rag_pipeline(str(document_path))
                adaptive_service = AdaptiveService(
                    learner=learner,
                    engine=DecisionEngine(),
                    rag_service=rag_service,
                )
                lesson = adaptive_service.get_lesson(selected_module)
                print("\nLesson")
                print("------")
                print(lesson)
            except Exception as exc:
                print(f"\nCould not generate lesson content: {exc}")
        else:
            print("\nNo source document is configured for this course yet.")


if __name__ == "__main__":
    main()
