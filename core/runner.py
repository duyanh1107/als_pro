from __future__ import annotations

from core.models import Course
from core.models import LessonDraft
from core.models import Module
from rag.pipeline import build_rag_pipeline
from services.content_service import clear_course_module_caches
from services.content_service import create_candidate_modules_from_toc
from services.content_service import get_active_module_strategy
from services.content_service import get_course
from services.content_service import get_document_path_for_course
from services.content_service import list_courses
from services.content_service import list_course_modules
from services.content_service import save_manual_course_modules
from services.content_service import set_active_module_strategy
from services.lesson_service import create_teacher_lesson_draft
from services.lesson_store import save_lesson_draft


def choose_course() -> Course:
    courses = list_courses()

    print("\nAvailable courses:")
    for index, course in enumerate(courses, start=1):
        print(f"{index}. {course.title} ({course.course_id})")
        print(f"   {course.description}")

    while True:
        choice = input("\nChoose a course by number: ").strip()
        if choice.isdigit():
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(courses):
                return courses[selected_index]
        print("Invalid course choice. Try again.")


def show_learner_summary(profile_name: str, learner) -> None:
    total_answers = learner.correct_count + learner.wrong_count
    print("\nLearner profile")
    print(f"Name: {profile_name}")
    print(f"Answered questions: {total_answers}")

    mastery = learner.get_all_mastery()
    if mastery:
        print("Known skill mastery:")
        for skill, value in mastery.items():
            print(f"- {skill}: {value:.2f}")
    else:
        print("No mastery history yet. Starting from a below-medium baseline.")


def choose_module(course: Course) -> Module | None:
    # Modules are loaded from cached TOC-derived data when available.
    modules = list_course_modules(course.course_id)
    if not modules:
        print("\nNo modules available yet for this course.")
        return None

    print(f"\nModule source: {get_active_module_strategy(course.course_id)}")

    print("\nAvailable modules:")
    for index, module in enumerate(modules, start=1):
        chapter_prefix = f"{module.chapter_title} -> " if module.chapter_title else ""
        page_suffix = f" (page {module.start_page})" if module.start_page is not None else ""
        print(f"{index}. {chapter_prefix}{module.title}{page_suffix}")

    while True:
        choice = input("\nChoose a module by number: ").strip()
        if choice.isdigit():
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(modules):
                return modules[selected_index]
        print("Invalid module choice. Try again.")


def run_teacher_module_setup() -> None:
    print("\nTeacher Module Setup")
    print("====================")

    course = choose_course()
    clear_course_module_caches(course.course_id)

    heuristic_modules = list_course_modules(course.course_id, strategy="heuristic")
    llm_modules = list_course_modules(course.course_id, strategy="llm")

    print(f"\nHeuristic modules: {len(heuristic_modules)}")
    _print_module_preview(heuristic_modules)
    print(f"\nLLM modules: {len(llm_modules)}")
    _print_module_preview(llm_modules)

    print("\nChoose module set:")
    print("1. Use heuristic modules")
    print("2. Use LLM modules")
    print("3. Build manual module set")

    while True:
        choice = input("\nChoose an option: ").strip()
        if choice == "1":
            set_active_module_strategy(course.course_id, "heuristic")
            print("\nActive module strategy set to heuristic.")
            return
        if choice == "2":
            set_active_module_strategy(course.course_id, "llm")
            print("\nActive module strategy set to llm.")
            return
        if choice == "3":
            _run_manual_module_selection(course)
            return
        print("Invalid choice. Try again.")


def run_teacher_console() -> None:
    print("\nTeacher Tools")
    print("=============")
    # Keep module curation and lesson drafting separate so a teacher can update
    # the module list without being forced to regenerate lesson content.
    print("1. Configure module set from TOC")
    print("2. Generate lesson draft for a module")
    print("3. Print module -> chunk links")

    while True:
        choice = input("\nChoose an option: ").strip()
        if choice == "1":
            run_teacher_module_setup()
            return
        if choice == "2":
            run_teacher_lesson_setup()
            return
        if choice == "3":
            run_module_chunk_debug()
            return
        print("Invalid choice. Try again.")


def run_teacher_lesson_setup() -> None:
    print("\nTeacher Lesson Draft Setup")
    print("==========================")

    course = choose_course()
    module = choose_module(course)
    if module is None:
        return

    document_path = get_document_path_for_course(course)
    if document_path is None:
        print("\nNo source document is configured for this course yet.")
        return

    mastery_score = _choose_target_mastery_score()

    print("\nLoading module sources...")
    rag_service, _, _ = build_rag_pipeline(str(document_path))

    print("Generating teacher lesson draft...")
    # Teacher drafts are generated from the same RAG retrieval path the learner uses,
    # but they are saved so future learner sessions can reuse reviewed content.
    draft = create_teacher_lesson_draft(course, module, rag_service, mastery_score)
    _print_draft_source_summary(draft)

    if _confirm_yes_no("\nDo you want to edit the lesson text now? (y/n): "):
        # A quick inline edit path is useful before opening the richer Streamlit review UI.
        draft.edited_lesson = _collect_multiline_text(
            "Paste the revised lesson below. Type END on its own line to finish.",
            initial_text=draft.edited_lesson or draft.generated_lesson,
        )
        draft.status = "reviewed"

    path = save_lesson_draft(draft)

    print("\nLesson draft saved.")
    print(f"Draft path: {path}")
    print("Open the local review app with:")
    print("streamlit run app/streamlit_app.py")


def run_module_chunk_debug() -> None:
    print("\nModule -> Chunk Debug")
    print("=====================")

    course = choose_course()
    modules = list_course_modules(course.course_id)
    if not modules:
        print("\nNo modules available for this course.")
        return

    document_path = get_document_path_for_course(course)
    if document_path is None:
        print("\nNo source document is configured for this course yet.")
        return

    print("\nLoading RAG index for debug mapping...")
    rag_service, _, _ = build_rag_pipeline(str(document_path))

    for module in modules:
        print("\n" + "-" * 80)
        print(f"Module: {module.title}")
        if module.chapter_title:
            print(f"Chapter: {module.chapter_title}")
        print(f"Module ID: {module.module_id}")

        chunks = rag_service.retrieve_for_module(module, k=5)
        if not chunks:
            print("No matching chunks found.")
            continue

        print(f"Matched chunks: {len(chunks)}")
        for index, chunk in enumerate(chunks, start=1):
            location = chunk.get("subsection") or chunk.get("section") or "Unknown section"
            page_range = chunk.get("page_range") or f"{chunk.get('start_page')}"
            part_suffix = ""
            if chunk.get("chunk_part_count", 1) > 1:
                part_suffix = (
                    f" | part {int(chunk.get('chunk_part_index', 0)) + 1}"
                    f"/{chunk.get('chunk_part_count')}"
                )
            print(f"{index}. {location} | pages {page_range}{part_suffix}")


def _run_manual_module_selection(course: Course) -> None:
    candidates = create_candidate_modules_from_toc(course)
    if not candidates:
        print("\nNo TOC candidates available for manual selection.")
        return

    print("\nManual module candidates:")
    for index, module in enumerate(candidates, start=1):
        chapter_prefix = f"{module.chapter_title} -> " if module.chapter_title else ""
        print(f"{index}. {chapter_prefix}{module.title}")

    print("\nEnter comma-separated module numbers to keep.")
    while True:
        raw = input("Selection: ").strip()
        try:
            indices = [int(part.strip()) - 1 for part in raw.split(",") if part.strip()]
        except ValueError:
            print("Invalid input. Use comma-separated numbers like 1,2,5")
            continue

        if not indices or any(index < 0 or index >= len(candidates) for index in indices):
            print("Invalid selection. Try again.")
            continue

        selected_modules = []
        for index in indices:
            module = candidates[index]
            module.selection_reason = "Selected manually by teacher."
            selected_modules.append(module)

        save_manual_course_modules(course, selected_modules)
        set_active_module_strategy(course.course_id, "manual")
        print(f"\nSaved {len(selected_modules)} manual modules and set strategy to manual.")
        return


def _print_module_preview(modules: list[Module], limit: int = 10) -> None:
    for module in modules[:limit]:
        chapter_prefix = f"{module.chapter_title} -> " if module.chapter_title else ""
        print(f"- {chapter_prefix}{module.title}")
    if len(modules) > limit:
        print(f"... and {len(modules) - limit} more")


def _choose_target_mastery_score() -> float:
    # The saved draft is keyed by mastery band so teachers can curate different
    # pacing versions of the same module for weaker vs stronger learners.
    print("\nSelect target learner mastery:")
    print("1. Low mastery lesson")
    print("2. Medium mastery lesson")
    print("3. High mastery lesson")

    while True:
        choice = input("\nChoose mastery band: ").strip()
        if choice == "1":
            return 0.25
        if choice == "2":
            return 0.55
        if choice == "3":
            return 0.85
        print("Invalid choice. Try again.")


def _print_draft_source_summary(draft: LessonDraft) -> None:
    print("\nRetrieved source chunks:")
    for index, chunk in enumerate(draft.source_chunks, start=1):
        location = chunk.get("subsection") or chunk.get("section") or "Unknown section"
        page_suffix = f" | page {chunk.get('start_page')}" if chunk.get("start_page") is not None else ""
        print(f"{index}. {location}{page_suffix}")

    print("\nGenerated lesson preview:")
    preview = (draft.generated_lesson or "").strip()
    if len(preview) > 800:
        preview = preview[:800].rstrip() + "..."
    print(preview)


def _confirm_yes_no(prompt: str) -> bool:
    while True:
        choice = input(prompt).strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _collect_multiline_text(prompt: str, initial_text: str | None = None) -> str:
    print(prompt)
    if initial_text:
        print("\nCurrent lesson:")
        print(initial_text)
        print("\nEnter replacement text. Type END on its own line to save.")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip() or (initial_text or "")
