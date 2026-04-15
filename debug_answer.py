from __future__ import annotations

import sys

from llm.answer_prompt_builder import build_answer_generation_prompt
from rag.pipeline import build_rag_pipeline
from rag.grounding import build_grounded_context
from services.content_service import get_course
from services.content_service import get_document_path_for_course
from services.content_service import list_course_modules


def main() -> int:
    course_id = sys.argv[1] if len(sys.argv) > 1 else "math"
    course = get_course(course_id)
    if course is None:
        print(f"Unknown course: {course_id}")
        return 1

    document_path = get_document_path_for_course(course)
    if document_path is None:
        print(f"No source document configured for course: {course_id}")
        return 1

    modules = list_course_modules(course_id)
    if not modules:
        print(f"No modules available for course: {course_id}")
        return 1

    print("=" * 80)
    print("GROUNDED ANSWER DEBUG")
    print("=" * 80)
    print(f"Course: {course.title} ({course.course_id})")
    print(f"Document: {document_path}")

    rag_service, _, _ = build_rag_pipeline(str(document_path))

    selected_module = choose_module(modules)
    if selected_module is None:
        return 1

    print("\nType a learner question to test grounded answer generation.")
    print("Type 'quit' to exit.")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return 0

        result = rag_service.select_transform_and_search(
            query,
            module=selected_module,
            debug=True,
        )

        grounded = build_grounded_context(query, result["chunks"])
        prompt = build_answer_generation_prompt(
            query,
            grounded_context=grounded["context_text"],
            grounded_sources=grounded["sources"],
        )
        answer = rag_service.generate_answer(query, result["chunks"])

        # Keep the debug output focused on pipeline stages and compact metadata.
        print("\n" + "-" * 80)
        print("PIPELINE")
        print(f"Transform: {result['selection']['transform']}")
        print(f"Reason: {result['selection']['reason']}")
        print(f"Candidates: {len(result.get('candidate_chunks', []))}")
        print(f"Reranked for LLM: {len(result.get('chunks', []))}")
        print(f"Grounded sources: {len(grounded.get('sources', []))}")
        print(f"Grounded context chars: {len(grounded.get('context_text', ''))}")
        print(f"Prompt chars: {len(prompt)}")

        # Show which chunks survived reranking without dumping full content.
        print("\nRERANKED CHUNKS")
        _print_chunk_summary(result.get("chunks", []))

        # Show the final sources used to build grounded context.
        print("\nGROUNDED SOURCES")
        for source in grounded["sources"]:
            print(
                f"- Source {source['source_id']}: "
                f"{source.get('chapter') or 'Unknown chapter'} -> "
                f"{source.get('subsection') or source.get('section') or 'Unknown section'} "
                f"| pages {source.get('page_range') or source.get('start_page')}"
            )

        # The answer is the only full-text artifact worth printing by default.
        print("\nGenerated answer:")
        print(answer or "No answer generated.")


def choose_module(modules: list) -> object | None:
    print("\nAvailable modules:")
    for index, module in enumerate(modules, start=1):
        chapter_prefix = f"{module.chapter_title} -> " if module.chapter_title else ""
        print(f"{index}. {chapter_prefix}{module.title}")

    while True:
        choice = input("\nChoose a module by number: ").strip()
        if choice.lower() in {"quit", "exit", "q"}:
            return None
        if choice.isdigit():
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(modules):
                return modules[selected_index]
        print("Invalid module choice. Try again.")


def _print_chunk_summary(chunks: list[dict]) -> None:
    if not chunks:
        print("No chunks selected.")
        return

    for index, chunk in enumerate(chunks, start=1):
        location = chunk.get("subsection") or chunk.get("section") or "Unknown section"
        chapter = chunk.get("chapter") or "Unknown chapter"
        page_range = chunk.get("page_range") or chunk.get("start_page")
        num_chars = len(chunk.get("content") or "")
        print(f"{index}. {chapter} -> {location} | pages {page_range} | chars {num_chars}")


if __name__ == "__main__":
    raise SystemExit(main())
