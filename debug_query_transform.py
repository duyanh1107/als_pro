from __future__ import annotations

import sys
from pathlib import Path

from rag.pipeline import build_rag_pipeline
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
    print("QUERY TRANSFORM DEBUG")
    print("=" * 80)
    print(f"Course: {course.title} ({course.course_id})")
    print(f"Document: {document_path}")

    rag_service, _, _ = build_rag_pipeline(str(document_path))

    selected_module = choose_module(modules)
    if selected_module is None:
        return 1

    print("\nType a learner question to test retrieval.")
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
        # Keep the debug output focused on retrieval stages instead of chunk bodies.
        print("\n" + "-" * 80)
        print("PIPELINE")
        print(f"Transform: {result['selection']['transform']}")
        print(f"Reason: {result['selection']['reason']}")
        print(f"Candidates: {len(result.get('candidate_chunks', []))}")
        print(f"Reranked for LLM: {len(result.get('chunks', []))}")
        print(f"Grounded sources: {len(result.get('grounded_sources', []))}")
        print(f"Grounded context chars: {len(result.get('grounded_context', ''))}")

        # Query info shows the transformed query state without printing large text payloads.
        print("\nQUERY INFO")
        for key, value in result["query_info"].items():
            if isinstance(value, list):
                print(f"- {key}:")
                for index, item in enumerate(value, start=1):
                    print(f"  {index}. {item}")
            else:
                print(f"- {key}: {value}")

        print("\nRERANK INFO")
        for key, value in result.get("rerank_info", {}).items():
            print(f"- {key}: {value}")

        # Only print metadata summaries for retrieved candidates.
        print("\nHYBRID CANDIDATES")
        candidate_chunks = result.get("candidate_chunks", [])
        if not candidate_chunks:
            print("No candidate chunks found.")
            continue

        _print_chunk_summary(candidate_chunks)

        print("\nRERANKED CHUNKS")
        _print_chunk_summary(result.get("chunks", []))

        print("\nGROUNDED SOURCES")
        for source in result.get("grounded_sources", []):
            print(
                f"- Source {source['source_id']}: "
                f"{source.get('chapter') or 'Unknown chapter'} -> "
                f"{source.get('subsection') or source.get('section') or 'Unknown section'} "
                f"| pages {source.get('page_range') or source.get('start_page')}"
            )


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
