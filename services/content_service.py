from __future__ import annotations

import json
from pathlib import Path

from core.models import Course
from core.models import Module
from data.courses.coding import CODING_COURSE
from data.courses.math import MATH_COURSE
from llm.module_filter import decide_module_candidate
from rag.toc_provider import TOC_DIR, parse_toc


COURSE_CATALOG: list[Course] = [MATH_COURSE, CODING_COURSE]
BASE_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = BASE_DIR / "data" / "modules"
MODULE_STRATEGY_PATH = MODULES_DIR / "active_strategy.json"


def list_courses() -> list[Course]:
    return COURSE_CATALOG


def get_course(course_id: str) -> Course | None:
    for course in COURSE_CATALOG:
        if course.course_id == course_id:
            return course
    return None


def get_toc_path_for_course(course: Course) -> Path | None:
    # A course only supports TOC-driven modules when it points to a source document stem.
    if not course.source_name:
        return None

    toc_path = TOC_DIR / f"{course.source_name}.txt"
    if not toc_path.exists():
        return None
    return toc_path


def get_document_path_for_course(course: Course) -> Path | None:
    # The course source name is reused to map modules back to the PDF used by RAG.
    if not course.source_name:
        return None

    document_path = BASE_DIR / "data" / "documents" / f"{course.source_name}.pdf"
    if not document_path.exists():
        return None
    return document_path


def create_modules_from_toc(
    course: Course,
    level: str = "section",
    use_llm_filter: bool = False,
) -> list[Module]:
    """
    Turn a course TOC into a sequence of learning modules.

    Current default:
    - one module per TOC section
    - chapter title is attached as parent context
    """
    toc_path = get_toc_path_for_course(course)
    if toc_path is None:
        return []

    toc_text = toc_path.read_text(encoding="utf-8")
    toc_entries = parse_toc(toc_text)

    modules: list[Module] = []
    current_chapter_title: str | None = None

    for entry in toc_entries:
        if entry["level"] == "chapter":
            # Remember the parent chapter so each section module keeps its broader context.
            current_chapter_title = entry["title"]
            continue

        if entry["level"] != level:
            continue

        # Filtering can be heuristic-only or LLM-backed depending on the caller.
        include_entry, reason = decide_module_candidate(
            course_title=course.title,
            course_description=course.description,
            chapter_title=current_chapter_title,
            section_title=entry["title"],
            level=entry["level"],
            use_llm=use_llm_filter,
        )
        if not include_entry:
            continue

        toc_number = _extract_toc_number(toc_text, entry["title"], entry["page"])
        modules.append(
            Module(
                module_id=f"{course.course_id}:{toc_number or len(modules) + 1}",
                course_id=course.course_id,
                title=entry["title"],
                primary_skill=course.primary_skill,
                skills=list(course.skills),
                chapter_title=current_chapter_title,
                toc_number=toc_number,
                start_page=entry["page"],
                level=entry["level"],
                selection_reason=reason,
            )
        )

    return modules


def list_course_modules(
    course_id: str,
    level: str = "section",
    use_llm_filter: bool = False,
    strategy: str | None = None,
) -> list[Module]:
    course = get_course(course_id)
    if course is None:
        return []
    selected_strategy = strategy or get_active_module_strategy(course_id)
    return load_or_create_course_modules(course, level=level, strategy=selected_strategy, use_llm_filter=use_llm_filter)


def get_modules_cache_path(course: Course, level: str, strategy: str) -> Path:
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    return MODULES_DIR / f"{course.course_id}_{level}_{strategy}.json"


def is_modules_cache_valid(course: Course, level: str, strategy: str) -> bool:
    toc_path = get_toc_path_for_course(course)
    cache_path = get_modules_cache_path(course, level, strategy)
    if toc_path is None or not cache_path.exists():
        return False

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    # Rebuild if the TOC file changed or if the filtering mode changed.
    return (
        payload.get("course_id") == course.course_id
        and payload.get("level") == level
        and payload.get("strategy") == strategy
        and payload.get("toc_path") == str(toc_path.resolve())
        and payload.get("toc_mtime") == toc_path.stat().st_mtime
    )


def save_course_modules(course: Course, modules: list[Module], level: str, strategy: str) -> None:
    toc_path = get_toc_path_for_course(course)
    if toc_path is None:
        return

    cache_path = get_modules_cache_path(course, level, strategy)
    payload = {
        "course_id": course.course_id,
        "level": level,
        "strategy": strategy,
        "toc_path": str(toc_path.resolve()),
        "toc_mtime": toc_path.stat().st_mtime,
        # Persist raw module data so future runs avoid repeating TOC filtering work.
        "modules": [module.to_dict() for module in modules],
    }

    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_course_modules(course: Course, level: str, strategy: str) -> list[Module]:
    cache_path = get_modules_cache_path(course, level, strategy)
    with cache_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return [Module.from_dict(item) for item in payload.get("modules", [])]


def load_or_create_course_modules(
    course: Course,
    level: str = "section",
    strategy: str = "heuristic",
    use_llm_filter: bool = False,
) -> list[Module]:
    # Module generation is cached because TOC + LLM filtering should not rerun every startup.
    if is_modules_cache_valid(course, level, strategy):
        return load_course_modules(course, level, strategy)

    if strategy == "manual":
        return []

    modules = create_modules_from_toc(course, level=level, use_llm_filter=(strategy == "llm" or use_llm_filter))
    save_course_modules(course, modules, level, strategy)
    return modules


def create_candidate_modules_from_toc(course: Course, level: str = "section") -> list[Module]:
    toc_path = get_toc_path_for_course(course)
    if toc_path is None:
        return []

    toc_text = toc_path.read_text(encoding="utf-8")
    toc_entries = parse_toc(toc_text)

    modules: list[Module] = []
    current_chapter_title: str | None = None

    for entry in toc_entries:
        if entry["level"] == "chapter":
            current_chapter_title = entry["title"]
            continue

        if entry["level"] != level:
            continue

        toc_number = _extract_toc_number(toc_text, entry["title"], entry["page"])
        modules.append(
            Module(
                module_id=f"{course.course_id}:{toc_number or len(modules) + 1}",
                course_id=course.course_id,
                title=entry["title"],
                primary_skill=course.primary_skill,
                skills=list(course.skills),
                chapter_title=current_chapter_title,
                toc_number=toc_number,
                start_page=entry["page"],
                level=entry["level"],
                selection_reason="Raw TOC candidate before filtering.",
            )
        )

    return modules


def save_manual_course_modules(course: Course, modules: list[Module], level: str = "section") -> None:
    save_course_modules(course, modules, level, "manual")


def set_active_module_strategy(course_id: str, strategy: str, level: str = "section") -> None:
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {}
    if MODULE_STRATEGY_PATH.exists():
        try:
            payload = json.loads(MODULE_STRATEGY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}

    payload[course_id] = {"strategy": strategy, "level": level}
    MODULE_STRATEGY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_active_module_strategy(course_id: str, default: str = "heuristic") -> str:
    if not MODULE_STRATEGY_PATH.exists():
        return default

    try:
        payload = json.loads(MODULE_STRATEGY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default

    course_payload = payload.get(course_id, {})
    return course_payload.get("strategy", default)


def clear_course_module_caches(course_id: str) -> None:
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    for path in MODULES_DIR.glob(f"{course_id}_*.json"):
        path.unlink(missing_ok=True)


def _extract_toc_number(toc_text: str, title: str, page: int) -> str | None:
    for line in toc_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if title in stripped and stripped.endswith(str(page)):
            parts = stripped.split(maxsplit=1)
            if parts:
                return parts[0]
    return None
