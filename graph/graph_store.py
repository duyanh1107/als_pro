from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = BASE_DIR / "data" / "graphs"


def get_course_graph_path(course_id: str) -> Path:
    """Return the JSON location for one course graph and create the folder if needed."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    return GRAPHS_DIR / f"{course_id}_module_graph.json"


def get_concept_graph_path(scope_id: str) -> Path:
    """Return the JSON location for a concept graph.

    The scope can be a single course id like ``math`` or a broader id such as
    ``catalog`` when we later want one graph shared across multiple courses.
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    return GRAPHS_DIR / f"{scope_id}_concept_graph.json"


def get_concept_extraction_cache_path(course_id: str) -> Path:
    """Return the JSON cache location for per-module concept extraction results.

    The cache is course-scoped because the relevant chunks and module ids are
    also scoped to one course's content index.
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    return GRAPHS_DIR / f"{course_id}_concept_extract_cache.json"


def save_module_graph(course_id: str, graph: dict[str, Any]) -> Path:
    """Persist the latest graph snapshot so the viewer can reuse it."""
    path = get_course_graph_path(course_id)
    with path.open("w", encoding="utf-8") as file:
        json.dump(graph, file, ensure_ascii=False, indent=2)
    return path


def save_concept_graph(scope_id: str, graph: dict[str, Any]) -> Path:
    """Persist the latest concept graph snapshot."""
    path = get_concept_graph_path(scope_id)
    with path.open("w", encoding="utf-8") as file:
        json.dump(graph, file, ensure_ascii=False, indent=2)
    return path


def save_concept_extraction_cache(course_id: str, cache: dict[str, Any]) -> Path:
    """Persist cached concept extraction results for one course."""
    path = get_concept_extraction_cache_path(course_id)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)
    return path


def load_module_graph(course_id: str) -> dict[str, Any]:
    """Load a previously built graph from disk."""
    path = get_course_graph_path(course_id)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_concept_graph(scope_id: str) -> dict[str, Any]:
    """Load a previously built concept graph from disk."""
    path = get_concept_graph_path(scope_id)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_concept_extraction_cache(course_id: str) -> dict[str, Any]:
    """Load cached concept extraction results for one course if present."""
    path = get_concept_extraction_cache_path(course_id)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
