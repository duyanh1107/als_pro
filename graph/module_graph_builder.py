from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.models import Course
    from core.models import Module
    from graph.graph_store import save_module_graph
    from services.content_service import get_course
    from services.content_service import list_course_modules
else:
    from core.models import Course
    from core.models import Module
    from .graph_store import save_module_graph
    from services.content_service import get_course
    from services.content_service import list_course_modules


SIMILARITY_THRESHOLD = 0.82


def build_course_module_graph_for_id(
    course_id: str,
    strategy: str | None = None,
) -> dict:
    """Build and persist the graph for one course id."""
    course = get_course(course_id)
    if course is None:
        raise ValueError(f"Unknown course: {course_id}")

    modules = list_course_modules(course_id, strategy=strategy)
    graph = build_course_module_graph(course, modules)
    save_module_graph(course_id, graph)
    return graph


def build_course_module_graph(course: Course, modules: list[Module]) -> dict:
    """Turn course modules into graph nodes plus two edge types."""
    ordered_modules = sorted(
        modules,
        key=lambda module: (
            module.start_page or 0,
            _toc_sort_key(module.toc_number),
            module.chapter_title or "",
            module.title.lower(),
        ),
    )
    nodes = [_build_module_node(module) for module in ordered_modules]
    module_vectors = _build_module_vectors(course, ordered_modules)
    edges = _build_edges(ordered_modules, module_vectors)

    # Keep graph metadata explicit so downstream recommendation logic can tell
    # this graph includes structural ordering plus semantic similarity.
    return {
        "course_id": course.course_id,
        "course_title": course.title,
        "graph_version": "phase_2_toc_plus_similarity",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _build_module_node(module: Module) -> dict:
    """Keep node data close to the original module so the graph stays explainable."""
    return {
        "id": module.module_id,
        "type": "module",
        "course_id": module.course_id,
        "title": module.title,
        "chapter_title": module.chapter_title,
        "toc_number": module.toc_number,
        "start_page": module.start_page,
        "primary_skill": module.primary_skill,
        "skills": list(module.skills),
        "level": module.level,
    }


def _build_edges(
    modules: list[Module],
    module_vectors: dict[str, np.ndarray],
) -> list[dict]:
    """Create two edge types:
    - prerequisite_of: safe TOC order signal
    - similar_to: embedding-based semantic closeness
    """
    edges: list[dict] = []

    for index, module in enumerate(modules):
        next_module = modules[index + 1] if index + 1 < len(modules) else None
        if next_module and module.chapter_title == next_module.chapter_title:
            # Consecutive modules inside the same chapter are the safest first-pass
            # prerequisite links when we only have TOC ordering and module metadata.
            edges.append(
                {
                    "source": module.module_id,
                    "target": next_module.module_id,
                    "relation": "prerequisite_of",
                    "directed": True,
                    "weight": 0.9,
                    "reason": "Consecutive modules in the same chapter usually build on each other.",
                }
            )

        for other in modules[index + 1 :]:
            similar_edge = _infer_similar_edge(module, other, module_vectors)
            if similar_edge is not None:
                edges.append(similar_edge)

    return edges


def _infer_similar_edge(
    left: Module,
    right: Module,
    module_vectors: dict[str, np.ndarray],
) -> dict | None:
    """Infer semantic similarity from module-level embedding centroids.

    This does not use chapter position. Two modules can be similar because
    their associated content is semantically close.
    """
    left_vector = module_vectors.get(left.module_id)
    right_vector = module_vectors.get(right.module_id)
    if left_vector is None or right_vector is None:
        return None

    similarity = _cosine_similarity(left_vector, right_vector)
    if similarity < SIMILARITY_THRESHOLD:
        return None

    weight = similarity
    reason_parts = [f"module embedding similarity={similarity:.2f}"]
    if left.course_id != right.course_id:
        reason_parts.append("cross-course similarity candidate")

    return {
        "source": left.module_id,
        "target": right.module_id,
        "relation": "similar_to",
        "directed": False,
        "weight": round(weight, 2),
        "reason": ", ".join(reason_parts),
    }


def _build_module_vectors(course: Course, modules: list[Module]) -> dict[str, np.ndarray]:
    """Build one vector per module from the chunk embeddings linked to it."""
    index_dir = _get_course_index_dir(course)
    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"
    if not chunks_path.exists() or not embeddings_path.exists():
        return {}

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    if len(chunks) != len(embeddings):
        return {}

    module_vectors: dict[str, np.ndarray] = {}
    for module in modules:
        # Represent each module by the centroid of the chunk embeddings that map
        # back to that module's section/subsection. This keeps similarity grounded
        # in the module's associated content instead of only its title text.
        matching_vectors = [
            embeddings[index]
            for index, chunk in enumerate(chunks)
            if _chunk_matches_module(chunk, module)
        ]
        if not matching_vectors:
            continue
        centroid = np.mean(np.stack(matching_vectors), axis=0)
        module_vectors[module.module_id] = _normalize_vector(centroid)

    return module_vectors


def _get_course_index_dir(course: Course) -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    source_name = course.source_name or course.course_id
    return base_dir / "data" / "indexes" / source_name


def _chunk_matches_module(chunk: dict, module: Module) -> bool:
    # A chunk belongs to a module when it comes from the same chapter and its
    # section or subsection title matches the module title.
    chunk_chapter = (chunk.get("chapter") or "").strip().lower()
    module_chapter = (module.chapter_title or "").strip().lower()
    if chunk_chapter != module_chapter:
        return False

    chunk_section = (chunk.get("section") or "").strip().lower()
    chunk_subsection = (chunk.get("subsection") or "").strip().lower()
    module_title = module.title.strip().lower()
    return chunk_section == module_title or chunk_subsection == module_title


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize so cosine similarity is just a dot product later."""
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _toc_sort_key(toc_number: str | None) -> tuple:
    """Sort "2.10" after "2.9" using numeric pieces where possible."""
    if not toc_number:
        return (float("inf"),)

    parts: list[int | str] = []
    for piece in toc_number.split("."):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            parts.append(piece)
    return tuple(parts)


def main() -> int:
    course_id = sys.argv[1] if len(sys.argv) > 1 else "math"
    graph = build_course_module_graph_for_id(course_id)
    print(f"Built module graph for course: {graph['course_id']}")
    print(f"Nodes: {graph['node_count']}")
    print(f"Edges: {graph['edge_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
