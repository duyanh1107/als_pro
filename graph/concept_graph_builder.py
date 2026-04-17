from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.models import Course
    from core.models import Module
    from graph.graph_store import load_concept_extraction_cache
    from graph.graph_store import save_concept_graph
    from graph.graph_store import save_concept_extraction_cache
    from llm.client import get_openai_client
    from services.concept_store import load_course_concept_review
    from services.content_service import get_course
    from services.content_service import list_course_modules
    from services.content_service import list_courses
else:
    from core.models import Course
    from core.models import Module
    from .graph_store import load_concept_extraction_cache
    from .graph_store import save_concept_graph
    from .graph_store import save_concept_extraction_cache
    from llm.client import get_openai_client
    from services.concept_store import load_course_concept_review
    from services.content_service import get_course
    from services.content_service import list_course_modules
    from services.content_service import list_courses


WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
GENERIC_CONCEPT_LABELS = {
    "summary",
    "some examples",
    "definitions and examples",
    "definitions",
    "examples",
    "the main result",
    "basic definitions",
    "further results",
    "further remarks",
    "further properties",
    "algebraic laws",
    "the algebraic laws",
}
PEDAGOGICAL_PREFIXES = (
    "the definition of ",
    "definition of ",
    "the notion of ",
    "notion of ",
    "what is ",
    "the basics of ",
    "basics of ",
    "an introduction to ",
    "introduction to ",
    "some applications of ",
    "applications of ",
    "application to ",
    "applications to ",
    "geometric applications of ",
    "further results on ",
    "further remarks on ",
    "further properties of ",
    "some results on ",
    "the basic definitions of ",
    "the beginning of ",
)
SPLIT_SEPARATORS = (" via ", ":", ";")
DEFAULT_CONCEPT_EXTRACT_MODEL = "gpt-4o-mini"


def build_course_concept_graph_for_id(
    course_id: str,
    strategy: str | None = None,
) -> dict:
    """Build and persist a concept graph for one course."""
    course = get_course(course_id)
    if course is None:
        raise ValueError(f"Unknown course: {course_id}")

    modules = list_course_modules(course_id, strategy=strategy)
    graph = build_course_concept_graph(course, modules)
    save_concept_graph(course_id, graph)
    return graph


def build_catalog_concept_graph(
    course_ids: list[str] | None = None,
    strategy: str | None = None,
) -> dict:
    """Build a reusable concept graph across every course that currently has modules.

    Shared concept ids are global on purpose, so the graph can grow across
    courses instead of being trapped inside one course-specific namespace.
    """
    allowed_course_ids = set(course_ids) if course_ids is not None else None
    courses = [
        course
        for course in list_courses()
        if allowed_course_ids is None or course.course_id in allowed_course_ids
    ]
    modules_by_course = {
        course.course_id: list_course_modules(course.course_id, strategy=strategy)
        for course in courses
    }
    graph = _build_concept_graph(
        scope="catalog",
        scope_id="catalog",
        courses=courses,
        modules_by_course=modules_by_course,
    )
    save_concept_graph("catalog", graph)
    return graph


def build_course_concept_graph(course: Course, modules: list[Module]) -> dict:
    """Build a concept graph scoped to one course.

    The builder itself is not math-specific: it works from module titles and
    matching subsection headings, so any future TOC-driven course can reuse it.
    """
    return _build_concept_graph(
        scope="course",
        scope_id=course.course_id,
        courses=[course],
        modules_by_course={course.course_id: modules},
    )


def _build_concept_graph(
    scope: str,
    scope_id: str,
    courses: list[Course],
    modules_by_course: dict[str, list[Module]],
) -> dict:
    # `concept_nodes` is keyed by global concept id so the same concept can be
    # reused across multiple courses instead of being duplicated per course.
    concept_nodes: dict[str, dict] = {}
    module_nodes: list[dict] = []
    edges: list[dict] = []
    ordered_modules_by_course: dict[str, list[Module]] = {}
    module_concepts: dict[str, list[dict]] = {}

    for course in courses:
        modules = sorted(
            modules_by_course.get(course.course_id, []),
            key=lambda module: (
                module.start_page or 0,
                _toc_sort_key(module.toc_number),
                module.chapter_title or "",
                module.title.lower(),
            ),
        )
        ordered_modules_by_course[course.course_id] = modules
        chunks = _load_course_chunks(course)
        concept_cache = load_concept_extraction_cache(course.course_id)
        concept_review = load_course_concept_review(course.course_id)
        cache_changed = False

        for module in modules:
            module_nodes.append(_build_module_node(module))
            # Concepts come from the module title plus matching subsection titles.
            # This keeps extraction grounded in course structure rather than trying
            # to mine arbitrary entities from raw paragraphs too early.
            extracted, used_cache = _extract_module_concepts(module, chunks, concept_cache)
            if not used_cache:
                cache_changed = True
            if not extracted:
                extracted = [_fallback_module_concept(module)]
            extracted = _apply_teacher_concept_review(extracted, concept_review)
            module_concepts[module.module_id] = extracted

            for concept in extracted:
                concept_node = concept_nodes.get(concept["id"])
                if concept_node is None:
                    concept_node = {
                        "id": concept["id"],
                        "type": "concept",
                        "label": concept["label"],
                        "aliases": [concept["label"]],
                        "courses": [module.course_id],
                        "module_ids": [module.module_id],
                        "source_titles": [concept["source_title"]],
                    }
                    concept_nodes[concept["id"]] = concept_node
                else:
                    if concept["label"] not in concept_node["aliases"]:
                        concept_node["aliases"].append(concept["label"])
                    if module.course_id not in concept_node["courses"]:
                        concept_node["courses"].append(module.course_id)
                    if module.module_id not in concept_node["module_ids"]:
                        concept_node["module_ids"].append(module.module_id)
                    if concept["source_title"] not in concept_node["source_titles"]:
                        concept_node["source_titles"].append(concept["source_title"])

                edges.append(
                    {
                        "source": module.module_id,
                        "target": concept["id"],
                        "relation": "module_teaches_concept",
                        "directed": True,
                        "weight": round(concept["score"], 2),
                        "reason": (
                            f"Concept extracted from module-linked title: "
                            f"{concept['source_title']}"
                        ),
                    }
                )

        # Cache writes happen once per course so we do not rewrite the file for
        # every module while still keeping rebuilds fast on the next run.
        if cache_changed:
            save_concept_extraction_cache(course.course_id, concept_cache)
        _merge_teacher_only_concepts(concept_nodes, course.course_id, concept_review)

    # Co-occurrence inside the same module is the safest first soft-link signal
    # for concepts that clearly belong to the same local teaching unit.
    edges.extend(_build_concept_related_edges(module_concepts))
    edges.extend(_build_concept_prerequisite_edges(ordered_modules_by_course, module_concepts))
    _annotate_concept_core_scores(concept_nodes, edges, module_count=len(module_nodes))

    nodes = module_nodes + list(concept_nodes.values())
    return {
        "scope": scope,
        "scope_id": scope_id,
        "graph_type": "concept_graph",
        "graph_version": "phase_2_llm_concept_graph_teacher_review",
        "course_ids": [course.course_id for course in courses],
        "course_titles": [course.title for course in courses],
        "module_count": len(module_nodes),
        "concept_count": len(concept_nodes),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _build_module_node(module: Module) -> dict:
    """Mirror the learner-facing module as a node inside the concept graph."""
    return {
        "id": module.module_id,
        "type": "module",
        "course_id": module.course_id,
        "title": module.title,
        "chapter_title": module.chapter_title,
        "toc_number": module.toc_number,
        "start_page": module.start_page,
        "level": module.level,
    }


def _load_course_chunks(course: Course) -> list[dict]:
    """Load chunk metadata so a module can inherit its subsection vocabulary."""
    index_dir = _get_course_index_dir(course)
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.exists():
        return []
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def _extract_module_concepts(
    module: Module,
    chunks: list[dict],
    concept_cache: dict[str, dict],
) -> tuple[list[dict], bool]:
    """Rank candidate concepts for a module.

    The primary path uses one LLM call per module so concept keywords can be
    normalized better than simple string splitting. If that fails, we fall back
    to the older heading-based heuristic extraction.
    """
    source_titles = [module.title]
    source_titles.extend(
        _module_subsection_titles(module, chunks)
    )
    relevant_chunks = _module_relevant_chunks(module, chunks)
    cache_key = _build_module_concept_cache_key(module, source_titles, relevant_chunks)
    cached = concept_cache.get(module.module_id)
    if cached and cached.get("cache_key") == cache_key:
        # Returning cached concepts avoids repeating one LLM call per module
        # when the module content and extraction model have not changed.
        return _deserialize_cached_concepts(cached.get("concepts", [])), True

    extracted = _extract_module_concepts_llm(module, source_titles, relevant_chunks)
    if extracted:
        concept_cache[module.module_id] = {
            "cache_key": cache_key,
            "concepts": _serialize_cached_concepts(extracted),
            "strategy": "llm",
        }
        return extracted, False

    extracted = _extract_module_concepts_heuristic(module, source_titles)
    concept_cache[module.module_id] = {
        "cache_key": cache_key,
        "concepts": _serialize_cached_concepts(extracted),
        "strategy": "heuristic_fallback",
    }
    return extracted, False


def _apply_teacher_concept_review(extracted: list[dict], concept_review) -> list[dict]:
    """Filter generated module concepts through the teacher's course-level review.

    Teachers can either trim the generated set (`augment`) or provide an
    explicit course concept list (`replace`). We keep module concept edges only
    when the concept survives that course-level review.
    """
    if concept_review is None:
        return extracted

    generated = list(extracted)
    removed_labels = {_normalize_label(label) for label in concept_review.removed_concepts}

    if concept_review.mode == "replace":
        allowed_labels = {_normalize_label(label) for label in concept_review.replacement_concepts}
        return [concept for concept in generated if concept["label"] in allowed_labels]

    return [concept for concept in generated if concept["label"] not in removed_labels]


def _merge_teacher_only_concepts(
    concept_nodes: dict[str, dict],
    course_id: str,
    concept_review,
) -> None:
    """Add teacher-entered concepts that are not present in generated module edges.

    This keeps course-level curation visible in the graph even before we build a
    finer teacher workflow for mapping every manual concept back to modules.
    """
    if concept_review is None:
        return

    if concept_review.mode == "replace":
        teacher_labels = concept_review.replacement_concepts
    else:
        teacher_labels = concept_review.added_concepts

    for raw_label in teacher_labels:
        label = _normalize_label(raw_label)
        if not _is_informative_label(label):
            continue
        concept_id = _concept_id(label)
        concept_node = concept_nodes.get(concept_id)
        if concept_node is None:
            concept_nodes[concept_id] = {
                "id": concept_id,
                "type": "concept",
                "label": label,
                "aliases": [label],
                "courses": [course_id],
                "module_ids": [],
                "source_titles": ["teacher override"],
            }
            continue

        if course_id not in concept_node["courses"]:
            concept_node["courses"].append(course_id)
        if "teacher override" not in concept_node["source_titles"]:
            concept_node["source_titles"].append("teacher override")


def _extract_module_concepts_llm(
    module: Module,
    source_titles: list[str],
    relevant_chunks: list[dict],
    model: str = DEFAULT_CONCEPT_EXTRACT_MODEL,
) -> list[dict]:
    """Ask the model for canonical concept keywords for one module.

    Headings stay in the prompt because they are compact summaries, but the
    model also sees every chunk linked to the module so it can tell whether a
    keyword is actually central in the content instead of only appearing in a
    title.
    """
    normalized_titles = [title.strip() for title in source_titles if title and title.strip()]
    if not normalized_titles and not relevant_chunks:
        return []

    unique_titles = list(dict.fromkeys(normalized_titles))
    chunk_blocks = []
    for index, chunk in enumerate(relevant_chunks, start=1):
        chunk_blocks.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"Pages: {chunk.get('page_range') or chunk.get('start_page') or 'unknown'}",
                    f"Section: {chunk.get('section') or 'Unknown'}",
                    f"Subsection: {chunk.get('subsection') or 'Unknown'}",
                    "Content:",
                    (chunk.get("content") or "").strip(),
                ]
            )
        )
    chunks_block = "\n\n".join(chunk_blocks) if chunk_blocks else "No matched chunks."
    prompt = f"""
You are extracting canonical concept keywords for an educational knowledge graph.

Course id: {module.course_id}
Module id: {module.module_id}
Module title: {module.title}
Chapter title: {module.chapter_title or "Unknown"}

Relevant headings from this module:
{chr(10).join(f"- {title}" for title in unique_titles)}

Relevant chunks from this module:
{chunks_block}

Task:
- Return the most important domain concepts taught by this module.
- Read all relevant chunks, not just the headings.
- Prefer concepts that recur across the chunks or are central to the explanations.
- Prefer reusable concept keywords, not pedagogical phrases.
- Keep concepts short and canonical.
- Singularize when natural, for example "matrices" -> "matrix".
- Avoid labels like "examples", "main result", "further remarks", "applications".
- Include at most 5 concepts.
- Output only concepts that are clearly supported by the headings and chunks above.

Return JSON only in this shape:
{{
  "concepts": ["matrix", "matrix addition", "vector"]
}}
"""

    try:
        response = get_openai_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        raw_concepts = payload.get("concepts", [])
        if not isinstance(raw_concepts, list):
            return []

        scored_candidates: dict[str, dict] = {}
        for raw_label in raw_concepts[:8]:
            normalized = _normalize_label(str(raw_label))
            if not _is_informative_label(normalized):
                continue
            concept_id = _concept_id(normalized)
            if concept_id not in scored_candidates:
                scored_candidates[concept_id] = {
                    "id": concept_id,
                    "label": normalized,
                    "score": 2.0,
                    "source_title": module.title,
                }

        ordered = sorted(
            scored_candidates.values(),
            key=lambda item: (-item["score"], len(item["label"]), item["label"]),
        )
        return ordered[:5]
    except Exception:
        return []


def _build_module_concept_cache_key(
    module: Module,
    source_titles: list[str],
    relevant_chunks: list[dict],
    model: str = DEFAULT_CONCEPT_EXTRACT_MODEL,
) -> str:
    """Fingerprint the extraction inputs for one module.

    If any heading, chunk content, or the extraction model changes, the cache
    key changes too and we rebuild concepts for that module only.
    """
    cache_payload = {
        "course_id": module.course_id,
        "module_id": module.module_id,
        "module_title": module.title,
        "chapter_title": module.chapter_title,
        "model": model,
        "source_titles": [title.strip() for title in source_titles if title and title.strip()],
        "chunks": [
            {
                "page_range": chunk.get("page_range"),
                "start_page": chunk.get("start_page"),
                "section": chunk.get("section"),
                "subsection": chunk.get("subsection"),
                "content": chunk.get("content"),
            }
            for chunk in relevant_chunks
        ],
    }
    serialized = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _serialize_cached_concepts(concepts: list[dict]) -> list[dict]:
    """Persist only the stable fields we need to reconstruct extracted concepts."""
    return [
        {
            "id": concept["id"],
            "label": concept["label"],
            "score": concept["score"],
            "source_title": concept["source_title"],
        }
        for concept in concepts
    ]


def _deserialize_cached_concepts(cached_concepts: list[dict]) -> list[dict]:
    """Normalize cached concept payloads back into the runtime shape."""
    concepts: list[dict] = []
    for concept in cached_concepts:
        concept_id = str(concept.get("id", "")).strip()
        label = str(concept.get("label", "")).strip()
        source_title = str(concept.get("source_title", "")).strip()
        if not concept_id or not label:
            continue
        concepts.append(
            {
                "id": concept_id,
                "label": label,
                "score": float(concept.get("score", 1.0)),
                "source_title": source_title or label,
            }
        )
    return concepts


def _extract_module_concepts_heuristic(
    module: Module,
    source_titles: list[str],
) -> list[dict]:
    """Fallback extractor used when the LLM path is unavailable."""
    scored_candidates: dict[str, dict] = {}

    for source_title in source_titles:
        source_weight = 1.0 if source_title != module.title else 2.0
        for label in _extract_concept_labels(source_title):
            concept_id = _concept_id(label)
            existing = scored_candidates.get(concept_id)
            if existing is None:
                scored_candidates[concept_id] = {
                    "id": concept_id,
                    "label": label,
                    "score": source_weight,
                    "source_title": source_title,
                }
            else:
                existing["score"] += source_weight

    ordered = sorted(
        scored_candidates.values(),
        key=lambda item: (-item["score"], len(item["label"]), item["label"]),
    )
    return ordered[:5]


def _module_subsection_titles(module: Module, chunks: list[dict]) -> list[str]:
    """Collect subsection headings that belong to one module only once."""
    seen: set[str] = set()
    titles: list[str] = []
    for chunk in _module_relevant_chunks(module, chunks):
        subsection = (chunk.get("subsection") or "").strip()
        if not subsection or subsection == module.title or subsection in seen:
            continue
        seen.add(subsection)
        titles.append(subsection)
    return titles


def _module_relevant_chunks(module: Module, chunks: list[dict]) -> list[dict]:
    """Return every chunk mapped to this module.

    The LLM extractor reads all of these chunks so concept selection is driven
    by the actual module content rather than headings alone.
    """
    relevant_chunks: list[dict] = []
    for chunk in chunks:
        if not _chunk_matches_module(chunk, module):
            continue
        relevant_chunks.append(chunk)
    return relevant_chunks


def _extract_concept_labels(title: str) -> list[str]:
    """Turn a heading into reusable concept labels.

    The goal is not perfect NLP. The goal is a stable, course-agnostic way to
    turn section-style headings into concept candidates.
    """
    cleaned = _clean_title(title)
    if not cleaned:
        return []

    candidates = [cleaned]
    for separator in SPLIT_SEPARATORS:
        next_candidates: list[str] = []
        for candidate in candidates:
            if separator in candidate:
                next_candidates.extend(part.strip() for part in candidate.split(separator))
            else:
                next_candidates.append(candidate)
        candidates = next_candidates

    # "and" sometimes separates two real concepts, but in pedagogical titles it
    # can also split phrases like "A Necessary and Sufficient Condition...".
    expanded_candidates: list[str] = []
    for candidate in candidates:
        expanded_candidates.append(candidate)
        expanded_candidates.extend(_split_and_candidate(candidate))
    candidates = expanded_candidates

    normalized_candidates: list[str] = []
    for candidate in candidates:
        candidate = _strip_prefix(candidate)
        normalized = _normalize_label(candidate)
        if not _is_informative_label(normalized):
            continue
        normalized_candidates.append(normalized)

    # Preserve order while removing duplicates.
    unique_labels = list(dict.fromkeys(normalized_candidates))
    return unique_labels


def _fallback_module_concept(module: Module) -> dict:
    """Guarantee that every module contributes at least one concept node."""
    label = _normalize_label(module.title) or module.title.strip().lower()
    return {
        "id": _concept_id(label),
        "label": label,
        "score": 1.0,
        "source_title": module.title,
    }


def _build_concept_prerequisite_edges(
    ordered_modules_by_course: dict[str, list[Module]],
    module_concepts: dict[str, list[dict]],
) -> list[dict]:
    """Project module order down to the primary concept of each neighboring module.

    This is a deliberately simple first pass: if module A tends to come before
    module B, we treat A's highest-ranked concept as a likely prerequisite for
    B's highest-ranked concept.
    """
    edges: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for modules in ordered_modules_by_course.values():
        for index, module in enumerate(modules):
            next_module = modules[index + 1] if index + 1 < len(modules) else None
            if next_module is None or module.chapter_title != next_module.chapter_title:
                continue

            left_primary = module_concepts.get(module.module_id, [])
            right_primary = module_concepts.get(next_module.module_id, [])
            if not left_primary or not right_primary:
                continue

            source_concept = left_primary[0]["id"]
            target_concept = right_primary[0]["id"]
            if source_concept == target_concept:
                continue

            pair = (source_concept, target_concept)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            edges.append(
                {
                    "source": source_concept,
                    "target": target_concept,
                    "relation": "concept_prerequisite_of",
                    "directed": True,
                    "weight": 0.85,
                    "reason": (
                        f"Primary concepts inherited the learning order between "
                        f"{module.module_id} and {next_module.module_id}."
                    ),
                }
            )

    return edges


def _build_concept_related_edges(
    module_concepts: dict[str, list[dict]],
) -> list[dict]:
    """Connect concepts that co-occur inside the same module.

    This is a soft undirected relation, not an ordering claim. It helps the
    concept graph stay connected when modules teach several related ideas.
    """
    pair_to_modules: dict[tuple[str, str], list[str]] = {}

    for module_id, concepts in module_concepts.items():
        concept_ids = list(dict.fromkeys(concept["id"] for concept in concepts))
        for index, left_id in enumerate(concept_ids):
            for right_id in concept_ids[index + 1 :]:
                if left_id == right_id:
                    continue
                pair = tuple(sorted((left_id, right_id)))
                pair_to_modules.setdefault(pair, []).append(module_id)

    edges: list[dict] = []
    for (left_id, right_id), module_ids in pair_to_modules.items():
        weight = min(0.45 + 0.15 * len(module_ids), 0.9)
        module_list = ", ".join(module_ids[:3])
        if len(module_ids) > 3:
            module_list += ", ..."
        edges.append(
            {
                "source": left_id,
                "target": right_id,
                "relation": "concept_related_to",
                "directed": False,
                "weight": round(weight, 2),
                "reason": f"Concepts co-occur in module(s): {module_list}",
            }
        )

    return edges


def _annotate_concept_core_scores(
    concept_nodes: dict[str, dict],
    edges: list[dict],
    module_count: int,
) -> None:
    """Estimate how central each concept is to the course.

    Phase 1 keeps the score simple and explainable:
    - concepts taught by many modules matter more
    - concepts that unlock later concepts matter more
    - concepts with many graph connections matter more
    """
    if not concept_nodes:
        return

    prerequisite_outgoing: dict[str, int] = {concept_id: 0 for concept_id in concept_nodes}
    prerequisite_incoming: dict[str, int] = {concept_id: 0 for concept_id in concept_nodes}
    related_degree: dict[str, int] = {concept_id: 0 for concept_id in concept_nodes}

    for edge in edges:
        if edge["relation"] == "concept_prerequisite_of":
            prerequisite_outgoing[edge["source"]] = prerequisite_outgoing.get(edge["source"], 0) + 1
            prerequisite_incoming[edge["target"]] = prerequisite_incoming.get(edge["target"], 0) + 1
        elif edge["relation"] == "concept_related_to":
            related_degree[edge["source"]] = related_degree.get(edge["source"], 0) + 1
            related_degree[edge["target"]] = related_degree.get(edge["target"], 0) + 1

    max_module_coverage = max(
        (len(node.get("module_ids", [])) for node in concept_nodes.values()),
        default=1,
    )
    max_prerequisite_outgoing = max(prerequisite_outgoing.values(), default=1)
    max_graph_degree = max(
        (
            prerequisite_outgoing.get(concept_id, 0)
            + prerequisite_incoming.get(concept_id, 0)
            + related_degree.get(concept_id, 0)
            for concept_id in concept_nodes
        ),
        default=1,
    )

    for concept_id, node in concept_nodes.items():
        module_hits = len(node.get("module_ids", []))
        module_coverage_score = (
            module_hits / max_module_coverage if max_module_coverage else 0.0
        )
        prerequisite_support_score = (
            prerequisite_outgoing.get(concept_id, 0) / max_prerequisite_outgoing
            if max_prerequisite_outgoing
            else 0.0
        )
        graph_degree = (
            prerequisite_outgoing.get(concept_id, 0)
            + prerequisite_incoming.get(concept_id, 0)
            + related_degree.get(concept_id, 0)
        )
        graph_connectivity_score = graph_degree / max_graph_degree if max_graph_degree else 0.0

        core_score = (
            0.5 * module_coverage_score
            + 0.3 * prerequisite_support_score
            + 0.2 * graph_connectivity_score
        )

        node["core_score"] = round(core_score, 2)
        node["importance"] = {
            "module_hits": module_hits,
            "module_count": module_count,
            "module_coverage_score": round(module_coverage_score, 2),
            "prerequisite_support_score": round(prerequisite_support_score, 2),
            "graph_connectivity_score": round(graph_connectivity_score, 2),
        }


def _get_course_index_dir(course: Course) -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    source_name = course.source_name or course.course_id
    return base_dir / "data" / "indexes" / source_name


def _chunk_matches_module(chunk: dict, module: Module) -> bool:
    # Reuse the same chapter/section/subsection mapping idea as the module graph
    # so both graph layers stay aligned to the same underlying content slices.
    chunk_chapter = (chunk.get("chapter") or "").strip().lower()
    module_chapter = (module.chapter_title or "").strip().lower()
    if chunk_chapter != module_chapter:
        return False

    chunk_section = (chunk.get("section") or "").strip().lower()
    chunk_subsection = (chunk.get("subsection") or "").strip().lower()
    module_title = module.title.strip().lower()
    return chunk_section == module_title or chunk_subsection == module_title


def _clean_title(title: str) -> str:
    """Normalize spacing first so later string rules behave predictably."""
    return WHITESPACE_PATTERN.sub(" ", title.replace("-", " ")).strip()


def _strip_prefix(title: str) -> str:
    """Remove pedagogical phrasing to keep the remaining label concept-like."""
    lowered = title.lower()
    for prefix in PEDAGOGICAL_PREFIXES:
        if lowered.startswith(prefix):
            return title[len(prefix):].strip()
    return title.strip()


def _normalize_label(label: str) -> str:
    """Lowercase and trim a label into a reusable canonical form."""
    cleaned = _clean_title(label)
    cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.lower().strip()


def _is_informative_label(label: str) -> bool:
    """Drop labels that look like teaching phrasing rather than domain concepts."""
    if not label or label in GENERIC_CONCEPT_LABELS:
        return False
    if label.startswith(("necessary ", "sufficient ", "further ", "basic ")):
        return False
    return len(label) >= 3 and any(char.isalpha() for char in label)


def _split_and_candidate(candidate: str) -> list[str]:
    """Split simple 'A and B' headings only when both halves still look useful."""
    if " and " not in candidate.lower():
        return []

    parts = [part.strip() for part in re.split(r"\band\b", candidate, flags=re.IGNORECASE)]
    if len(parts) != 2:
        return []
    if not all(_looks_like_subconcept(part) for part in parts):
        return []
    return parts


def _looks_like_subconcept(text: str) -> bool:
    """Check whether a split fragment is still concept-like on its own."""
    normalized = _normalize_label(text)
    if not normalized or normalized in GENERIC_CONCEPT_LABELS:
        return False
    if normalized.startswith(("necessary ", "sufficient ", "further ", "basic ")):
        return False
    return True


def _concept_id(label: str) -> str:
    """Global concept ids are text-derived so they can be reused across courses."""
    slug = NON_ALNUM_PATTERN.sub("_", label.lower()).strip("_")
    return f"concept:{slug}"


def _toc_sort_key(toc_number: str | None) -> tuple:
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
    scope_id = sys.argv[1] if len(sys.argv) > 1 else "math"
    if scope_id == "catalog":
        graph = build_catalog_concept_graph()
    else:
        graph = build_course_concept_graph_for_id(scope_id)

    print(f"Built concept graph for scope: {graph['scope_id']}")
    print(f"Modules: {graph['module_count']}")
    print(f"Concepts: {graph['concept_count']}")
    print(f"Edges: {graph['edge_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
