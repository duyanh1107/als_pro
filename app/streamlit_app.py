from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - only triggered when the optional UI dependency is missing.
    raise SystemExit(
        "Streamlit is not installed. Install it with `pip install streamlit` to run the teacher review app."
    ) from exc

from llm.lesson_generator import get_mastery_band
from services.content_service import get_course
from services.lesson_store import delete_lesson_draft
from services.lesson_store import list_lesson_drafts
from services.lesson_store import save_lesson_draft


st.set_page_config(page_title="Adaptive Learning AI Teacher Review", layout="wide")

ALLOWED_STATUSES = ["draft", "reviewed", "approved", "archived"]


def main() -> None:
    st.title("Teacher Lesson Review")
    st.caption("Review RAG-grounded lesson drafts, inspect source chunks, and edit before release.")

    drafts = list_lesson_drafts()
    if not drafts:
        st.warning("No lesson drafts found yet. Generate one from teacher mode in main.py first.")
        return

    draft_options = {
        format_draft_label(draft): draft
        for draft in drafts
    }
    st.sidebar.header("Draft Picker")
    selected_label = st.sidebar.selectbox("Lesson draft", list(draft_options.keys()))
    draft = draft_options[selected_label]
    st.sidebar.caption("Only `reviewed` and `approved` drafts are served to learners.")
    st.sidebar.write(f"Status: `{draft.status}`")
    st.sidebar.write(f"Chunks: `{len(draft.source_chunks)}`")

    course = get_course(draft.course_id)
    allowed_skills = list(course.skills) if course is not None else []
    if draft.primary_skill and draft.primary_skill not in allowed_skills:
        allowed_skills.append(draft.primary_skill)
    ai_weight_source = draft.ai_recommended_skill_weights or build_default_skill_weights(
        allowed_skills,
        draft.primary_skill,
    )
    weight_source = draft.skill_weights or ai_weight_source
    if not draft.ai_recommended_skill_weights:
        draft.ai_recommended_skill_weights = dict(ai_weight_source)
    if not draft.skill_weights:
        draft.skill_weights = dict(weight_source)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Course", draft.course_id)
    metric_cols[1].metric("Mastery Band", draft.mastery_band)
    metric_cols[2].metric("Source Chunks", len(draft.source_chunks))
    metric_cols[3].metric("Status", draft.status)

    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("Module Settings")
            st.caption("Edit the metadata that controls how this lesson is labeled and delivered.")
            st.write(f"Course ID: `{draft.course_id}`")
            st.write(f"Module ID: `{draft.module_id}`")

            module_title = st.text_input("Module title", value=draft.module_title)
            chapter_title = st.text_input("Chapter title", value=draft.chapter_title or "")
            if allowed_skills:
                primary_skill = st.selectbox(
                    "Primary skill",
                    options=allowed_skills,
                    index=allowed_skills.index(draft.primary_skill) if draft.primary_skill in allowed_skills else 0,
                    help="Main label for the module. Learner pacing falls back to this when skill weights are missing.",
                )
            else:
                st.warning("No backend skill list is configured for this course. Primary skill is shown read-only.")
                primary_skill = draft.primary_skill
                st.code(primary_skill or "Unknown")

            settings_cols = st.columns(2)
            with settings_cols[0]:
                mastery_score = st.slider(
                    "Target mastery score",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(draft.mastery_score),
                    step=0.05,
                    help="This controls which learner band this draft is intended for.",
                )
                derived_mastery_band = get_mastery_band(mastery_score)
                st.caption(f"Derived mastery band: `{derived_mastery_band}`")
            with settings_cols[1]:
                status = st.selectbox(
                    "Status",
                    options=ALLOWED_STATUSES,
                    index=ALLOWED_STATUSES.index(draft.status) if draft.status in ALLOWED_STATUSES else 0,
                    help="Only reviewed and approved drafts are used for learner delivery.",
                )

        with st.container(border=True):
            st.subheader("Skill Weights")
            st.caption(
                "AI suggests independent relevance scores from 0 to 1. "
                "Teachers can override them to reflect what this module really trains."
            )
            weight_inputs: dict[str, float] = {}
            for skill in allowed_skills:
                ai_weight = float(ai_weight_source.get(skill, 0.0))
                default_weight = float(weight_source.get(skill, 0.0))
                weight_inputs[skill] = st.slider(
                    label=skill,
                    min_value=0.0,
                    max_value=1.0,
                    value=min(1.0, round(default_weight, 2)),
                    step=0.05,
                    help=f"AI recommended: {ai_weight:.2f}",
                )

            st.caption(
                "These weights do not need to sum to 1. "
                "The backend uses them as independent importance signals for weighted module mastery."
            )

    with right_col:
        lesson_tab, source_tab = st.tabs(["Lesson Review", "Source Evidence"])

        with lesson_tab:
            top_tabs = st.tabs(["Generated", "Edited Preview"])
            with top_tabs[0]:
                source_mode = st.radio(
                    "View mode",
                    options=["Rendered", "Raw"],
                    horizontal=True,
                    key="generated_view_mode",
                )
                if source_mode == "Rendered":
                    render_lesson_content(draft.generated_lesson)
                else:
                    st.text_area(
                        "Original generated lesson",
                        value=draft.generated_lesson,
                        height=260,
                        disabled=True,
                    )

            edited_text = st.text_area(
                "Teacher-edited lesson",
                value=draft.edited_lesson or draft.generated_lesson,
                height=420,
            )

            with top_tabs[1]:
                render_lesson_content(edited_text)

        with source_tab:
            st.caption("Inspect the retrieved evidence used to build this lesson draft.")
            # The saved draft already includes source snapshots so review does not need
            # another retrieval/indexing pass just to inspect provenance.
            for index, chunk in enumerate(draft.source_chunks, start=1):
                label = chunk.get("subsection") or chunk.get("section") or f"Chunk {index}"
                with st.expander(f"{index}. {label}", expanded=index == 1):
                    st.caption(
                        f"Chapter: {chunk.get('chapter') or 'Unknown'} | "
                        f"Page: {chunk.get('start_page') if chunk.get('start_page') is not None else 'Unknown'}"
                    )
                    source_tabs = st.tabs(["Rendered", "Raw"])
                    with source_tabs[0]:
                        render_lesson_content(chunk.get("content") or chunk.get("preview") or "")
                    with source_tabs[1]:
                        st.text_area(
                            f"Source preview {index}",
                            value=chunk.get("content") or chunk.get("preview") or "",
                            height=220,
                            disabled=True,
                            key=f"source_{index}",
                        )

        if st.button("Save lesson draft", type="primary", use_container_width=True):
            # Saving only updates the reviewed draft on disk; learner delivery will
            # automatically pick up the edited text on the next run.
            old_mastery_band = draft.mastery_band
            draft.module_title = module_title.strip() or draft.module_title
            draft.chapter_title = chapter_title.strip() or None
            draft.primary_skill = primary_skill.strip() or draft.primary_skill
            draft.mastery_score = mastery_score
            draft.mastery_band = derived_mastery_band
            draft.draft_id = f"{draft.course_id}:{draft.module_id}:{draft.mastery_band}"
            draft.ai_recommended_skill_weights = clamp_weight_inputs(ai_weight_source)
            draft.skill_weights = clamp_weight_inputs(weight_inputs)
            draft.edited_lesson = edited_text
            draft.status = status
            if old_mastery_band != draft.mastery_band:
                # A mastery-band change means this draft now belongs to a different
                # learner delivery bucket, so remove the old file before saving.
                delete_lesson_draft(draft.course_id, draft.module_id, old_mastery_band)
            save_lesson_draft(draft)
            st.success("Lesson draft saved.")


def render_lesson_content(text: str) -> None:
    if not text.strip():
        st.caption("No content to render.")
        return

    normalized = normalize_math_delimiters(text)

    # Split on block LaTeX so math expressions render with st.latex while the
    # surrounding lesson text still renders as normal Markdown.
    parts = normalized.split("$$")
    for index, part in enumerate(parts):
        cleaned = part.strip()
        if not cleaned:
            continue

        if index % 2 == 1:
            st.latex(normalize_latex_block(cleaned))
        else:
            st.markdown(cleaned)


def normalize_math_delimiters(text: str) -> str:
    normalized = text

    # Support common LLM output that uses \[ ... \] instead of $$ ... $$.
    normalized = normalized.replace(r"\[", "$$").replace(r"\]", "$$")
    normalized = normalized.replace(r"\(", "$").replace(r"\)", "$")

    # Support bare [ ... ] display blocks when the content looks like actual math.
    normalized = re.sub(
        r"(?ms)^\[(.+?)\]$",
        lambda match: f"$${match.group(1).strip()}$$" if looks_like_math(match.group(1)) else match.group(0),
        normalized,
    )
    return normalized


def looks_like_math(text: str) -> bool:
    math_markers = (
        r"\begin",
        r"\frac",
        r"\times",
        r"\cdot",
        "=",
        "&",
        "^",
        "_",
        "pmatrix",
        "bmatrix",
    )
    return any(marker in text for marker in math_markers)


def normalize_latex_block(text: str) -> str:
    normalized = text.strip()

    # Some generated lessons collapse matrix row breaks to a single backslash.
    # Convert those separators to `\\` so MathJax can render matrix environments.
    normalized = re.sub(r"(?<!\\)\\(?![\\a-zA-Z])", r"\\\\", normalized)
    normalized = re.sub(r"\s\\\s", r" \\\\ ", normalized)
    return normalized


def build_default_skill_weights(allowed_skills: list[str], primary_skill: str) -> dict[str, float]:
    if not allowed_skills:
        return {}
    if len(allowed_skills) == 1:
        return {allowed_skills[0]: 1.0}

    weights = {skill: 0.0 for skill in allowed_skills}
    if primary_skill in weights:
        weights[primary_skill] = 0.9
    remaining_skills = [skill for skill in allowed_skills if skill != primary_skill]
    if remaining_skills:
        for skill in remaining_skills:
            weights[skill] = 0.35 if skill.startswith("global:") else 0.6
    return weights


def clamp_weight_inputs(weight_inputs: dict[str, float]) -> dict[str, float]:
    return {skill: max(0.0, min(1.0, float(value))) for skill, value in weight_inputs.items()}


def format_draft_label(draft) -> str:
    return f"{draft.course_id} | {draft.module_title} | {draft.mastery_band}"


if __name__ == "__main__":
    main()
