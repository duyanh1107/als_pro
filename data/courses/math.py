from core.models import Course
from data.skills.global_skill import GLOBAL_SKILLS
from data.skills.math_skill import MATH_SKILLS


def _prefixed(skills: list[str], namespace: str) -> list[str]:
    return [f"{namespace}:{skill}" for skill in skills]


MATH_COURSE = Course(
    course_id="math",
    title="Linear Algebra Foundations",
    description="Start with matrices, row operations, and vector spaces.",
    primary_skill="math:conceptual_understanding",
    source_name="linear_algebra",
    # Each course exposes its domain skills plus the shared global skill taxonomy.
    skills=_prefixed(MATH_SKILLS, "math") + _prefixed(GLOBAL_SKILLS, "global"),
)
