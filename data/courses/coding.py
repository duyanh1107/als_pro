from core.models import Course
from data.skills.code_skill import CODE_SKILLS
from data.skills.global_skill import GLOBAL_SKILLS


def _prefixed(skills: list[str], namespace: str) -> list[str]:
    return [f"{namespace}:{skill}" for skill in skills]


CODING_COURSE = Course(
    course_id="coding",
    title="Programming Fundamentals",
    description="Start with programming logic, decomposition, and code understanding.",
    primary_skill="code:conceptual_understanding",
    source_name=None,
    # Each course exposes its domain skills plus the shared global skill taxonomy.
    skills=_prefixed(CODE_SKILLS, "code") + _prefixed(GLOBAL_SKILLS, "global"),
)
