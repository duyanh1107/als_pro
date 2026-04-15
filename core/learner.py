from core.bkt import BKTModel


class Learner:
    """
    Multi-skill learner using BKT per skill

    Each skill (e.g. "math:procedural_fluency") has its own BKT model.
    The learner does NOT have a single global level anymore.
    Instead, we track mastery per skill.
    """

    def __init__(self, learner_id: int):
        self.id = learner_id

        # Store BKT model per skill
        # Example:
        # {
        #   "math:procedural_fluency": BKTModel(),
        #   "global:logical_reasoning": BKTModel()
        # }
        self.skills = {}

        # Simple counters (not used by BKT, only for stats/debug)
        self.correct_count = 0
        self.wrong_count = 0

    def _get_or_create_bkt(self, skill: str) -> BKTModel:
        """
        Get BKT model for a skill.
        If it does not exist yet → create a new one.

        This allows dynamic skill creation (no need to predefine all skills).
        """
        if skill not in self.skills:
            self.skills[skill] = BKTModel()
        return self.skills[skill]

    def update(self, correct: bool, skill_tags: list[str]) -> dict:
        """
        Update learner after answering ONE question.

        Flow:
        1. Receive result (correct / wrong)
        2. Question is tagged with multiple skills
           (e.g. ["math:procedural_fluency", "global:logical_reasoning"])
        3. For EACH skill:
            → update its BKT model using the same observation
        4. Return updated mastery per skill

        Important:
        - One question can affect multiple skills
        - Each skill is updated independently
        """

        # Update simple counters (for reporting only)
        if correct:
            self.correct_count += 1
        else:
            self.wrong_count += 1

        mastery_updates = {}

        # 🔥 Core idea: update EACH skill separately
        for skill in skill_tags:
            bkt = self._get_or_create_bkt(skill)

            # BKT update = Bayesian update of mastery probability
            mastery = bkt.update(correct)

            # Store result
            mastery_updates[skill] = mastery

        return mastery_updates

    def get_mastery(self, skill: str) -> float:
        """
        Get current mastery (P(Know)) for a specific skill.

        If skill not seen before → it will be initialized automatically.
        """
        return self._get_or_create_bkt(skill).p_know

    def get_all_mastery(self) -> dict:
        """
        Return mastery for all learned skills.

        Example output:
        {
            "math:procedural_fluency": 0.72,
            "global:logical_reasoning": 0.55
        }
        """
        return {k: v.p_know for k, v in self.skills.items()}

    def to_dict(self) -> dict:
        return {
            "learner_id": self.id,
            "correct_count": self.correct_count,
            "wrong_count": self.wrong_count,
            "skills": {
                skill: {
                    "p_know": model.p_know,
                    "p_learn": model.p_learn,
                    "p_guess": model.p_guess,
                    "p_slip": model.p_slip,
                }
                for skill, model in self.skills.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Learner":
        learner = cls(data["learner_id"])
        learner.correct_count = data.get("correct_count", 0)
        learner.wrong_count = data.get("wrong_count", 0)

        for skill, state in data.get("skills", {}).items():
            learner.skills[skill] = BKTModel(
                p_init=state.get("p_know", 0.3),
                p_learn=state.get("p_learn", 0.05),
                p_guess=state.get("p_guess", 0.25),
                p_slip=state.get("p_slip", 0.1),
            )

        return learner
