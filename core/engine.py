class DecisionEngine:
    """
    Skill-aware decision engine
    Use learner mastery to control difficulty, hints, pacing
    """

    def __init__(self):
        pass

    def get_skill_mastery(self, learner, skill: str) -> float:
        return learner.get_mastery(skill)

    def get_difficulty(self, learner, skill: str) -> int:
        """
        Map mastery → difficulty (1-5)
        """
        mastery = learner.get_mastery(skill)

        # simple mapping
        difficulty = round(mastery * 5)

        return max(1, min(5, difficulty))

    def should_give_hint(self, learner, skill: str, attempt: int) -> bool:
        """
        Decide if hint should be shown
        """
        mastery = learner.get_mastery(skill)

        # low mastery → give hint earlier
        if mastery < 0.4:
            return True

        # otherwise only after 1 wrong attempt
        return attempt >= 1

    def should_skip(self, learner, skill: str) -> bool:
        """
        Skip if learner already strong
        """
        mastery = learner.get_mastery(skill)
        return mastery > 0.85

    def select_primary_skill(self, question):
        """
        Pick main skill of a question (for difficulty control)

        Assumption: first skill is primary
        """
        return question.skills[0]