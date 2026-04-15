# BKT estimates a learner’s mastery probability for a skill over time,
# updating belief from noisy correct/incorrect answers (accounting for guess and slip).

class BKTModel:
    """
    Bayesian Knowledge Tracing for a single skill
    """

    def __init__(
        self,
        p_init: float = 0.3,
        p_learn: float = 0.05,
        p_guess: float = 0.25,
        p_slip: float = 0.1,
    ):
        self.p_know = p_init
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip

    def update(self, correct: bool) -> float:
        p = self.p_know

        if correct:
            num = p * (1 - self.p_slip)
            den = num + (1 - p) * self.p_guess
        else:
            num = p * self.p_slip
            den = num + (1 - p) * (1 - self.p_guess)

        p_given_obs = num / den

        # learning transition
        self.p_know = p_given_obs + (1 - p_given_obs) * self.p_learn

        return self.p_know