from Learner import Learner

class RewardShaping(Learner):
    def __init__(self, td_learner=None, shaper=None):
        self.learner = td_learner
        self.shaper = shaper

    def shaping_reward(self, state, state_ns):
        potential_s = self.shaper.potential(state)
        potential_ns = self.shaper.potential(state_ns)
        return self.learner.gamma*potential_ns - potential_s