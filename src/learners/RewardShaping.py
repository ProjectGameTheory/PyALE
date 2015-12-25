from monitors.Event import episode_ended
from Learner import Learner


class RewardShaping(Learner):

    def __init__(self, learner=None, shaper=None):
        super(RewardShaping, self).__init__()
        self.learner = learner
        self.shaper = shaper

    def reset(self):
        self.learner.reset()

    def shaping_reward(self, state, state_ns):
        potential_s = self.shaper.potential(state)
        potential_ns = self.shaper.potential(state_ns)
        return self.learner.gamma * potential_ns - potential_s

    def start(self, state):
        super(RewardShaping, self).start(state)
        self.shaper.start()
        return self.learner.start(state)

    def step(self, reward, state, time):
        phi_ns = self.learner.features.phi(state)
        s = self.shaping_reward(self.learner.phi, phi_ns)
        reward += s
        super(RewardShaping, self).step(reward, state, time)
        return self.learner.step(reward, state, time)

    def end(self, trial, episode, reward):
        super(RewardShaping, self).end(trial, episode, reward)
        self.learner.end(trial, episode, reward)
        #episode_ended.notify(self.num_steps, self.tot_reward)
