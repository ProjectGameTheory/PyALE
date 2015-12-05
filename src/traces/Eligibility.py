import numpy as np

class Eligibility(object):
    def __init__(self, lambda_=0.9, actions=None, shape=None):
        self.lambda_ = lambda_
        self.shape = shape
        self.actions = np.array(actions)
        self.values = np.zeros(self.shape)
        self.reset()

    def reset(self):
        self.values *= 0

    def update(self, state, action, gamma):
        action_idx = np.where(self.actions == action)[0][0]
        self.values *= gamma*self.lambda_
        #state must be binary
        #non sparse update
        self.values[:,action_idx] += state
        #sparse update
        #self.values[state, action_idx] += 1
        #keep trace vector sparse
        self.values[self.values < 1e-4] = 0.
        self.values = np.clip(self.values,0.,5.)