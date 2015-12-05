from Policy import Policy
import numpy as np

class EGreedy(Policy):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def select_action(self, actions=[], values=[]):
        acts = np.arange(values.size)
        if (np.random.rand()< self.epsilon):
            action_idx = np.random.choice(acts)
        else:
            max_acts = acts[values == np.max(values)]
            action_idx = np.random.choice(max_acts)
        return actions[action_idx]
