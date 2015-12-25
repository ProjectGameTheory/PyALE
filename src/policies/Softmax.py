from Policy import Policy
import numpy as np

class Softmax(Policy):
    def __init__(self, temperature=0.0):
        self.temperature = temperature

    def select_action(self, actions=[], values=[],**kwargs):
        total = sum([np.exp(float(v) / self.temperature) for v in values])
        probs = [np.exp(float(v) / self.temperature) / total for v in values]
        return np.random.choice(range(len(values)), p=probs)
