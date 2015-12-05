import numpy as np

class Policy(object):
    def select_action(self, actions=[], values=[]):
        assert len(actions) == len(values), 'length of actions should be same as values'

    def max_indexes(self, actions, rewards):
        indexes = []
        m = np.max(rewards)
        for i in range(len(rewards)):
            if rewards[i] == m:
                indexes.append(actions[i])
        return indexes