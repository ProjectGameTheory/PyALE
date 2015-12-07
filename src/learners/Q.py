from TDLearner import TDLearner
import numpy as np


class Q(TDLearner):

    def update_theta(self, alpha, reward, phi_ns):
        # temporal difference
        delta = reward - self.get_value(self.phi, self.action)
        action_ns, values_ns = self.select_action(phi_ns)
        action_ns_idx = self.idx_action(action_ns)
        delta += self.gamma * np.max(values_ns)
        update = alpha * delta * self.trace.values
        # reset traces if action not optimal
        if action_ns_idx not in self.max_actions_idx(values_ns):
            self.trace.reset()
        # update state and action
        self.save_phi_action(phi_ns, action_ns)
        return update
