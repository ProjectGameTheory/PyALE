from TDLearner import TDLearner

class Sarsa(TDLearner):

    def update_theta(self, alpha, reward, phi_ns):
        # temporal difference
        delta = reward - self.get_value(self.phi, self.action)
        action_ns, values_ns = self.select_action(phi_ns)
        action_ns_idx = self.idx_action(action_ns)
        delta += self.gamma * values_ns[action_ns_idx]
        update = alpha * delta * self.trace.values
        # update state and action
        self.save_phi_action(phi_ns, action_ns)
        return update