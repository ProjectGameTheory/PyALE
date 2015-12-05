from Learner import Learner
import numpy as np


class Sarsa(Learner):

    def start(self, state):
        super(Sarsa, self).start(state)
        phi_ns = self.features.phi(state)

        values_ns = self.get_all_values(phi_ns)
        action_ns = self.policy.select_action(
            actions=self.actions, values=values_ns)
        # update state and action
        self.save_phi_action(phi_ns, action_ns)
        return action_ns

    def step(self, reward, state):
        super(Sarsa, self).step(reward, state)
        alpha = self.alpha
        if self.normalization:
            reward = self.normalize_reward(reward)
            alpha = self.normalize_alpha(self.phi)

        phi_ns = self.features.phi(state)
        self.trace.update(self.phi, self.action, self.gamma)
        delta = reward - self.get_value(self.phi, self.action)
        values_ns = self.get_all_values(phi_ns)
        action_ns = self.policy.select_action(
            actions=self.actions, values=values_ns)
        action_ns_idx = np.where(self.actions == action_ns)[0][0]
        delta += self.gamma * values_ns[action_ns_idx]
        # update weights
        self.theta += alpha * delta * self.trace.values
        # update state and action
        self.save_phi_action(phi_ns, action_ns)
        return action_ns

    def end(self, reward):
        super(Sarsa, self).end(reward)
        alpha = self.alpha
        if self.normalization:
            reward = self.normalize_reward(reward)
            alpha = self.normalize_alpha(self.phi)

        self.trace.update(self.phi, self.action, self.gamma)
        delta = reward - self.get_value(self.phi, self.action)
        # update weights
        self.theta += alpha * delta * self.trace.values
