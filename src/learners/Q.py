from Learner import Learner
import numpy as np


class Q(Learner):

    def start(self, state):
        # reset trace
        super(Q, self).start(state)
        phi_ns = self.features.phi(state)
        values_ns = self.get_all_values(phi_ns)
        action_ns = self.policy.select_action(
            actions=self.actions, values=values_ns)
        # update state and action
        self.save_phi_action(phi_ns, action_ns)
        return action_ns

    def step(self, reward, state):
        super(Q, self).step(reward, state)
        alpha = self.alpha
        if self.normalization:
            reward = self.normalize_reward(reward)
            alpha = self.normalize_alpha(self.phi)
        
        self.trace.update(self.phi, self.action, self.gamma)
        # temporal difference
        delta = reward - self.get_value(self.phi, self.action)
        # next action
        # features from state
        phi_ns = self.features.phi(state)
        values_ns = self.get_all_values(phi_ns)
        action_ns = self.policy.select_action(
            actions=self.actions, values=values_ns)
        action_ns_idx = np.where(self.actions == action_ns)[0][0]
        delta += self.gamma * np.max(values_ns)
        max_actions = np.arange(values_ns.size)[values_ns == np.max(values_ns)]
        # td update
        self.theta += alpha * delta * self.trace.values
        # reset traces if action not optimal
        if action_ns_idx not in max_actions:
            self.trace.reset()
        # update state and action
        self.save_phi_action(phi_ns, action_ns)
        return action_ns

    def end(self, reward):
        super(Q, self).end(reward)
        alpha = self.alpha
        if self.normalization:
            reward = self.normalize_reward(reward)
            alpha = self.normalize_alpha(self.phi)
        self.trace.update(self.phi, self.action, self.gamma)
        #reward = self.normalize_reward(reward)
        delta = reward - self.get_value(self.phi, self.action)
        # update weights
        self.theta += alpha * delta * self.trace.values
